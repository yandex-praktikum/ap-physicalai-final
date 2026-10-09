"""Real PPO training; no scripted controller or distillation in this module."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from physical_ai.env import DISCOUNT, REWARD_VERSION, ManipulationEnv
from physical_ai.scenes import ROBOTS, TASKS
from physical_ai.reset_states import bank_digest


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rl(path, robot, task, **load_kwargs):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if (meta["robot"], meta["task"]) != (robot, task):
        raise ValueError("Checkpoint robot/task mismatch")
    if meta.get("algorithm") != "PPO" or meta.get("schema") != 1:
        raise ValueError("Expected a schema-1 PPO checkpoint")
    if meta["sha256"] != sha256(path):
        raise ValueError("Checkpoint hash mismatch")
    return PPO.load(path, device="cpu", **load_kwargs), meta


def transfer_parameters(source, target, source_reference=None):
    """Copy actor/critic weights; preserve old features when adding a distractor."""
    old = source.policy.state_dict()
    new = target.policy.state_dict()
    source_raw = source.observation_space.shape[0]
    target_raw = target.observation_space.shape[0]
    from physical_ai.curriculum import BoundedPolicy

    relative = isinstance(source.policy, BoundedPolicy)
    cross_robot = relative and (
        source.policy.features_extractor.proprio_dim
        != target.policy.features_extractor.proprio_dim
    )
    adjustments = {}
    if cross_robot:
        if source_reference is None:
            raise ValueError(
                "Cross-robot transfer requires a source reference observation"
            )
        sp = source.policy.features_extractor.proprio_dim
        tp = target.policy.features_extractor.proprio_dim
        sq, tq = sp // 2, tp // 2
        source_cols = (
            list(range(sq - 2, sq))
            + list(range(sp - 2, sp))
            + list(range(sp, source_raw))
        )
        target_cols = (
            list(range(tq - 2, tq))
            + list(range(tp - 2, tp))
            + list(range(tp, tp + source_raw - sp))
        )
        source_cols += list(
            range(source_raw, source.policy.features_extractor.features_dim)
        )
        target_cols += list(
            range(
                target_raw,
                target_raw + source.policy.features_extractor.features_dim - source_raw,
            )
        )
        removed = list(range(sq - 2)) + list(range(sq, sp - 2))
        with torch.no_grad():
            reference = source.policy.features_extractor(
                torch.as_tensor(source_reference[None], dtype=torch.float32)
            )[0]
    for name, value in old.items():
        if name not in new:
            raise ValueError(f"Incompatible transfer parameter: {name}")
        if name == "action_net.2.scale":
            continue  # The target's explicit Cartesian scale is authoritative.
        if cross_robot and name in (
            "mlp_extractor.policy_net.0.weight",
            "mlp_extractor.value_net.0.weight",
        ):
            expanded = torch.zeros_like(new[name])
            expanded[:, target_cols] = value[:, source_cols]
            new[name] = expanded
            adjustments[name.replace("weight", "bias")] = (
                value[:, removed] @ reference[removed]
            )
        elif value.shape == new[name].shape:
            new[name] = value
        elif name == "log_std" and target.use_sde and not source.use_sde:
            # gSDE has one scale per latent feature/action, unlike diagonal noise.
            # Keep its explicit initialization while transferring the mean actor.
            continue
        elif (
            name
            in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight")
            and target_raw >= source_raw
        ):
            expanded = torch.zeros_like(new[name])
            expanded[:, :source_raw] = value[:, :source_raw]
            if relative:
                expanded[:, target_raw : target_raw + value.shape[1] - source_raw] = (
                    value[:, source_raw:]
                )
            new[name] = expanded
        else:
            raise ValueError(f"Incompatible transfer shape: {name}")
    for name, adjustment in adjustments.items():
        new[name] = new[name] + adjustment
    target.policy.load_state_dict(new)


def train(
    robot,
    task,
    out,
    total_steps=2000000,
    n_steps=1024,
    n_envs=4,
    seed=0,
    learning_rate=3e-4,
    ent_coef=0.01,
    reward_stage="place",
    resume=None,
    parameters=None,
    checkpoint_every=250000,
    curriculum_bank=None,
    curriculum_phase=236,
    normal_fraction=0.1,
    bounded_policy=False,
    reset_std=None,
    max_steps=1000,
    transfer=None,
    use_sde=False,
    sde_sample_freq=8,
    curriculum_window=0,
    actor_head_scale=1.0,
    cartesian_action_scale=1.0,
    exploration_scale=1.0,
):
    if total_steps <= 0 or n_steps < 2 or n_envs < 1:
        raise ValueError("Invalid training budget")
    if resume and transfer:
        raise ValueError("Choose resume or transfer, not both")
    if not 0 < actor_head_scale <= 1:
        raise ValueError("actor_head_scale must be in (0, 1]")
    if not 0 < cartesian_action_scale <= 1:
        raise ValueError("cartesian_action_scale must be in (0, 1]")
    if not np.isfinite(exploration_scale) or exploration_scale <= 0:
        raise ValueError("exploration_scale must be finite and positive")
    torch.set_num_threads(1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.zip").exists():
        raise FileExistsError(
            "Use a new run directory; --resume reads a prior checkpoint"
        )
    kwargs = dict(
        robot=robot, task=task, parameters=parameters, reward_stage=reward_stage
    )

    def make():
        if curriculum_bank:
            from physical_ai.curriculum import CurriculumEnv

            return Monitor(
                CurriculumEnv(
                    **kwargs,
                    max_steps=max_steps,
                    state_bank=curriculum_bank,
                    phase=curriculum_phase,
                    normal_fraction=normal_fraction,
                    phase_window=curriculum_window,
                )
            )
        return Monitor(ManipulationEnv(**kwargs, max_steps=max_steps))

    env = (
        SubprocVecEnv([make for _ in range(n_envs)], start_method="spawn")
        if n_envs > 1
        else DummyVecEnv([make])
    )
    env.seed(seed)
    config = dict(
        robot=robot,
        task=task,
        total_steps=total_steps,
        n_steps=n_steps,
        n_envs=n_envs,
        seed=seed,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        reward_stage=reward_stage,
        parameters=parameters or {},
        resume=str(resume) if resume else None,
        gamma=DISCOUNT,
        reward_version=REWARD_VERSION,
        checkpoint_every=checkpoint_every,
        curriculum_bank=str(curriculum_bank) if curriculum_bank else None,
        curriculum_phase=curriculum_phase if curriculum_bank else None,
        curriculum_bank_sha256=(
            bank_digest(curriculum_bank) if curriculum_bank else None
        ),
        normal_fraction=normal_fraction,
        bounded_policy=bounded_policy,
        reset_std=reset_std,
        max_steps=max_steps,
        transfer=str(transfer) if transfer else None,
        scenes_sha256=sha256(Path(__file__).with_name("scenes.py")),
        environment_sha256=sha256(Path(__file__).with_name("env.py")),
        trainer_sha256=sha256(Path(__file__)),
        policy_source_sha256=sha256(Path(__file__).with_name("curriculum.py")),
        use_sde=use_sde,
        sde_sample_freq=sde_sample_freq,
        curriculum_window=curriculum_window,
        actor_head_scale=actor_head_scale,
        cartesian_action_scale=cartesian_action_scale,
        exploration_scale=exploration_scale,
    )
    try:
        if resume:
            policy, _ = load_rl(
                resume,
                robot,
                task,
                env=env,
                gamma=DISCOUNT,
                n_steps=n_steps,
                seed=seed,
                learning_rate=learning_rate,
                ent_coef=ent_coef,
                sde_sample_freq=sde_sample_freq,
                batch_size=min(256, n_steps * n_envs),
            )
        else:
            from physical_ai.curriculum import (
                BoundedPolicy,
                ScaledPolicy,
                ManipulationFeatures,
            )

            policy_kwargs = {"net_arch": dict(pi=[256, 256], vf=[256, 256])}
            if bounded_policy or cartesian_action_scale != 1:
                policy_kwargs = dict(
                    net_arch=dict(pi=[128, 128], vf=[128, 128]),
                    log_std_init=-1.0,
                    features_extractor_class=ManipulationFeatures,
                    features_extractor_kwargs={
                        "proprio_dim": 2 * (ROBOTS[robot][2] + 2)
                    },
                )
            source = None
            if transfer:
                source_meta = json.loads(
                    Path(transfer).with_suffix(".json").read_text()
                )
                source, source_meta = load_rl(
                    transfer, source_meta["robot"], source_meta["task"]
                )
                policy_kwargs = copy.deepcopy(source.policy_kwargs)
                if source_meta["robot"] != robot:
                    if not isinstance(source.policy, BoundedPolicy):
                        raise ValueError(
                            "Cross-robot transfer requires ManipulationFeatures"
                        )
                    policy_kwargs["features_extractor_kwargs"]["proprio_dim"] = 2 * (
                        ROBOTS[robot][2] + 2
                    )
                if use_sde and not source.use_sde:
                    policy_kwargs["log_std_init"] = -3.0
                config["transfer_source"] = {
                    key: source_meta[key]
                    for key in ["robot", "task", "timesteps", "sha256"]
                }
            policy_class = (
                type(source.policy)
                if source is not None
                else BoundedPolicy if bounded_policy else "MlpPolicy"
            )
            if cartesian_action_scale != 1:
                if source is not None and not isinstance(source.policy, BoundedPolicy):
                    raise ValueError("Scaled transfer requires a bounded source policy")
                policy_class = ScaledPolicy
                policy_kwargs["cartesian_scale"] = cartesian_action_scale
            policy = PPO(
                policy_class,
                env,
                n_steps=n_steps,
                batch_size=min(256, n_steps * n_envs),
                n_epochs=10,
                learning_rate=learning_rate,
                ent_coef=ent_coef,
                gamma=DISCOUNT,
                policy_kwargs=policy_kwargs,
                seed=seed,
                device="cpu",
                verbose=0,
                use_sde=use_sde or bool(source is not None and source.use_sde),
                sde_sample_freq=sde_sample_freq,
            )
            if source is not None:
                source_reference = None
                if source_meta["robot"] != robot:
                    with ManipulationEnv(
                        source_meta["robot"], source_meta["task"]
                    ) as reference_env:
                        source_reference, _ = reference_env.reset(seed=0)
                    config["transfer_mode"] = (
                        "reference_joint_task_space_initialization"
                    )
                    config["source_reference"] = source_reference.tolist()
                transfer_parameters(source, policy, source_reference=source_reference)
                scale_ratio = getattr(policy.policy, "cartesian_scale", 1.0) / getattr(
                    source.policy, "cartesian_scale", 1.0
                )
                if scale_ratio != 1:
                    with torch.no_grad():
                        policy.policy.log_std[..., :3].add_(np.log(scale_ratio))
                    config["cartesian_noise_scale_ratio"] = scale_ratio
        if actor_head_scale != 1:
            head = next(
                layer
                for layer in policy.policy.action_net.modules()
                if isinstance(layer, torch.nn.Linear)
            )
            with torch.no_grad():
                head.weight.mul_(actor_head_scale)
                head.bias.mul_(actor_head_scale)
        if reset_std is not None:
            if policy.use_sde:
                raise ValueError(
                    "reset_std is only for diagonal Gaussian action noise, not gSDE"
                )
            std = np.asarray(reset_std, dtype=float)
            if std.shape != (4,) or not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError("reset_std requires four finite positive values")
            with torch.no_grad():
                policy.policy.log_std.copy_(
                    torch.log(torch.as_tensor(std, dtype=torch.float32))
                )
        if exploration_scale != 1:
            with torch.no_grad():
                policy.policy.log_std.add_(np.log(exploration_scale))
        config["policy_class"] = type(policy.policy).__name__
        config["cartesian_action_scale"] = getattr(
            policy.policy, "cartesian_scale", 1.0
        )
        config["use_sde"] = policy.use_sde
        config["sde_sample_freq"] = policy.sde_sample_freq
        config["noise_scale_kind"] = "latent_feature" if policy.use_sde else "action"
        config["initial_action_std"] = policy.policy.log_std.detach().exp().tolist()
        (out / "config.json").write_text(json.dumps(config, indent=2))
        policy.set_logger(configure(str(out), ["csv", "tensorboard"]))
        before = torch.cat(
            [p.detach().flatten().cpu() for p in policy.policy.parameters()]
        )

        def save_checkpoint(stem):
            path = out / (stem + ".zip")
            policy.save(path)
            after = torch.cat(
                [p.detach().flatten().cpu() for p in policy.policy.parameters()]
            )
            meta = dict(
                schema=1,
                algorithm="PPO",
                robot=robot,
                task=task,
                timesteps=policy.num_timesteps,
                parameter_delta_l2=float(torch.linalg.vector_norm(after - before)),
                seed=seed,
                reward_stage=reward_stage,
                sha256=sha256(path),
                config=config,
            )
            path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
            return meta

        class PeriodicSave(BaseCallback):
            def _on_training_start(self):
                self.next_save = self.num_timesteps + checkpoint_every

            def _on_step(self):
                if checkpoint_every > 0 and self.num_timesteps >= self.next_save:
                    save_checkpoint(f"step_{self.num_timesteps}")
                    self.next_save += checkpoint_every
                return True

        policy.learn(
            total_timesteps=total_steps,
            reset_num_timesteps=not bool(resume),
            callback=PeriodicSave(),
        )
        return save_checkpoint("last")
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot", choices=ROBOTS, required=True)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--total-steps", type=int, default=2000000)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument(
        "--reward-stage", choices=["reach", "lift", "place"], default="place"
    )
    p.add_argument("--resume")
    p.add_argument("--transfer")
    p.add_argument("--parameters")
    p.add_argument("--checkpoint-every", type=int, default=250000)
    p.add_argument("--curriculum-bank")
    p.add_argument("--curriculum-phase", type=int, default=236)
    p.add_argument("--normal-fraction", type=float, default=0.1)
    p.add_argument("--bounded-policy", action="store_true")
    p.add_argument("--reset-std", type=float, nargs=4)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--use-sde", action="store_true")
    p.add_argument("--sde-sample-freq", type=int, default=8)
    p.add_argument("--curriculum-window", type=int, default=0)
    p.add_argument("--actor-head-scale", type=float, default=1.0)
    p.add_argument("--cartesian-action-scale", type=float, default=1.0)
    p.add_argument("--exploration-scale", type=float, default=1.0)
    a = vars(p.parse_args())
    a["parameters"] = (
        json.loads(Path(a["parameters"]).read_text()) if a["parameters"] else None
    )
    print(json.dumps(train(**a), indent=2))


if __name__ == "__main__":
    main()
