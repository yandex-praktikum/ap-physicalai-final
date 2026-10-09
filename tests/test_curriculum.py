import json
import hashlib
from pathlib import Path
import numpy as np
import pytest
import torch
from physical_ai.env import ManipulationEnv
from physical_ai.curriculum import CurriculumEnv
from physical_ai.reset_states import read_bank, write_bank


def bank_file(tmp_path):
    with ManipulationEnv() as env:
        row = {
            key: getattr(env.data, key)[None].copy() for key in ["qpos", "qvel", "ctrl"]
        }
        row.update(
            ee_target=env.ee_target[None].copy(),
            goals=env.goals[None].copy(),
            rgba=env.model.geom_rgba[None].copy(),
            phase=np.array([236]),
            metadata=json.dumps(
                dict(
                    robot="ur5e",
                    task="cup_plate",
                    scenes_sha256=hashlib.sha256(
                        (
                            Path(__file__).resolve().parents[1]
                            / "physical_ai/scenes.py"
                        ).read_bytes()
                    ).hexdigest(),
                )
            ),
        )
    path = tmp_path / "bank"
    metadata = json.loads(row.pop("metadata"))
    write_bank(path, row, metadata)
    return path


def test_curriculum_reset_preserves_observation_contract(tmp_path):
    bank = bank_file(tmp_path)
    with CurriculumEnv(state_bank=bank, phase=236, normal_fraction=0) as env:
        a, info = env.reset(seed=44)
        b, _ = env.reset(seed=44)
        np.testing.assert_array_equal(a, b)
        assert env.observation_space.contains(a)
        assert info["curriculum_phase"] == 236
        assert not env.success()
        with ManipulationEnv() as base:
            expected, _ = base.reset(seed=55)
            env.normal_fraction = 1
            actual, _ = env.reset(seed=55)
            np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="robot/task"):
        CurriculumEnv(robot="iiwa14", state_bank=bank)


def test_bounded_policy_checkpoint_preserves_actions_and_means(tmp_path):
    from physical_ai.rl import train, load_rl

    train(
        "ur5e",
        "cup_plate",
        tmp_path,
        total_steps=64,
        n_steps=32,
        n_envs=1,
        bounded_policy=True,
        ent_coef=0.0001,
        reset_std=[0.3, 0.3, 0.7, 0.8],
    )
    model, meta = load_rl(tmp_path / "last.zip", "ur5e", "cup_plate")
    assert meta["config"]["policy_class"] == "BoundedPolicy"
    obs = torch.randn(8, *model.observation_space.shape) * 100
    with torch.no_grad():
        dist = model.policy.get_distribution(obs).distribution
    assert torch.isfinite(dist.mean).all() and torch.max(torch.abs(dist.mean)) <= 1


def test_distractor_transfer_preserves_initial_action_function():
    from stable_baselines3 import PPO
    from physical_ai.curriculum import BoundedPolicy, ManipulationFeatures
    from physical_ai.rl import transfer_parameters

    kwargs = dict(
        net_arch=dict(pi=[16, 16], vf=[16, 16]),
        features_extractor_class=ManipulationFeatures,
        features_extractor_kwargs={"proprio_dim": 16},
    )
    with ManipulationEnv() as base, ManipulationEnv(
        task="cup_distractor"
    ) as distractor:
        source = PPO(BoundedPolicy, base, policy_kwargs=kwargs, n_steps=8, batch_size=8)
        target = PPO(
            BoundedPolicy, distractor, policy_kwargs=kwargs, n_steps=8, batch_size=8
        )
        transfer_parameters(source, target)
        for seed in [8, 11, 32]:
            a, _ = base.reset(seed=seed)
            b, _ = distractor.reset(seed=seed)
            expected = source.predict(a, deterministic=True)[0]
            actual = target.predict(b, deterministic=True)[0]
            np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_bank_cannot_restore_stale_goals_into_modified_scene(tmp_path):
    bank = bank_file(tmp_path)
    with pytest.raises(ValueError, match="base scenes"):
        CurriculumEnv(
            state_bank=bank,
            parameters={"body": {"target0": {"pos": [0.58, 0.16, 0.414]}}},
        )


def test_transfer_to_sde_preserves_deterministic_actor():
    from stable_baselines3 import PPO
    from physical_ai.curriculum import BoundedPolicy, ManipulationFeatures
    from physical_ai.rl import transfer_parameters

    kwargs = dict(
        net_arch=dict(pi=[16, 16], vf=[16, 16]),
        features_extractor_class=ManipulationFeatures,
        features_extractor_kwargs={"proprio_dim": 16},
    )
    with ManipulationEnv() as env:
        source = PPO(BoundedPolicy, env, policy_kwargs=kwargs, n_steps=8, batch_size=8)
        target = PPO(
            BoundedPolicy,
            env,
            policy_kwargs=kwargs,
            n_steps=8,
            batch_size=8,
            use_sde=True,
        )
        transfer_parameters(source, target)
        obs, _ = env.reset(seed=91)
        np.testing.assert_allclose(
            source.predict(obs, deterministic=True)[0],
            target.predict(obs, deterministic=True)[0],
            atol=1e-6,
        )


def test_curriculum_window_samples_only_requested_stage_interval(tmp_path):
    path = bank_file(tmp_path)
    original, metadata = read_bank(path)
    rows = {key: np.repeat(value, 3, axis=0) for key, value in original.items()}
    rows["phase"] = np.array([234, 235, 236])
    path = tmp_path / "window"
    write_bank(path, rows, metadata)
    with CurriculumEnv(
        state_bank=path, phase=234, phase_window=1, normal_fraction=0
    ) as env:
        observed = {env.reset(seed=seed)[1]["curriculum_phase"] for seed in range(20)}
    assert observed == {234, 235}


def test_cross_robot_transfer_matches_reference_joint_adapter():
    from stable_baselines3 import PPO
    from physical_ai.curriculum import BoundedPolicy, ManipulationFeatures
    from physical_ai.rl import transfer_parameters

    def make(env, p):
        return PPO(
            BoundedPolicy,
            env,
            n_steps=8,
            batch_size=8,
            policy_kwargs=dict(
                net_arch=dict(pi=[16, 16], vf=[16, 16]),
                features_extractor_class=ManipulationFeatures,
                features_extractor_kwargs={"proprio_dim": p},
            ),
        )

    with ManipulationEnv("ur5e") as ur, ManipulationEnv("iiwa14") as kuka:
        source, target = make(ur, 16), make(kuka, 18)
        reference, _ = ur.reset(seed=0)
        transfer_parameters(source, target, source_reference=reference)
        for seed in [0, 13, 42]:
            obs, _ = kuka.reset(seed=seed)
            obs[:7] += np.arange(7) * 0.1
            adapter = np.r_[
                reference[:6], obs[7:9], np.zeros(6), obs[16:18], obs[18:]
            ].astype(np.float32)
            np.testing.assert_allclose(
                target.predict(obs, deterministic=True)[0],
                source.predict(adapter, deterministic=True)[0],
                atol=1e-6,
            )


def test_scaled_cross_robot_checkpoint_roundtrip(tmp_path):
    from physical_ai.rl import train, load_rl

    train(
        "ur5e",
        "cup_plate",
        tmp_path / "source",
        total_steps=16,
        n_steps=8,
        n_envs=1,
        bounded_policy=True,
    )
    train(
        "iiwa14",
        "cup_plate",
        tmp_path / "target",
        total_steps=16,
        n_steps=8,
        n_envs=1,
        transfer=tmp_path / "source/last.zip",
        cartesian_action_scale=0.25,
        use_sde=True,
    )
    policy, meta = load_rl(tmp_path / "target/last.zip", "iiwa14", "cup_plate")
    assert meta["config"]["transfer_source"]["robot"] == "ur5e"
    assert meta["config"]["cartesian_action_scale"] == 0.25
    with ManipulationEnv("iiwa14") as env:
        obs, _ = env.reset(seed=77)
        action, _ = policy.predict(obs, deterministic=True)
        assert np.max(np.abs(action[:3])) <= 0.25 and abs(action[3]) <= 1


def test_exploration_scale_changes_sde_noise_without_mean_reinitialization(tmp_path):
    import torch
    from physical_ai.rl import train, load_rl

    source = tmp_path / "source"
    train(
        "iiwa14",
        "cup_shelf",
        source,
        total_steps=16,
        n_steps=16,
        n_envs=1,
        bounded_policy=True,
        use_sde=True,
        max_steps=2,
    )
    before, _ = load_rl(source / "last.zip", "iiwa14", "cup_shelf")
    output = tmp_path / "target"
    train(
        "iiwa14",
        "cup_shelf",
        output,
        total_steps=16,
        n_steps=16,
        n_envs=1,
        resume=source / "last.zip",
        learning_rate=0,
        max_steps=2,
        exploration_scale=3.0,
    )
    after, meta = load_rl(output / "last.zip", "iiwa14", "cup_shelf")
    assert torch.allclose(after.policy.log_std, before.policy.log_std + np.log(3))
    for name, value in before.policy.state_dict().items():
        if name != "log_std":
            assert torch.equal(value, after.policy.state_dict()[name])
    assert meta["config"]["exploration_scale"] == 3.0
    assert meta["parameter_delta_l2"] == 0


def test_bank_model_fingerprint_survives_unrelated_source_edit_but_rejects_model_change(
    tmp_path,
):
    from physical_ai.curriculum import scene_fingerprint

    original = bank_file(tmp_path)
    arrays, metadata = read_bank(original)
    metadata["scenes_sha256"] = "old-source-version"
    metadata["model_sha256"] = scene_fingerprint("ur5e", "cup_plate")
    compatible = tmp_path / "compatible"
    write_bank(compatible, arrays, metadata)
    with CurriculumEnv(state_bank=compatible, phase=236) as env:
        assert env.reset(seed=0)[0].shape == (32,)
    metadata["model_sha256"] = "wrong-model"
    incompatible = tmp_path / "incompatible"
    write_bank(incompatible, arrays, metadata)
    with pytest.raises(ValueError, match="model hash"):
        CurriculumEnv(state_bank=incompatible, phase=236)
