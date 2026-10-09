"""Episode-aligned demonstrations and lazy episode loading."""

import argparse
import json
from pathlib import Path
import numpy as np
from physical_ai.env import ManipulationEnv, RUNTIME_PROVENANCE
from physical_ai.rl import load_rl, sha256


def episode_key(meta):
    # Same reset, even from a different expert, must not cross a data split.
    return json.dumps(
        {k: meta[k] for k in ["robot", "task", "seed", "parameters"]}, sort_keys=True
    )


def EpisodeDataset(directory):
    if not (Path(directory) / "meta/info.json").is_file():
        raise ValueError(f"Expected a LeRobot v3 dataset directory: {directory}")
    from physical_ai.lerobot_data import LeRobotEpisodeDataset

    return LeRobotEpisodeDataset(directory)


def check_disjoint(train, validation):
    if {episode_key(m) for m in train.metadata} & {
        episode_key(m) for m in validation.metadata
    }:
        raise ValueError("Train/validation episode overlap")
    if (train.metadata[0]["robot"], train.metadata[0]["task"]) != (
        validation.metadata[0]["robot"],
        validation.metadata[0]["task"],
    ):
        raise ValueError("Train/validation robot/task mismatch")


def collect(
    robot,
    task,
    checkpoint,
    out,
    episodes=100,
    seed=1000,
    max_attempts=500,
    only_success=True,
    scene_bank=None,
    max_steps=1000,
):
    if episodes < 1 or max_attempts < episodes:
        raise ValueError("Invalid collection budget")
    policy, metadata = load_rl(checkpoint, robot, task)
    from physical_ai.lerobot_data import LeRobotWriter

    out = Path(out)
    if out.exists():
        if any(out.iterdir()):
            raise FileExistsError("Collection output must be empty")
        out.rmdir()
    writer = LeRobotWriter(out, robot, task)
    try:
        bank = scene_bank or [{}]
        accepted = 0
        attempts = []
        for attempt in range(max_attempts):
            parameters = bank[attempt % len(bank)]
            with ManipulationEnv(
                robot, task, parameters=parameters, max_steps=max_steps
            ) as env:
                state, _ = env.reset(seed=seed + attempt)
                rgb = []
                prop = []
                actions = []
                preflight = None
                if only_success:
                    # A deterministic state-only pass avoids expensive RGB rendering
                    # for rejected episodes. Accepted episodes are replayed exactly.
                    for step in range(max_steps):
                        action, _ = policy.predict(state, deterministic=True)
                        state, _, done, truncated, info = env.step(action)
                        if done or truncated:
                            break
                    preflight = dict(
                        success=info["is_success"], steps=step + 1, state=state.copy()
                    )
                    if preflight["success"]:
                        state, _ = env.reset(seed=seed + attempt)
                if not only_success or preflight["success"]:
                    for _ in range(max_steps):
                        obs = env.bc_observation()
                        action, _ = policy.predict(state, deterministic=True)
                        rgb.append(obs["rgb"])
                        prop.append(obs["proprio"])
                        actions.append(action)
                        state, _, done, truncated, info = env.step(action)
                        if done or truncated:
                            break
                    if preflight is not None and (
                        not info["is_success"]
                        or len(actions) != preflight["steps"]
                        or not np.allclose(state, preflight["state"], atol=1e-6, rtol=0)
                    ):
                        raise RuntimeError(
                            "Deterministic PPO replay differed from preflight; episode not saved"
                        )
                meta = dict(
                    robot=robot,
                    task=task,
                    seed=seed + attempt,
                    parameters=parameters,
                    expert_sha256=sha256(checkpoint),
                    source="ppo",
                    success=info["is_success"],
                    failure=info["failure"],
                    runtime_provenance=dict(RUNTIME_PROVENANCE),
                )
                attempts.append({"seed": seed + attempt, "success": info["is_success"]})
                if info["is_success"] or not only_success:
                    writer.add_episode(
                        np.asarray(rgb),
                        np.asarray(prop),
                        np.asarray(actions),
                        meta,
                    )
                    accepted += 1
            (out / "manifest.json").write_text(
                json.dumps(
                    dict(
                        schema=1,
                        format="lerobot_v3",
                        robot=robot,
                        task=task,
                        accepted=accepted,
                        requested=episodes,
                        attempts=attempts,
                        only_success=only_success,
                        state_only_preflight=only_success,
                        runtime_provenance=dict(RUNTIME_PROVENANCE),
                    ),
                    indent=2,
                )
            )
            if accepted == episodes:
                return
        raise RuntimeError(
            f"Collected {accepted}/{episodes} episodes in {max_attempts} attempts; evaluate/improve the PPO expert first. Partial data and attempt log retained."
        )
    finally:
        writer.finalize()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["robot", "task", "checkpoint", "out"]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--max-attempts", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--include-failures", action="store_true")
    p.add_argument("--scene-bank")
    a = vars(p.parse_args())
    a["only_success"] = not a.pop("include_failures")
    a["scene_bank"] = (
        json.loads(Path(a["scene_bank"]).read_text()) if a["scene_bank"] else None
    )
    collect(**a)


if __name__ == "__main__":
    main()
