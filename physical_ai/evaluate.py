"""Fixed-seed rollout evaluation shared by students and graders."""

import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from physical_ai.env import ManipulationEnv, RUNTIME_PROVENANCE


def wilson_interval(successes, n):
    if n < 1 or not 0 <= successes <= n:
        raise ValueError("Invalid success counts")
    p = successes / n
    z = 1.959963984540054
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def checkpoint_policy(path, kind, robot, task):
    if kind == "rl":
        from physical_ai.rl import load_rl

        model, meta = load_rl(path, robot, task)
        return lambda obs: model.predict(obs, deterministic=True)[0], "state", meta
    from physical_ai.bc import load_bc

    if Path(path).suffix == ".ts":
        from physical_ai.rl import sha256

        meta = json.loads(Path(path).with_suffix(".json").read_text())
        if meta.get("schema") != 1 or meta.get("sha256") != sha256(path):
            raise ValueError("Invalid exported policy metadata/hash")
        model = torch.jit.load(str(path), map_location="cpu").eval()
    else:
        model, meta = load_bc(path)
    if (meta["robot"], meta["task"]) != (robot, task):
        raise ValueError("Checkpoint robot/task mismatch")
    torch.set_num_threads(1)

    def predict(obs):
        with torch.no_grad():
            rgb = (
                torch.from_numpy(obs["rgb"].copy())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                / 255
            )
            prop = torch.from_numpy(obs["proprio"]).unsqueeze(0)
            return model(rgb, prop).squeeze(0).numpy()

    if hasattr(model, "reset"):
        predict.reset = model.reset
    return predict, "bc", meta


def evaluate(
    robot,
    task,
    policy,
    observation,
    seeds,
    parameters=None,
    max_steps=1000,
    video_dir=None,
):
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be nonempty and unique")
    rows = []
    with ManipulationEnv(
        robot, task, parameters=parameters, max_steps=max_steps
    ) as env:
        for seed in seeds:
            state, _ = env.reset(seed=int(seed))
            goal_slot = 0
            if task == "color_match":
                goal_slot = int(
                    np.argmin(
                        [
                            np.linalg.norm(
                                env.goals[0]
                                - (env.data.body(f"target{i}").xpos + [0, 0, 0.034])
                            )
                            for i in range(2)
                        ]
                    )
                )
            frames = []
            total_reward = 0
            if hasattr(policy, "reset"):
                with torch.no_grad():
                    policy.reset()
            for step in range(max_steps):
                obs = state if observation == "state" else env.bc_observation()
                if video_dir:
                    frames.append(env.render())
                action = policy(obs)
                state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                if done or truncated:
                    break
            rows.append(
                dict(
                    seed=int(seed),
                    goal_slot=goal_slot,
                    success=bool(info["is_success"]),
                    steps=step + 1,
                    reward=total_reward,
                    failure=info["failure"]
                    or ("" if info["is_success"] else "timeout"),
                )
            )
            if video_dir:
                import imageio.v2 as imageio

                dest = Path(video_dir)
                dest.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(dest / f"{robot}_{task}_{seed}.mp4", frames, fps=20)
    successes = sum(r["success"] for r in rows)
    by_goal_slot = {}
    for slot in sorted({row["goal_slot"] for row in rows}):
        group = [row for row in rows if row["goal_slot"] == slot]
        count = sum(row["success"] for row in group)
        by_goal_slot[str(slot)] = dict(
            n=len(group), successes=count, success_rate=count / len(group)
        )
    return dict(
        robot=robot,
        task=task,
        parameters=parameters or {},
        max_steps=max_steps,
        success_rate=successes / len(rows),
        successes=successes,
        n=len(rows),
        wilson_95=wilson_interval(successes, len(rows)),
        episodes=rows,
        by_goal_slot=by_goal_slot,
        runtime_provenance=dict(RUNTIME_PROVENANCE),
    )


def save_result(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    with path.with_suffix(".csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["episodes"][0]))
        writer.writeheader()
        writer.writerows(result["episodes"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["robot", "task", "checkpoint", "out"]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--kind", choices=["rl", "bc"], required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=50000)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--parameters")
    p.add_argument("--video-dir")
    a = p.parse_args()
    params = json.loads(Path(a.parameters).read_text()) if a.parameters else None
    policy, obs, meta = checkpoint_policy(a.checkpoint, a.kind, a.robot, a.task)
    result = evaluate(
        a.robot,
        a.task,
        policy,
        obs,
        range(a.seed, a.seed + a.episodes),
        params,
        a.max_steps,
        a.video_dir,
    )
    result["checkpoint"] = str(a.checkpoint)
    result["kind"] = a.kind
    result["checkpoint_sha256"] = meta["sha256"]
    save_result(result, a.out)
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
