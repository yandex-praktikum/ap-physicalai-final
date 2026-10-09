import importlib.util
import json
import numpy as np
import pytest


def test_pipeline_modules_exist():
    for name in ["data", "bc", "rl", "evaluate"]:
        assert importlib.util.find_spec("physical_ai." + name) is not None


def test_episode_roundtrip_rejects_misalignment_and_overlap(tmp_path):
    from physical_ai.data import EpisodeDataset, check_disjoint
    from physical_ai.lerobot_data import LeRobotWriter

    meta = dict(
        robot="ur5e",
        task="cup_plate",
        seed=42,
        parameters={},
        expert_sha256="abc",
        success=True,
        source="ppo",
    )
    rgb = np.zeros((3, 84, 84, 3), np.uint8)
    prop = np.zeros((3, 16), np.float32)
    actions = np.zeros((3, 4), np.float32)
    writer = LeRobotWriter(tmp_path / "episodes", "ur5e", "cup_plate")
    with pytest.raises(ValueError, match="length"):
        writer.add_episode(rgb, prop, actions[:2], meta)
    writer.add_episode(rgb, prop, actions, meta)
    writer.finalize()
    ds = EpisodeDataset(tmp_path / "episodes")
    assert len(ds) == 3 and ds[0][0].shape == (3, 84, 84)
    with pytest.raises(ValueError, match="overlap"):
        check_disjoint(ds, ds)


def test_bc_training_and_checkpoint_roundtrip(tmp_path):
    import torch
    from physical_ai.bc import BCPolicy, save_bc, load_bc

    torch.manual_seed(0)
    model = BCPolicy(16, True)
    rgb = torch.rand(4, 3, 84, 84)
    prop = torch.randn(4, 16)
    target = torch.ones(4, 4) * 0.3
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    before = model(rgb, prop).detach().clone()
    loss = (model(rgb, prop) - target).square().mean()
    loss.backward()
    opt.step()
    assert not torch.equal(before, model(rgb, prop))
    path = tmp_path / "bc.pt"
    save_bc(path, model, dict(robot="ur5e", task="cup_plate"))
    loaded, meta = load_bc(path)
    torch.testing.assert_close(model(rgb, prop), loaded(rgb, prop))


def test_ppo_updates_and_reloads(tmp_path):
    from physical_ai.rl import train, load_rl
    from physical_ai.env import ManipulationEnv

    out = tmp_path / "ppo"
    train("ur5e", "cup_plate", out, total_steps=64, n_steps=32, n_envs=1, seed=4)
    policy, meta = load_rl(out / "last.zip", "ur5e", "cup_plate")
    assert meta["algorithm"] == "PPO" and meta["timesteps"] >= 64
    with ManipulationEnv() as env:
        obs, _ = env.reset(seed=5)
        action, _ = policy.predict(obs, deterministic=True)
        assert action.shape == (4,) and np.isfinite(action).all()
    with pytest.raises(ValueError, match="robot/task"):
        load_rl(out / "last.zip", "iiwa14", "cup_plate")


def test_wilson_interval_and_rollout_timeout():
    from physical_ai.evaluate import wilson_interval, evaluate

    assert wilson_interval(0, 10)[0] == 0
    low, high = wilson_interval(5, 10)
    assert 0.23 < low < 0.24 and 0.76 < high < 0.77
    result = evaluate(
        "ur5e",
        "cup_plate",
        lambda obs: np.array([0, 0, 0, 1]),
        "state",
        seeds=[1, 2],
        max_steps=2,
    )
    assert result["success_rate"] == 0
    assert all(x["failure"] == "timeout" for x in result["episodes"])
    assert result["runtime_provenance"]["success_version"] == 2
    assert len(result["runtime_provenance"]["environment_sha256"]) == 64


def test_resume_applies_requested_rollout_size_and_seed(tmp_path):
    from physical_ai.rl import train, load_rl

    train(
        "ur5e",
        "cup_plate",
        tmp_path / "first",
        total_steps=32,
        n_steps=32,
        n_envs=1,
        seed=0,
    )
    import json
    from physical_ai.rl import sha256
    from physical_ai.env import DISCOUNT

    checkpoint = tmp_path / "first/last.zip"
    old, meta = load_rl(checkpoint, "ur5e", "cup_plate")
    old.gamma = 0.9
    old.save(checkpoint)
    meta["sha256"] = sha256(checkpoint)
    checkpoint.with_suffix(".json").write_text(json.dumps(meta))
    train(
        "ur5e",
        "cup_plate",
        tmp_path / "second",
        total_steps=32,
        n_steps=16,
        n_envs=1,
        seed=17,
        resume=tmp_path / "first/last.zip",
    )
    policy, meta = load_rl(tmp_path / "second/last.zip", "ur5e", "cup_plate")
    assert policy.n_steps == 16 and policy.seed == 17
    assert policy.gamma == policy.rollout_buffer.gamma == DISCOUNT
    assert meta["config"]["gamma"] == DISCOUNT


def test_episode_batch_sampler_never_loses_or_duplicates_frames():
    from physical_ai.bc import EpisodeBatchSampler

    class Data:
        metadata = [{}, {}]
        offsets = [0, 5, 12]

    batches = list(EpisodeBatchSampler(Data(), 3, 12))
    assert sorted(i for batch in batches for i in batch) == list(range(12))
    assert max(map(len, batches)) <= 3


def test_periodic_ppo_checkpoint_is_loadable(tmp_path):
    from physical_ai.rl import train, load_rl

    train(
        "ur5e",
        "cup_plate",
        tmp_path,
        total_steps=64,
        n_steps=32,
        n_envs=1,
        checkpoint_every=32,
    )
    model, meta = load_rl(tmp_path / "step_64.zip", "ur5e", "cup_plate")
    assert model.num_timesteps == meta["timesteps"] == 64
    assert meta["parameter_delta_l2"] > 0


def test_color_evaluation_reports_both_target_slots():
    import numpy as np
    from physical_ai.evaluate import evaluate

    result = evaluate(
        "ur5e",
        "color_match",
        lambda obs: np.array([0, 0, 0, 1]),
        "state",
        range(12),
        max_steps=1,
    )
    assert set(result["by_goal_slot"]) == {"0", "1"}
    assert sum(group["n"] for group in result["by_goal_slot"].values()) == 12
    assert all(group["successes"] == 0 for group in result["by_goal_slot"].values())


def test_curriculum_recipe_runs_dependencies_and_verifies_reuse(tmp_path):
    import json
    import pytest
    from physical_ai.train_curriculum import run_recipe, dependency_order
    from physical_ai.rl import load_rl, sha256

    recipe = dict(
        schema=1,
        stages=[
            dict(
                id="source",
                robot="ur5e",
                task="cup_plate",
                train=dict(total_steps=16, n_steps=8, n_envs=1, bounded_policy=True),
            ),
            dict(
                id="target",
                robot="ur5e",
                task="cup_distractor",
                transfer_from="source",
                train=dict(
                    total_steps=16,
                    n_steps=8,
                    n_envs=1,
                    bounded_policy=True,
                    use_sde=True,
                ),
            ),
            dict(
                id="final",
                robot="ur5e",
                task="cup_distractor",
                resume_from="target",
                train=dict(
                    total_steps=16, n_steps=8, n_envs=1, use_sde=True, sde_sample_freq=5
                ),
            ),
        ],
        outputs={"ur5e/cup_distractor": "final"},
    )
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    out = tmp_path / "runs"
    final = run_recipe(path, "ur5e/cup_distractor", out)
    model, meta = load_rl(final, "ur5e", "cup_distractor")
    assert (
        model.use_sde and model.sde_sample_freq == 5 and meta["parameter_delta_l2"] > 0
    )
    digest = sha256(final)
    assert run_recipe(path, "ur5e/cup_distractor", out) == final
    assert sha256(final) == digest
    source, source_meta = load_rl(out / "source/last.zip", "ur5e", "cup_plate")
    import torch

    with torch.no_grad():
        next(source.policy.parameters()).add_(0.001)
    source.save(out / "source/last.zip")
    source_meta["sha256"] = sha256(out / "source/last.zip")
    (out / "source/last.json").write_text(json.dumps(source_meta))
    with pytest.raises(ValueError, match="changed"):
        run_recipe(path, "ur5e/cup_distractor", out)
    recipe["stages"][0]["train"]["total_steps"] = 32
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="changed"):
        run_recipe(path, "ur5e/cup_distractor", out)
    recipe["stages"][0]["resume_from"] = "target"
    with pytest.raises(ValueError, match="cycle"):
        dependency_order(recipe, "ur5e/cup_distractor")


def test_curriculum_recipe_rejects_wrong_output_pair():
    import pytest
    from physical_ai.train_curriculum import dependency_order

    recipe = dict(
        stages=[dict(id="a", robot="ur5e", task="cup_plate")],
        outputs={"iiwa14/cup_plate": "a"},
    )
    with pytest.raises(ValueError, match="target"):
        dependency_order(recipe, "iiwa14/cup_plate")


def test_success_only_collector_does_not_render_failed_attempts(tmp_path, monkeypatch):
    from physical_ai import data

    class Policy:
        def predict(self, state, deterministic=True):
            return np.array([0, 0, 0, 1]), None

    monkeypatch.setattr(data, "load_rl", lambda *args: (Policy(), {}))

    def forbidden_render(self):
        raise AssertionError("Failed attempts should not render RGB")

    monkeypatch.setattr(data.ManipulationEnv, "bc_observation", forbidden_render)
    checkpoint = tmp_path / "dummy.zip"
    checkpoint.write_bytes(b"test")
    with pytest.raises(RuntimeError, match="Collected 0/1"):
        data.collect(
            "ur5e",
            "cup_plate",
            checkpoint,
            tmp_path / "data",
            episodes=1,
            max_attempts=1,
            max_steps=2,
        )
    manifest = json.loads((tmp_path / "data/manifest.json").read_text())
    assert manifest["runtime_provenance"]["success_version"] == 2
