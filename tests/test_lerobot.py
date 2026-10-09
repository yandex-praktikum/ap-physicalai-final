import json
import numpy as np
import pytest


def test_lerobot_roundtrip_alignment_provenance_and_bc(tmp_path):
    from physical_ai.data import EpisodeDataset, check_disjoint
    from physical_ai.lerobot_data import LeRobotWriter
    from physical_ai.bc import train
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    for split, seed in [("train", 1), ("val", 2)]:
        writer = LeRobotWriter(tmp_path / split, "ur5e", "cup_plate")
        for i, length in enumerate([3, 5]):
            rgb = np.full((length, 84, 84, 3), 30 + i * 150, dtype=np.uint8)
            prop = np.arange(length * 16, dtype=np.float32).reshape(length, 16) / 100
            action = np.full((length, 4), i / 2, dtype=np.float32)
            meta = dict(
                robot="ur5e",
                task="cup_plate",
                seed=seed + i * 10,
                parameters={},
                expert_sha256="a" * 64,
                source="ppo",
                success=True,
            )
            writer.add_episode(rgb, prop, action, meta)
        writer.finalize()
    official = LeRobotDataset(
        "local/ur5e_cup_plate", root=tmp_path / "train", video_backend="pyav"
    )
    assert official.meta.info["codebase_version"] == "v3.0"
    assert official.num_episodes == 2 and len(official) == 8
    assert official[3]["episode_index"].item() == 1
    np.testing.assert_allclose(official[3]["action"].numpy(), 0.5)
    ds, val = EpisodeDataset(tmp_path / "train"), EpisodeDataset(tmp_path / "val")
    assert ds.offsets == [0, 3, 8] and ds.metadata[1]["seed"] == 11
    assert ds[3][0].shape == (3, 84, 84)
    assert abs(float(ds[3][0].mean()) - 180 / 255) < 0.03
    np.testing.assert_allclose(ds[3][1].numpy(), np.arange(16) / 100)
    np.testing.assert_allclose(ds[3][2].numpy(), 0.5)
    check_disjoint(ds, val)
    with pytest.raises(ValueError, match="overlap"):
        check_disjoint(ds, ds)
    history = train(
        tmp_path / "train",
        tmp_path / "val",
        tmp_path / "bc",
        epochs=1,
        use_proprio=True,
    )
    assert np.isfinite(history[0]["train_mse"]) and (tmp_path / "bc/best.ts").exists()


def test_lerobot_writer_rejects_invalid_or_duplicate_episode(tmp_path):
    from physical_ai.lerobot_data import LeRobotWriter

    writer = LeRobotWriter(tmp_path / "data", "iiwa14", "cup_plate")
    rgb = np.zeros((2, 84, 84, 3), np.uint8)
    prop = np.zeros((2, 18), np.float32)
    act = np.zeros((2, 4), np.float32)
    meta = dict(
        robot="iiwa14",
        task="cup_plate",
        seed=1,
        parameters={},
        source="ppo",
        success=True,
    )
    with pytest.raises(ValueError, match="length"):
        writer.add_episode(rgb, prop, act[:1], meta)
    writer.add_episode(rgb, prop, act, meta)
    with pytest.raises(ValueError, match="Duplicate"):
        writer.add_episode(rgb, prop, act, meta)
    writer.finalize()
    with pytest.raises(FileExistsError):
        LeRobotWriter(tmp_path / "data", "iiwa14", "cup_plate")


def test_collector_defaults_to_native_lerobot(tmp_path, monkeypatch):
    from physical_ai import data
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    class Policy:
        def predict(self, state, deterministic=True):
            return np.array([0, 0, 0, 1], dtype=np.float32), None

    monkeypatch.setattr(data, "load_rl", lambda *args: (Policy(), {}))
    checkpoint = tmp_path / "expert.zip"
    checkpoint.write_bytes(b"test policy fixture")
    out = tmp_path / "collected"
    data.collect(
        "ur5e",
        "cup_plate",
        checkpoint,
        out,
        episodes=1,
        max_attempts=1,
        only_success=False,
        max_steps=2,
    )
    ds = LeRobotDataset("local/ur5e_cup_plate", root=out, video_backend="pyav")
    assert not list(out.glob("*.npz")) and len(ds) == 2
    assert not ds[0]["next.done"].item() and ds[1]["next.done"].item()
    np.testing.assert_allclose(ds[1]["action"].numpy(), [0, 0, 0, 1])
    assert json.loads((out / "manifest.json").read_text())["format"] == "lerobot_v3"


def test_reset_bank_roundtrip_is_exact_and_detects_changed_content(tmp_path):
    from physical_ai.reset_states import write_bank, read_bank, bank_digest
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    arrays = {
        "qpos": np.array([[1.000000000000001, 2.0], [3.0, 4.0]], dtype=np.float64),
        "rgba": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        "phase": np.array([12, 236], dtype=np.int64),
    }
    metadata = dict(robot="ur5e", task="cup_plate", model_sha256="test")
    root = tmp_path / "bank"
    write_bank(root, arrays, metadata)
    actual, meta = read_bank(root)
    assert meta == metadata
    for key, value in arrays.items():
        assert actual[key].dtype == value.dtype
        np.testing.assert_array_equal(actual[key], value)
    official = LeRobotDataset("local/reset_ur5e_cup_plate", root=root)
    assert official.meta.info["codebase_version"] == "v3.0" and len(official) == 2
    before = bank_digest(root)
    (root / "meta/reset_bank.json").write_text(
        (root / "meta/reset_bank.json").read_text() + "\n"
    )
    assert bank_digest(root) != before


def test_demonstration_loader_accepts_only_lerobot(tmp_path):
    from physical_ai.data import EpisodeDataset

    with pytest.raises(ValueError, match="LeRobot"):
        EpisodeDataset(tmp_path)
