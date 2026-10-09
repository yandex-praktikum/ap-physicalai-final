"""LeRobot v3 demonstrations (Parquet + MP4) with per-episode provenance."""

from bisect import bisect_right
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

IMAGE = "observation.images.front"
STATE = "observation.state"


class LeRobotWriter:
    def __init__(self, root, robot, task):
        self.root = Path(root)
        if self.root.exists():
            raise FileExistsError(self.root)
        self.robot, self.task = robot, task
        self.proprio_dim = {"ur5e": 16, "iiwa14": 18}[robot]
        self.metadata = []
        self.keys = set()
        self.dataset = LeRobotDataset.create(
            repo_id=f"local/{robot}_{task}",
            root=self.root,
            robot_type=robot,
            fps=20,
            use_videos=True,
            vcodec="h264",
            video_backend="pyav",
            encoder_threads=2,
            features={
                IMAGE: dict(
                    dtype="video",
                    shape=(84, 84, 3),
                    names=["height", "width", "channels"],
                ),
                STATE: dict(dtype="float32", shape=(self.proprio_dim,), names=None),
                "action": dict(
                    dtype="float32", shape=(4,), names=["dx", "dy", "dz", "gripper"]
                ),
                "next.done": dict(dtype="bool", shape=(1,), names=None),
            },
        )

    def add_episode(self, rgb, proprio, actions, metadata):
        from physical_ai.data import episode_key

        n = len(actions)
        if not n or len(rgb) != n or len(proprio) != n:
            raise ValueError("Episode length mismatch or empty episode")
        if rgb.shape != (n, 84, 84, 3) or rgb.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB (T,84,84,3)")
        if proprio.shape != (n, self.proprio_dim) or actions.shape != (n, 4):
            raise ValueError("Invalid action/proprio shapes")
        if (
            not np.isfinite(proprio).all()
            or not np.isfinite(actions).all()
            or np.abs(actions).max() > 1.00001
        ):
            raise ValueError("Invalid finite/range contract")
        if (metadata["robot"], metadata["task"]) != (self.robot, self.task):
            raise ValueError("Episode robot/task mismatch")
        key = episode_key(metadata)
        if key in self.keys:
            raise ValueError("Duplicate episode reset")
        for i in range(n):
            self.dataset.add_frame(
                {
                    IMAGE: rgb[i],
                    STATE: proprio[i].astype(np.float32),
                    "action": actions[i].astype(np.float32),
                    "next.done": np.array([i == n - 1], dtype=bool),
                    "task": self.task,
                }
            )
        self.dataset.save_episode(parallel_encoding=False)
        record = dict(metadata, episode_index=len(self.metadata), length=n)
        self.metadata.append(record)
        self.keys.add(key)
        with (self.root / "meta/provenance.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")

    def finalize(self):
        self.dataset.finalize()
        self.dataset.stop_image_writer()


class LeRobotEpisodeDataset(Dataset):
    """Load the official format; cache one decoded episode for shuffled BC batches."""

    def __init__(self, directory):
        from physical_ai.data import episode_key

        self.root = Path(directory)
        self.metadata = [
            json.loads(line)
            for line in (self.root / "meta/provenance.jsonl").read_text().splitlines()
        ]
        if not self.metadata:
            raise ValueError("No episodes")
        first = self.metadata[0]
        self.dataset = LeRobotDataset(
            f"local/{first['robot']}_{first['task']}",
            root=self.root,
            video_backend="pyav",
        )
        if self.dataset.meta.info["codebase_version"] != "v3.0":
            raise ValueError("Expected LeRobot v3.0")
        self.proprio_dim = self.dataset.features[STATE]["shape"][0]
        self.offsets = [0]
        for i, m in enumerate(self.metadata):
            if m["episode_index"] != i or m["length"] != int(
                self.dataset.meta.episodes[i]["length"]
            ):
                raise ValueError("Episode provenance length/index mismatch")
            if (m["robot"], m["task"]) != (first["robot"], first["task"]):
                raise ValueError("Expected one robot/task per dataset")
            self.offsets.append(self.offsets[-1] + m["length"])
        if len(self.metadata) != self.dataset.num_episodes or self.offsets[-1] != len(
            self.dataset
        ):
            raise ValueError("Episode provenance count mismatch")
        if len({episode_key(m) for m in self.metadata}) != len(self.metadata):
            raise ValueError("Duplicate episode resets")
        self._table = self.dataset.hf_dataset.with_format("numpy")
        self._cache_index = None

    def __len__(self):
        return self.offsets[-1]

    def iter_proprio(self):
        for start, end in zip(self.offsets, self.offsets[1:]):
            yield np.stack(self._table[start:end][STATE]).astype(np.float64)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ep = bisect_right(self.offsets, index) - 1
        if self._cache_index != ep:
            start, end = self.offsets[ep : ep + 2]
            rows = self._table[start:end]
            if not np.all(rows["episode_index"] == ep):
                raise ValueError("Episode frames out of order")
            meta = self.dataset.meta.episodes[ep]
            timestamps = (
                np.asarray(rows["timestamp"], dtype=float)
                + meta[f"videos/{IMAGE}/from_timestamp"]
            ).tolist()
            video = self.root / self.dataset.meta.get_video_file_path(ep, IMAGE)
            frames = decode_video_frames(
                video, timestamps, self.dataset.tolerance_s, "pyav"
            )
            prop = np.stack(rows[STATE]).astype(np.float32)
            actions = np.stack(rows["action"]).astype(np.float32)
            if not np.isfinite(prop).all() or not np.isfinite(actions).all():
                raise ValueError("Invalid finite contract")
            self._cache = frames, torch.from_numpy(prop), torch.from_numpy(actions)
            self._cache_index = ep
        i = index - self.offsets[ep]
        return tuple(x[i].clone() for x in self._cache)
