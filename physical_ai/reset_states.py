"""Exact simulator reset states stored as numeric LeRobot v3 datasets."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def bank_digest(root):
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"Empty reset-state dataset: {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def write_bank(root, arrays, metadata):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    count = len(arrays["phase"])
    if not count or any(len(v) != count for v in arrays.values()):
        raise ValueError("Reset-state row count mismatch")
    spec = {}
    features = {}
    for key, array in arrays.items():
        if (
            array.dtype
            not in (np.dtype("float32"), np.dtype("float64"), np.dtype("int64"))
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"Invalid reset-state array: {key}")
        spec[key] = dict(shape=list(array.shape[1:]), dtype=str(array.dtype))
        features[f"sim.{key}"] = dict(
            dtype=str(array.dtype), shape=(int(np.prod(array.shape[1:])),), names=None
        )
    dataset = LeRobotDataset.create(
        repo_id=f"local/reset_{metadata['robot']}_{metadata['task']}",
        root=root,
        fps=20,
        robot_type=metadata["robot"],
        features=features,
        use_videos=False,
    )
    try:
        for i in range(count):
            frame = {
                f"sim.{key}": np.asarray(value[i]).reshape(-1)
                for key, value in arrays.items()
            }
            dataset.add_frame(dict(frame, task=f"Reset states for {metadata['task']}"))
        dataset.save_episode()
    finally:
        dataset.finalize()
    (root / "meta/reset_bank.json").write_text(
        json.dumps(dict(schema=1, metadata=metadata, arrays=spec), indent=2)
    )


def read_bank(root):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    spec = json.loads((root / "meta/reset_bank.json").read_text())
    if info["codebase_version"] != "v3.0" or spec["schema"] != 1:
        raise ValueError("Expected a LeRobot v3 reset-state dataset")
    # Read stored Arrow dtypes directly: reset coordinates must retain float64.
    table = pa.concat_tables(
        [pq.read_table(p) for p in sorted((root / "data").rglob("*.parquet"))]
    )
    indices = table["index"].combine_chunks().to_numpy()
    order = np.argsort(indices)
    if not np.array_equal(indices[order], np.arange(info["total_frames"])):
        raise ValueError("Reset-state indices are missing or duplicated")
    arrays = {}
    for key, layout in spec["arrays"].items():
        column = table[f"sim.{key}"].combine_chunks()
        flat = (column.flatten() if hasattr(column, "flatten") else column).to_numpy(
            zero_copy_only=False
        )
        arrays[key] = (
            flat.astype(layout["dtype"], copy=False)
            .reshape((len(order), *layout["shape"]))[order]
            .copy()
        )
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"Nonfinite reset states: {key}")
    return arrays, spec["metadata"]
