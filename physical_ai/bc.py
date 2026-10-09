"""Trainable visual baseline; optional proprioception is an explicit ablation."""

import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from physical_ai.data import EpisodeDataset, check_disjoint


class EpisodeBatchSampler(Sampler):
    """Shuffle episodes and frames; bound decompression to one episode at a time."""

    def __init__(self, dataset, batch_size, seed):
        self.dataset, self.batch_size = dataset, batch_size
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for episode in self.rng.permutation(len(self.dataset.metadata)):
            start, end = self.dataset.offsets[episode : episode + 2]
            indices = self.rng.permutation(np.arange(start, end))
            for i in range(0, len(indices), self.batch_size):
                yield indices[i : i + self.batch_size].tolist()

    def __len__(self):
        return sum(
            (b - a + self.batch_size - 1) // self.batch_size
            for a, b in zip(self.dataset.offsets, self.dataset.offsets[1:])
        )


class BCPolicy(nn.Module):
    def __init__(self, proprio_dim, use_proprio=False):
        super().__init__()
        self.proprio_dim, self.use_proprio = proprio_dim, use_proprio
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(1024 + (proprio_dim if use_proprio else 0), 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Tanh(),
        )
        self.register_buffer("proprio_mean", torch.zeros(proprio_dim))
        self.register_buffer("proprio_std", torch.ones(proprio_dim))

    def forward(self, rgb, proprio):
        features = self.encoder(rgb)
        if self.use_proprio:
            features = torch.cat(
                [features, (proprio - self.proprio_mean) / self.proprio_std], dim=-1
            )
        return self.head(features)


def save_bc(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            schema=1,
            state_dict=model.state_dict(),
            proprio_dim=model.proprio_dim,
            use_proprio=model.use_proprio,
            metadata=metadata,
        ),
        path,
    )
    was_training = model.training
    model.eval()
    torch.jit.script(model).save(str(path.with_suffix(".ts")))
    from physical_ai.rl import sha256

    path.with_suffix(".json").write_text(
        json.dumps(
            dict(metadata, schema=1, sha256=sha256(path.with_suffix(".ts"))), indent=2
        )
    )
    model.train(was_training)


def load_bc(path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("schema") != 1:
        raise ValueError("Unknown BC checkpoint schema")
    model = BCPolicy(state["proprio_dim"], state["use_proprio"]).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, state["metadata"]


def train(
    train_dir,
    val_dir,
    out,
    epochs=30,
    batch_size=64,
    lr=3e-4,
    use_proprio=False,
    seed=0,
    device="cpu",
    image_augmentation=False,
):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    training, validation = EpisodeDataset(train_dir), EpisodeDataset(val_dir)
    check_disjoint(training, validation)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.pt").exists():
        raise FileExistsError("Use a new BC run directory")
    model = BCPolicy(training.proprio_dim, use_proprio).to(device)
    # Compute statistics on training episodes ONLY.
    count = 0
    total = np.zeros(training.proprio_dim)
    squares = total.copy()
    for p in training.iter_proprio():
        count += len(p)
        total += p.sum(0)
        squares += (p * p).sum(0)
    model.proprio_mean.copy_(
        torch.tensor(total / count, dtype=torch.float32, device=device)
    )
    model.proprio_std.copy_(
        torch.tensor(
            np.sqrt(np.maximum(squares / count - (total / count) ** 2, 1e-4)),
            dtype=torch.float32,
            device=device,
        )
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loaders = [
        DataLoader(
            training,
            batch_sampler=EpisodeBatchSampler(training, batch_size, seed),
            num_workers=0,
        ),
        DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0),
    ]
    first = training.metadata[0]
    meta = dict(
        data_format="lerobot_v3",
        robot=first["robot"],
        task=first["task"],
        seed=seed,
        use_proprio=use_proprio,
        image_augmentation=image_augmentation,
        train_episode_keys=[
            dict(
                robot=m["robot"],
                task=m["task"],
                seed=m["seed"],
                parameters=m["parameters"],
            )
            for m in training.metadata
        ],
        val_episode_keys=[
            dict(
                robot=m["robot"],
                task=m["task"],
                seed=m["seed"],
                parameters=m["parameters"],
            )
            for m in validation.metadata
        ],
    )
    (out / "config.json").write_text(
        json.dumps(
            dict(meta, epochs=epochs, batch_size=batch_size, lr=lr, device=device),
            indent=2,
        )
    )
    writer = SummaryWriter(str(out))
    best = float("inf")
    history = []
    try:
        for epoch in range(epochs):
            losses = []
            for split, loader in enumerate(loaders):
                model.train(split == 0)
                loss_sum = 0.0
                n = 0
                for rgb, proprio, actions in loader:
                    rgb, proprio, actions = [
                        x.to(device) for x in [rgb, proprio, actions]
                    ]
                    if split == 0 and image_augmentation:
                        # Mild illumination noise; no hue permutation that would erase task semantics.
                        rgb = (
                            rgb
                            * torch.empty((len(rgb), 1, 1, 1), device=device).uniform_(
                                0.85, 1.15
                            )
                            + torch.randn_like(rgb) * 0.015
                        ).clamp(0, 1)
                    with torch.set_grad_enabled(split == 0):
                        loss = (model(rgb, proprio) - actions).square().mean()
                        if split == 0:
                            opt.zero_grad()
                            loss.backward()
                            opt.step()
                    loss_sum += float(loss.detach()) * len(rgb)
                    n += len(rgb)
                losses.append(loss_sum / n)
            history.append(
                dict(epoch=epoch + 1, train_mse=losses[0], val_mse=losses[1])
            )
            for key, value in history[-1].items():
                writer.add_scalar(key, value, epoch + 1)
            save_bc(out / "last.pt", model, dict(meta, epoch=epoch + 1))
            if losses[1] < best:
                best = losses[1]
                save_bc(out / "best.pt", model, dict(meta, epoch=epoch + 1))
            (out / "history.json").write_text(json.dumps(history, indent=2))
    finally:
        writer.close()
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["train-dir", "val-dir", "out"]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--use-proprio", action="store_true")
    p.add_argument("--image-augmentation", action="store_true")
    print(json.dumps(train(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
