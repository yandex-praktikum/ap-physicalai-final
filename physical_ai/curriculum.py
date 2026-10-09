"""Optional PPO reset curriculum and bounded-action policy.

State banks contain physically reached simulator states, never action labels.
Evaluation and demonstration collection always use ordinary ManipulationEnv resets.
"""

from pathlib import Path
import json
import hashlib
import numpy as np
import mujoco
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from physical_ai.reset_states import read_bank
from physical_ai.env import ManipulationEnv
from physical_ai.scenes import ROOT, build_scene


def scene_fingerprint(robot, task):
    """Hash this exact scene, independent of the checkout's absolute path."""
    xml = build_scene(robot, task).replace(str(ROOT), "${PROJECT_ROOT}")
    return hashlib.sha256(xml.encode()).hexdigest()


class ManipulationFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space, proprio_dim):
        self.proprio_dim = proprio_dim
        self.objects = (observation_space.shape[0] - proprio_dim - 6) // 10
        super().__init__(
            observation_space, observation_space.shape[0] + 3 + 6 * self.objects
        )

    def forward(self, obs):
        p = self.proprio_dim
        q = p // 2
        scaled = obs.clone()
        scaled[:, : q - 2] /= np.pi
        scaled[:, q - 2 : q] *= 20
        scaled[:, q:p] *= 0.1
        ee = obs[:, p : p + 3]
        parts = [scaled, 10 * (obs[:, p + 3 : p + 6] - ee)]
        for i in range(self.objects):
            start = p + 6 + 10 * i
            position = obs[:, start : start + 3]
            goal = obs[:, start + 7 : start + 10]
            parts += [10 * (position - ee), 10 * (goal - position)]
        return torch.cat(parts, dim=1)


class BoundedPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Keep Gaussian means inside the executable action range. Parameter
        # objects are unchanged, so the optimizer built above remains valid.
        self.action_net = nn.Sequential(self.action_net, nn.Tanh())


class CartesianMeanScale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", torch.tensor([scale, scale, scale, 1.0]))

    def forward(self, means):
        return means * self.scale


class ScaledPolicy(BoundedPolicy):
    """Bound Cartesian action means to slower motions; keep jaw range unchanged."""

    def __init__(self, *args, cartesian_scale=0.25, **kwargs):
        if not 0 < cartesian_scale <= 1:
            raise ValueError("cartesian_scale must be in (0, 1]")
        self.cartesian_scale = cartesian_scale
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net.append(CartesianMeanScale(self.cartesian_scale))

    def _get_constructor_parameters(self):
        return dict(
            super()._get_constructor_parameters(), cartesian_scale=self.cartesian_scale
        )


class CurriculumEnv(ManipulationEnv):
    def __init__(
        self,
        *args,
        state_bank,
        phase=236,
        normal_fraction=0.1,
        phase_window=0,
        **kwargs
    ):
        self.bank = None
        self.phase = phase
        self.normal_fraction = normal_fraction
        if phase_window < 0:
            raise ValueError("phase_window must be nonnegative")
        self.phase_window = phase_window
        super().__init__(*args, **kwargs)
        if self.parameters:
            raise ValueError("Curriculum banks support unmodified base scenes only")
        self.bank, metadata = read_bank(state_bank)
        if (metadata["robot"], metadata["task"]) != (self.robot, self.task):
            raise ValueError("Curriculum bank robot/task mismatch")
        scene_hash = hashlib.sha256(
            Path(__file__).with_name("scenes.py").read_bytes()
        ).hexdigest()
        if metadata.get("model_sha256"):
            if metadata["model_sha256"] != scene_fingerprint(self.robot, self.task):
                raise ValueError("Curriculum bank model hash mismatch; regenerate bank")
        elif metadata.get("scenes_sha256") != scene_hash:
            raise ValueError(
                "Curriculum bank scene source hash mismatch; regenerate bank"
            )
        if self.bank["qpos"].shape[1] != self.model.nq:
            raise ValueError("Curriculum bank model dimensions mismatch")
        if phase not in self.bank["phase"]:
            raise ValueError("Curriculum phase missing from state bank")
        if not 0 <= normal_fraction <= 1:
            raise ValueError("normal_fraction must be in [0, 1]")

    def reset(self, *, seed=None, options=None):
        state, info = super().reset(seed=seed, options=options)
        if (
            self.bank is None
            or self.phase == 0
            or self.np_random.random() < self.normal_fraction
        ):
            return state, info
        candidates = np.flatnonzero(
            (self.bank["phase"] >= self.phase)
            & (self.bank["phase"] <= self.phase + self.phase_window)
        )
        index = int(self.np_random.choice(candidates))
        for name in ["qpos", "qvel", "ctrl"]:
            getattr(self.data, name)[:] = self.bank[name][index]
        self.ee_target = self.bank["ee_target"][index].copy()
        self.goals = self.bank["goals"][index].copy()
        self.model.geom_rgba[:] = self.bank["rgba"][index]
        mujoco.mj_forward(self.model, self.data)
        self._previous_score = self._potential()
        return self.get_privileged_state(), dict(
            info, curriculum_phase=int(self.bank["phase"][index])
        )
