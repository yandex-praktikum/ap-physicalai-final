import importlib.util
import numpy as np
import pytest


def test_environment_is_implemented():
    assert importlib.util.find_spec("physical_ai.env") is not None


@pytest.mark.parametrize("robot", ["ur5e", "iiwa14"])
@pytest.mark.parametrize(
    "task", ["cup_plate", "cup_shelf", "cup_distractor", "color_match"]
)
def test_reset_step_and_observation_contract(robot, task):
    from physical_ai.env import ManipulationEnv

    with ManipulationEnv(robot, task) as env:
        first, _ = env.reset(seed=21)
        pose = env.data.qpos.copy()
        second, _ = env.reset(seed=21)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(pose, env.data.qpos)
        np.testing.assert_allclose(
            env.data.site("grasp").xmat.reshape(3, 3), np.diag([1, -1, -1]), atol=0.01
        )
        assert env.action_space.shape == (4,)
        assert env.observation_space.contains(first)
        assert not env.success()
        for _ in range(12):
            obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 0, 1]))
            assert np.isfinite(obs).all() and np.isfinite(reward)
        for name in env.object_names:
            assert 0.41 < env.data.body(name).xpos[2] < 0.48
        assert np.linalg.norm(env.data.site("grasp").xpos - env.ee_target) < 0.025
        with pytest.raises(ValueError):
            env.step(np.array([np.nan, 0, 0, 0]))


def test_success_requires_release_stability_and_all_objects():
    from physical_ai.env import ManipulationEnv
    import mujoco

    with ManipulationEnv("ur5e", "cup_distractor") as env:
        env.reset(seed=5)
        for i, name in enumerate(env.object_names):
            adr = env.model.joint(name + "_free").qposadr[0]
            env.data.qpos[adr : adr + 3] = env.goals[i]
        mujoco.mj_forward(env.model, env.data)
        env.data.qvel[:] = 0
        assert env.placement_complete()
        assert not env.success()  # one instant is insufficient
        adr = env.model.joint("cup1_free").qposadr[0]
        env.data.qpos[adr] += 0.12
        mujoco.mj_forward(env.model, env.data)
        assert not env.placement_complete()


def test_bc_observation_does_not_contain_object_state():
    from physical_ai.env import ManipulationEnv

    with ManipulationEnv("iiwa14", "cup_distractor") as env:
        env.reset(seed=0)
        obs = env.bc_observation()
        assert set(obs) == {"rgb", "proprio"}
        assert obs["rgb"].shape == (84, 84, 3)
        assert obs["rgb"].dtype == np.uint8
        assert obs["proprio"].shape == (18,)


def test_orientation_override_replaces_alternative_mjcf_representations():
    from physical_ai.env import ManipulationEnv

    with ManipulationEnv(
        parameters={
            "camera": {"overview": {"quat": [1, 0, 0, 0]}},
            "geom": {"cup0_wall1": {"quat": [1, 0, 0, 0]}},
        }
    ) as env:
        np.testing.assert_allclose(env.model.camera("overview").quat, [1, 0, 0, 0])


def test_fast_spinning_object_cannot_complete_placement():
    from physical_ai.env import ManipulationEnv
    import mujoco

    with ManipulationEnv() as env:
        joint = env.model.joint("cup0_free")
        qa, va = int(joint.qposadr[0]), int(joint.dofadr[0])
        env.data.qpos[qa : qa + 3] = env.goals[0]
        env.data.qvel[va + 3 : va + 6] = [0, 0, 2.0]
        mujoco.mj_forward(env.model, env.data)
        assert not env.placement_complete()


def test_distractor_movement_fails_episode():
    from physical_ai.env import ManipulationEnv
    import mujoco

    with ManipulationEnv("ur5e", "cup_distractor") as env:
        adr = int(env.model.joint("cup1_free").qposadr[0])
        env.data.qpos[adr] += 0.04
        mujoco.mj_forward(env.model, env.data)
        _, _, done, _, info = env.step([0, 0, 0, 1])
        assert done and not info["is_success"]
        assert info["failure"] == "distractor_moved"


def test_color_match_varies_goal_and_preserves_semantic_pairing():
    from physical_ai.env import ManipulationEnv

    with ManipulationEnv("ur5e", "color_match") as env:
        seen = set()
        for seed in range(12):
            env.reset(seed=seed)
            assert env.object_names == ["cup0"]
            color = env.model.geom("cup0_bottom").rgba
            matching = [
                i
                for i in range(2)
                if np.allclose(color, env.model.geom(f"plate{i}").rgba)
            ]
            assert len(matching) == 1
            i = matching[0]
            seen.add(i)
            np.testing.assert_allclose(
                env.goals[0], env.data.body(f"target{i}").xpos + [0, 0, 0.034]
            )
        assert seen == {0, 1}
