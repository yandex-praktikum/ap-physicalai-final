"""Inspect named MJCF parameters, preview scenes and apply JSON edits."""

import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import mujoco
from physical_ai.env import ManipulationEnv
from physical_ai.scenes import ROBOTS, TASKS, build_scene


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot", choices=ROBOTS, default="ur5e")
    p.add_argument("--task", choices=TASKS, default="cup_plate")
    p.add_argument("--parameters")
    p.add_argument("--list-parameters", choices=["body", "geom", "camera", "light"])
    p.add_argument("--snapshot")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    params = json.loads(Path(a.parameters).read_text()) if a.parameters else None
    if a.list_parameters:
        root = ET.fromstring(build_scene(a.robot, a.task, params))
        for node in root.iter(a.list_parameters):
            if node.get("name"):
                print(json.dumps(node.attrib, ensure_ascii=False))
        return
    with ManipulationEnv(a.robot, a.task, parameters=params) as env:
        env.reset(seed=a.seed)
        if a.snapshot:
            from PIL import Image

            r = mujoco.Renderer(env.model, 480, 640)
            r.update_scene(env.data, camera="overview")
            path = Path(a.snapshot)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(r.render()).save(path)
            r.close()
            print(path)
            return
        from mujoco import viewer as mjviewer

        with mjviewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running():
                start = time.monotonic()
                mujoco.mj_step(env.model, env.data, nstep=20)
                viewer.sync()
                time.sleep(max(0, 0.02 - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
