"""Build self-contained MJCF scenes from pinned Menagerie models."""

from pathlib import Path
import copy
import math
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROBOTS = {
    "ur5e": ("universal_robots_ur5e", "ur5e.xml", 6),
    "iiwa14": ("kuka_iiwa_14", "iiwa14.xml", 7),
}
TASKS = ("cup_plate", "cup_shelf", "cup_distractor", "color_match")
COLORS = [(0.8, 0.12, 0.12, 1), (0.1, 0.3, 0.85, 1), (0.15, 0.7, 0.25, 1)]
TABLE = 0.4


def vec(values):
    return " ".join(f"{float(v):.8g}" for v in values)


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def scene_layout(task):
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    if task == "cup_plate":
        return [[0.42, -0.14, 0.438]], [[0.48, 0.16, 0.448]]
    if task == "cup_shelf":
        return [[0.42, -0.14, 0.438]], [[0.48, 0.17, 0.588]]
    if task == "cup_distractor":
        return [[0.42, -0.14, 0.438], [0.63, 0, 0.438]], [
            [0.48, 0.16, 0.448],
            [0.63, 0, 0.438],
        ]
    return [[0.46, -0.14, 0.438]], [[0.37, 0.16, 0.448], [0.61, 0.16, 0.448]]


def build_scene(robot, task, parameters=None):
    """Parameters are general MJCF edits: section -> name -> attributes.

    Supported sections: body, geom, light, camera. Structural changes should be
    performed in the MJCF or generator directly. Unknown names/attributes fail.
    """
    folder, filename, _ = ROBOTS[robot]
    root = ET.parse(ROOT / "assets" / folder / filename).getroot()
    root.set("model", f"{robot}_{task}")
    compiler = root.find("compiler")
    compiler.set("meshdir", str(ROOT / "assets" / folder / "assets"))
    root.remove(root.find("keyframe"))
    opt = root.find("option")
    opt.set("timestep", ".001")
    opt.set("integrator", "implicitfast")
    opt.set("cone", "elliptic")
    add(root, "visual")
    add(root.find("visual"), "global", offwidth="640", offheight="480")
    default = add(root.find("default"), "default", **{"class": "tool"})
    add(
        default,
        "geom",
        type="box",
        friction="1.3 .01 .001",
        condim="4",
        rgba=".2 .2 .23 1",
    )
    add(
        default,
        "joint",
        type="slide",
        damping="3",
        armature=".01",
        limited="true",
        range="0 .045",
    )
    wb = root.find("worldbody")
    base = wb.find("body[@name='base']")
    base.set("pos", "0 0 .4")
    add(wb, "geom", name="floor", type="plane", size="2 2 .1", rgba=".25 .28 .3 1")
    table = add(wb, "body", name="table", pos=".35 0 .375")
    add(
        table,
        "geom",
        name="table_top",
        type="box",
        size=".55 .45 .025",
        rgba=".62 .58 .50 1",
        friction="1 .01 .001",
    )
    add(
        wb,
        "light",
        name="key_light",
        pos="0 -1 2",
        dir=".2 .3 -1",
        diffuse=".85 .85 .85",
    )
    add(
        wb,
        "camera",
        name="overview",
        pos="1.25 -1.15 1.35",
        xyaxes=".75 .66 0 -.34 .39 .86",
        fovy="48",
    )
    # Tool frame inherits the Menagerie flange attachment transform.
    attachment = root.find(".//site[@name='attachment_site']")
    parent = next(b for b in root.iter("body") if attachment in list(b))
    tool = add(
        parent,
        "body",
        name="tool",
        pos=attachment.get("pos", "0 0 0"),
        quat=attachment.get("quat", "1 0 0 0"),
        childclass="tool",
    )
    add(tool, "geom", name="palm", size=".038 .04 .015", mass=".15")
    add(tool, "site", name="grasp", pos="0 0 .09", size=".003", rgba="0 1 0 1")
    add(tool, "camera", name="wrist", pos=".06 0 .01", xyaxes="0 -1 0 1 0 0", fovy="75")
    actuators = root.find("actuator")
    for sign, label in [(1, "left"), (-1, "right")]:
        finger = add(tool, "body", name=f"{label}_finger", pos=f"0 {sign*.008} 0")
        add(finger, "joint", name=f"{label}_jaw", axis=f"0 {sign} 0")
        add(
            finger,
            "geom",
            name=f"{label}_pad",
            pos="0 0 .072",
            size=".035 .006 .043",
            mass=".05",
        )
        add(
            actuators,
            "position",
            name=f"{label}_jaw_motor",
            joint=f"{label}_jaw",
            kp="150",
            kv="5",
            ctrlrange="0 .045",
            forcerange="-8 8",
        )
    starts, goals = scene_layout(task)
    if robot == "iiwa14" and task == "cup_shelf":
        goals = [[0.48, 0.17, 0.488]]
    for i, start in enumerate(starts):
        cup = add(wb, "body", name=f"cup{i}", pos=vec(start))
        add(cup, "freejoint", name=f"cup{i}_free")
        add(
            cup,
            "geom",
            name=f"cup{i}_bottom",
            type="cylinder",
            size=".027 .004",
            pos="0 0 -.03",
            mass=".03",
            rgba=vec(COLORS[i]),
            friction="1.2 .01 .001",
        )
        # A visible lid prevents a finger entering the hollow vessel and hooking
        # its inside wall. The beginner task uses closed drinking cups.
        add(cup, "geom", name=f"cup{i}_lid", type="cylinder", size=".027 .003",
            pos="0 0 .032", mass=".006", rgba=vec(COLORS[i]), friction="1.2 .01 .001")
        # Hollow cup: collision geometry follows the visible walls.
        for j in range(12):
            theta = 2 * math.pi * j / 12
            add(
                cup,
                "geom",
                name=f"cup{i}_wall{j}",
                type="box",
                size=".004 .008 .032",
                pos=vec([0.026 * math.cos(theta), 0.026 * math.sin(theta), 0]),
                euler=f"0 0 {theta}",
                mass=".004",
                rgba=vec(COLORS[i]),
                friction="1.2 .01 .001",
            )
        for j, (a, b) in enumerate(
            [
                ((0.03, 0, -0.02), (0.048, 0, -0.02)),
                ((0.048, 0, -0.02), (0.048, 0, 0.02)),
                ((0.048, 0, 0.02), (0.03, 0, 0.02)),
            ]
        ):
            add(
                cup,
                "geom",
                name=f"cup{i}_handle{j}",
                type="capsule",
                fromto=vec(a + b),
                size=".004",
                mass=".002",
                rgba=vec(COLORS[i]),
            )
    for i, goal in enumerate(goals):
        target = add(
            wb, "body", name=f"target{i}", pos=vec([goal[0], goal[1], goal[2] - 0.034])
        )
        if task in ("cup_plate", "color_match") or (
            task == "cup_distractor" and i == 0
        ):
            add(
                target,
                "geom",
                name=f"plate{i}",
                type="cylinder",
                size=".067 .005",
                pos="0 0 -.005",
                rgba=vec(COLORS[i]),
            )
            for j in range(20):
                theta = 2 * math.pi * j / 20
                add(
                    target,
                    "geom",
                    name=f"plate{i}_rim{j}",
                    type="sphere",
                    size=".008",
                    pos=vec([0.064 * math.cos(theta), 0.064 * math.sin(theta), 0.001]),
                    rgba=vec(COLORS[i]),
                )
        elif task == "cup_distractor":
            add(
                target,
                "geom",
                name=f"marker{i}",
                type="cylinder",
                size=".065 .0003",
                pos="0 0 -.004",
                rgba=vec(COLORS[i]),
                contype="0",
                conaffinity="0",
            )
        else:
            add(
                target,
                "geom",
                name="shelf_surface",
                type="box",
                size=".13 .09 .015",
                pos="0 0 -.015",
                rgba=".6 .4 .2 1",
            )
            for side in [-1, 1]:
                add(
                    target,
                    "geom",
                    name=f"shelf_leg_{side}",
                    type="box",
                    size=".012 .075 .012" if robot == "iiwa14" else ".012 .075 .075",
                    pos=f"{side*.115} 0 -.042" if robot == "iiwa14" else f"{side*.115} 0 -.09",
                    rgba=".5 .3 .16 1",
                )
    apply_parameters(root, parameters or {})
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def apply_parameters(root, parameters):
    allowed = {
        "body": {"pos", "quat"},
        "geom": {"rgba", "friction", "mass", "size", "pos", "quat"},
        "light": {"pos", "dir", "diffuse", "ambient"},
        "camera": {"pos", "quat", "fovy"},
    }
    for section, entries in parameters.items():
        if section not in allowed:
            raise ValueError(f"Unsupported section: {section}")
        for name, attrs in entries.items():
            node = root.find(f".//{section}[@name='{name}']")
            if node is None:
                raise ValueError(f"Unknown {section}: {name}")
            for key, value in attrs.items():
                if key not in allowed[section]:
                    raise ValueError(f"Unsupported {section} attribute: {key}")
                numbers = np.asarray(value, dtype=float).reshape(-1)
                if not np.isfinite(numbers).all():
                    raise ValueError("Scene parameters must be finite")
                if key == "quat":
                    for alternative in ("euler", "xyaxes", "zaxis", "axisangle"):
                        node.attrib.pop(alternative, None)
                node.set(key, vec(numbers))


def write_scenes():
    out = ROOT / "scenes"
    out.mkdir(exist_ok=True)
    for robot in ROBOTS:
        for task in TASKS:
            xml = build_scene(robot, task)
            from physical_ai.env import ManipulationEnv

            with ManipulationEnv(robot, task) as env:
                tree = ET.fromstring(xml)
                for geom in tree.iter("geom"):
                    name = geom.get("name")
                    if name and name.startswith("cup"):
                        geom.set("rgba", vec(env.model.geom(name).rgba))
                keys = add(tree, "keyframe")
                add(
                    keys,
                    "key",
                    name="home",
                    qpos=vec(env.data.qpos),
                    ctrl=vec(env.data.ctrl),
                )
                ET.indent(tree)
                xml = ET.tostring(tree, encoding="unicode")
            # Saved scenes remain relocatable after clone/export.
            folder = ROBOTS[robot][0]
            xml = xml.replace(
                str(ROOT / "assets" / folder / "assets"), f"../assets/{folder}/assets"
            )
            (out / f"{robot}_{task}.xml").write_text(xml)


if __name__ == "__main__":
    write_scenes()
