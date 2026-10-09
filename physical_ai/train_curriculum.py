"""Run an explicit, inspectable PPO curriculum recipe and its dependencies."""

import argparse
import hashlib
import json
from pathlib import Path
from physical_ai.reset_states import bank_digest
import shutil

from physical_ai.rl import train, load_rl, sha256
from physical_ai.scenes import ROOT, ROBOTS, TASKS


def dependency_order(recipe, target):
    if target not in {f"{robot}/{task}" for robot in ROBOTS for task in TASKS}:
        raise ValueError(f"Invalid robot/task target: {target}")
    nodes = {node["id"]: node for node in recipe["stages"]}
    if len(nodes) != len(recipe["stages"]):
        raise ValueError("Duplicate curriculum stage ID")
    if target not in recipe["outputs"]:
        raise ValueError(f"Unknown target: {target}")
    ordered, active, visited = [], set(), set()

    def visit(key):
        if key in active:
            raise ValueError("Curriculum dependency cycle")
        if key in visited:
            return
        if key not in nodes or Path(key).name != key or key in (".", ".."):
            raise ValueError(f"Invalid stage ID: {key}")
        node = nodes[key]
        active.add(key)
        parents = [node[k] for k in ("resume_from", "transfer_from") if node.get(k)]
        if len(parents) > 1:
            raise ValueError("A stage cannot both resume and transfer")
        for parent in parents:
            visit(parent)
        active.remove(key)
        visited.add(key)
        ordered.append(node)

    visit(recipe["outputs"][target])
    output = nodes[recipe["outputs"][target]]
    if target != f'{output["robot"]}/{output["task"]}':
        raise ValueError("Requested target does not match output robot/task")
    return ordered


def run_recipe(recipe_path, target, out):
    recipe = json.loads(Path(recipe_path).read_text())
    if recipe.get("schema") != 1:
        raise ValueError("Unknown curriculum recipe schema")
    out = Path(out).resolve()
    for node in dependency_order(recipe, target):
        stage_dir = out / node["id"]
        kwargs = dict(node["train"])
        bank = kwargs.get("curriculum_bank")
        if bank:
            kwargs["curriculum_bank"] = str(ROOT / bank)
        parent_hashes = {
            field: sha256(out / node[field] / "last.zip")
            for field in ("resume_from", "transfer_from")
            if node.get(field)
        }
        fingerprint = dict(
            node=node,
            parent_sha256=parent_hashes,
            bank_sha256=bank_digest(ROOT / bank) if bank else None,
            scenes_sha256=sha256(ROOT / "physical_ai/scenes.py"),
            env_sha256=sha256(ROOT / "physical_ai/env.py"),
            trainer_sha256=sha256(ROOT / "physical_ai/rl.py"),
            policy_sha256=sha256(ROOT / "physical_ai/curriculum.py"),
        )
        node_hash = hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True).encode()
        ).hexdigest()
        record = stage_dir / "recipe_node.json"
        if record.exists() and json.loads(record.read_text())["sha256"] != node_hash:
            raise ValueError(
                f'Recipe or environment changed for existing stage {node["id"]}'
            )
        checkpoint = stage_dir / "last.zip"
        if checkpoint.exists():
            if not record.exists():
                raise ValueError(
                    f"Existing checkpoint lacks recipe provenance: {checkpoint}"
                )
            load_rl(checkpoint, node["robot"], node["task"])
            print("Reuse verified stage", node["id"], flush=True)
            continue
        stage_dir.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(dict(sha256=node_hash, **fingerprint), indent=2))
        for field, argument in [
            ("resume_from", "resume"),
            ("transfer_from", "transfer"),
        ]:
            if node.get(field):
                kwargs[argument] = str(out / node[field] / "last.zip")
        print("Train stage", node["id"], flush=True)
        train(node["robot"], node["task"], stage_dir, **kwargs)
    source = out / recipe["outputs"][target] / "last.zip"
    final = out / "experts" / target / "last.zip"
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, final)
    shutil.copy2(source.with_suffix(".json"), final.with_suffix(".json"))
    manifest_path = out / "outputs.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[target] = dict(
        checkpoint=str(final.relative_to(out)),
        sha256=sha256(final),
        source_stage=recipe["outputs"][target],
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("Final checkpoint:", final, flush=True)
    return final


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", required=True)
    p.add_argument(
        "--target", required=True, help="robot/task, for example ur5e/cup_plate"
    )
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run_recipe(a.recipe, a.target, a.out)
