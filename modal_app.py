"""Single-command Modal entrypoint for milestone PPO training.

Usage:
    modal run modal_app.py --config configs/ppo_debug.yaml
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import modal
import yaml


APP_NAME = "cs285-final-project-ppo"
PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = "/root/project"
OUTPUT_VOLUME_NAME = "cs285-final-project-outputs"
OUTPUT_MOUNT_PATH = f"{REMOTE_ROOT}/outputs"


app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "pyyaml", "matplotlib")
    .add_local_file(PROJECT_ROOT / "pyproject.toml", remote_path=f"{REMOTE_ROOT}/pyproject.toml")
    .add_local_dir(PROJECT_ROOT / "src", remote_path=f"{REMOTE_ROOT}/src")
    .add_local_dir(PROJECT_ROOT / "scripts", remote_path=f"{REMOTE_ROOT}/scripts")
    .add_local_dir(PROJECT_ROOT / "configs", remote_path=f"{REMOTE_ROOT}/configs")
    .add_local_dir(PROJECT_ROOT / "sample_data", remote_path=f"{REMOTE_ROOT}/sample_data")
)


def _rewrite_config_for_modal(config_path: str) -> str:
    source_path = Path(REMOTE_ROOT) / config_path
    config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} must be a mapping.")

    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError(f"Config at {config_path} is missing 'experiment'.")
    experiment["output_dir"] = OUTPUT_MOUNT_PATH

    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"Config at {config_path} is missing 'data'.")
    for key in ("train_path", "val_path", "test_path"):
        value = data.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Config field data.{key} must be a non-empty string.")
        data[key] = str(Path(REMOTE_ROOT) / value)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
        return handle.name


@app.function(
    image=image,
    cpu=4,
    timeout=60 * 60,
    volumes={OUTPUT_MOUNT_PATH: output_volume},
)
def train_ppo_remote(config: str = "configs/ppo_debug.yaml") -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    if REMOTE_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_ROOT)

    from scripts.train_ppo import train

    temp_config_path = _rewrite_config_for_modal(config)
    run_dir = Path(train(Path(temp_config_path)))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    output_volume.commit()
    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "volume_name": OUTPUT_VOLUME_NAME,
    }


@app.local_entrypoint()
def main(config: str = "configs/ppo_debug.yaml") -> None:
    result = train_ppo_remote.remote(config=config)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        "Artifacts persisted to Modal volume "
        f"{OUTPUT_VOLUME_NAME}. Example download:\n"
        f"  modal volume get {OUTPUT_VOLUME_NAME} {Path(result['run_dir']).name}/summary.json summary.json"
    )
