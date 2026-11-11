import sys

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CONFIG_DIR = REPO / "scalable_real2sim" / "ur7e_payload_id" / "configs"


def load_config(path_or_name: str) -> dict:
    p = Path(path_or_name)
    if not p.exists():
        p = CONFIG_DIR / f"{path_or_name}.yaml"
    return yaml.safe_load(p.read_text())
