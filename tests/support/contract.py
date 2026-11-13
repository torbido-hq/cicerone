from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONTRACT_DIR = Path(__file__).resolve().parents[1] / "contract"


def load_contract(name: str) -> Any:
    return json.loads((CONTRACT_DIR / name).read_text())
