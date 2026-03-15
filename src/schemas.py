from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_schema(schemas_dir: Path, filename: str) -> Dict[str, Any]:
    path = schemas_dir / filename
    return json.loads(path.read_text(encoding="utf-8"))
