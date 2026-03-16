from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .exceptions import SchemaLoadError


def load_schema(schemas_dir: Path, filename: str) -> dict[str, Any]:
    """Load and parse a JSON schema file.

    Args:
        schemas_dir: Directory containing schema files.
        filename: Schema filename to load.

    Returns:
        Parsed schema object.

    Raises:
        SchemaLoadError: If schema file does not exist.
    """
    path = schemas_dir / filename
    if not path.exists():
        raise SchemaLoadError(f"Schema file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
