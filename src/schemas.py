from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .exceptions import SchemaLoadError


def load_schema(schemas_dir: Path, filename: str) -> Dict[str, Any]:
    """Load a JSON schema file from disk.

    Args:
        schemas_dir: Root directory containing schema files.
        filename: Schema filename relative to ``schemas_dir``.

    Returns:
        Parsed schema as a dictionary.

    Raises:
        SchemaLoadError: If the schema file is missing or invalid JSON.
    """
    path = schemas_dir / filename
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SchemaLoadError(f"Schema file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaLoadError(f"Schema file is not valid JSON: {path}") from exc
