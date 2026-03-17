from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .exceptions import SchemaLoadError

_schema_cache: dict[tuple[str, str], dict[str, Any]] = {}


def load_schema(schemas_dir: Path, filename: str) -> dict[str, Any]:
    """Load and parse a JSON schema file (cached after first read).

    Args:
        schemas_dir: Directory containing schema files.
        filename: Schema filename to load.

    Returns:
        Parsed schema object.

    Raises:
        SchemaLoadError: If schema file does not exist.
    """
    cache_key = (str(schemas_dir), filename)
    cached = _schema_cache.get(cache_key)
    if cached is not None:
        return cached
    path = schemas_dir / filename
    if not path.exists():
        raise SchemaLoadError(f"Schema file not found: {path}")
    schema = json.loads(path.read_text(encoding="utf-8"))
    _schema_cache[cache_key] = schema
    return schema
