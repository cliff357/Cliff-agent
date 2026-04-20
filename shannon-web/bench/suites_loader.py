"""Load benchmark suites from YAML files."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

_SUITES_DIR = Path(__file__).parent / "suites"


def list_suites() -> list[dict]:
    """Return metadata for all available suites."""
    out = []
    if not _SUITES_DIR.exists():
        return out
    for f in sorted(_SUITES_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(f.read_text()) or {}
            out.append({
                "id": f.stem,
                "name": data.get("name", f.stem),
                "description": data.get("description", ""),
                "task_count": len(data.get("tasks", [])),
                "path": str(f),
            })
        except Exception as e:
            out.append({"id": f.stem, "name": f.stem, "error": str(e)})
    return out


def load_suite(suite_id: str) -> dict[str, Any]:
    """Load a suite's full YAML content."""
    # Safe filename (no path traversal)
    safe = Path(suite_id).name
    path = _SUITES_DIR / f"{safe}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Suite not found: {suite_id}")
    data = yaml.safe_load(path.read_text()) or {}
    data["id"] = safe
    return data
