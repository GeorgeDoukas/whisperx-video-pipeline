"""Resume-safe checkpoint management.

The checkpoint is a single JSON file stored alongside the output files.
Each pipeline stage calls :meth:`Checkpoint.mark_done` when it finishes so
that a subsequent run can skip already-completed stages.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class Checkpoint:
    """Persists pipeline stage results to a JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Dict[str, Any] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                self.data = json.load(fh)

    # ------------------------------------------------------------------
    # Generic key/value helpers
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self._save()

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def is_done(self, stage: str) -> bool:
        return bool(self.data.get(f"{stage}_done", False))

    def mark_done(self, stage: str, data: Any = None) -> None:
        self.data[f"{stage}_done"] = True
        if data is not None:
            self.data[f"{stage}_data"] = data
        self._save()

    def get_stage_data(self, stage: str) -> Optional[Any]:
        return self.data.get(f"{stage}_data")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, ensure_ascii=False, indent=2)
