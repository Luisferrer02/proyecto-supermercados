"""
Model persistence helpers
=========================
Saves trained model weights to a versioned path while keeping a stable
`latest` symlink-free alias so downstream scripts (05_predict.py) always
know where to load from.

Layout under mlops/results/:
    mlp.pth                                  <- canonical "latest" (for 05_predict.py)
    transformer.pth                          <- canonical "latest"
    models/
        mlp_20250421_143210_a1b2c3.pth       <- timestamped archive
        transformer_20250421_143210_a1b2c3.pth
        manifest.json                         <- registry of every saved version

The canonical path is always overwritten with the newest model; the
archive directory accumulates every run so previous models can be
diffed or rolled back. The manifest tracks metadata (timestamp, epochs,
metrics, origin script) for audit.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

MANIFEST_FILENAME = "manifest.json"


def _state_dict_hash(state_dict: Dict[str, Any]) -> str:
    """Compute a short content hash of a state dict for traceability."""
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        t = state_dict[k]
        if isinstance(t, torch.Tensor):
            h.update(k.encode("utf-8"))
            h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:8]


def save_model(
    model: torch.nn.Module,
    name: str,
    results_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Save a model weights to both canonical and versioned paths.

    Returns a dict describing the saved version:
        {
          "name": "mlp",
          "canonical_path": "results/mlp.pth",
          "archive_path":   "results/models/mlp_20250421_143210_a1b2c3.pth",
          "hash":           "a1b2c3",
          "timestamp":      "20250421_143210",
        }
    """
    results_dir = Path(results_dir)
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    state = model.state_dict()
    short_hash = _state_dict_hash(state)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    archive_name = f"{name}_{ts}_{short_hash}.pth"
    archive_path = models_dir / archive_name
    canonical_path = results_dir / f"{name}.pth"

    torch.save(state, archive_path)
    torch.save(state, canonical_path)

    record = {
        "name": name,
        "canonical_path": str(canonical_path.relative_to(results_dir)),
        "archive_path": str(archive_path.relative_to(results_dir)),
        "hash": short_hash,
        "timestamp": ts,
        "size_bytes": archive_path.stat().st_size,
    }
    if metadata:
        record["metadata"] = metadata

    _append_manifest(models_dir, record)
    return record


def _append_manifest(models_dir: Path, record: Dict[str, Any]) -> None:
    """Append a save record to models/manifest.json, keeping full history."""
    import fcntl

    manifest_path = models_dir / MANIFEST_FILENAME
    lock_path = models_dir / f".{MANIFEST_FILENAME}.lock"

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            history: list = []
            if manifest_path.exists():
                try:
                    history = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if not isinstance(history, list):
                        history = []
                except json.JSONDecodeError:
                    history = []

            history.append(record)
            manifest_path.write_text(
                json.dumps(history, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

