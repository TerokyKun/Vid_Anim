"""Local model store driven by ``models/manifest.json``.

Models are never committed to Git. This module resolves a model *id* to a local
file, downloading it once from the URL declared in the manifest when needed.
No user video, frame or landmark ever leaves the machine — only public model
weights are fetched, and only if they are missing.

The manifest is intentionally data-only so users can drop their own ONNX models
in ``models/`` and add an entry.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional


class ModelError(RuntimeError):
    """Raised when a model cannot be found, resolved or verified."""


def repo_root() -> str:
    """Best-effort repository root (two levels above ``core/``)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def models_dir() -> str:
    env = os.environ.get("VIDAN_MODELS", "")
    if env:
        return env
    return os.path.join(repo_root(), "models")


def manifest_path() -> str:
    env = os.environ.get("VIDAN_MANIFEST", "")
    if env:
        return env
    return os.path.join(models_dir(), "manifest.json")


@dataclass
class ModelEntry:
    model_id: str
    kind: str
    file: str
    url: str = ""
    sha256: str = ""
    description: str = ""
    extra: Optional[Dict] = None

    @property
    def local_path(self) -> str:
        return os.path.join(models_dir(), self.file)


def load_manifest(path: Optional[str] = None) -> Dict[str, ModelEntry]:
    p = path or manifest_path()
    if not os.path.exists(p):
        raise ModelError(
            "Model manifest not found at {!r}. Set VIDAN_MANIFEST or create "
            "models/manifest.json.".format(p)
        )
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    out: Dict[str, ModelEntry] = {}
    for mid, spec in (data.get("models") or {}).items():
        known = {"kind", "file", "url", "sha256", "description"}
        out[mid] = ModelEntry(
            model_id=mid,
            kind=spec.get("kind", "unknown"),
            file=spec["file"],
            url=spec.get("url", ""),
            sha256=spec.get("sha256", ""),
            description=spec.get("description", ""),
            extra={k: v for k, v in spec.items() if k not in known},
        )
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        os.replace(tmp, dest)
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise ModelError("Failed to download model from {!r}: {}".format(url, exc)) from exc


def resolve_model(
    model_id: str,
    download: bool = True,
    manifest: Optional[Dict[str, ModelEntry]] = None,
    verify: bool = False,
) -> str:
    """Return the local path for ``model_id``, downloading it if required."""
    entries = manifest or load_manifest()
    if model_id not in entries:
        raise ModelError(
            "Unknown model id {!r}. Available: {}".format(
                model_id, ", ".join(sorted(entries)) or "(none)"
            )
        )
    entry = entries[model_id]
    path = entry.local_path
    if not os.path.exists(path):
        if not entry.url:
            raise ModelError(
                "Model {!r} is not present at {!r} and the manifest defines no "
                "download URL. Place the file there manually.".format(model_id, path)
            )
        if not download:
            raise ModelError(
                "Model {!r} missing at {!r} (download disabled).".format(model_id, path)
            )
        _download(entry.url, path)
    if verify and entry.sha256:
        got = _sha256(path)
        if got.lower() != entry.sha256.lower():
            raise ModelError(
                "Model {!r} checksum mismatch: expected {}, got {}".format(
                    model_id, entry.sha256, got
                )
            )
    return path


def provided(manifest: Optional[Dict[str, ModelEntry]] = None) -> List[str]:
    """Ids whose files already exist locally."""
    entries = manifest or load_manifest()
    return [mid for mid, e in entries.items() if os.path.exists(e.local_path)]
