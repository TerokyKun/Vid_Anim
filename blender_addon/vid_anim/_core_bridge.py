"""Locate the shared ``core`` package from inside the Blender add-on.

The add-on may be installed as a zip copy inside Blender's ``scripts/addons``
while the ``core`` package lives next to the repository. We search a few
candidate roots (an env var, the repo layout, and known paths) and expose the
imported module as ``core``. If it cannot be found the attribute is ``None`` and
callers fall back to the add-on's built-in behaviour.
"""

from __future__ import annotations

import os
import sys


def _candidate_roots():
    here = os.path.dirname(os.path.abspath(__file__))
    roots = []
    env = os.environ.get("VIDAN_ROOT", "")
    if env:
        roots.append(env)
    # .../vid_an/blender_addon/vid_an -> repo root is three levels up
    roots.append(os.path.normpath(os.path.join(here, "..", "..", "..")))
    roots.append(os.path.normpath(os.path.join(here, "..", "..")))
    roots.append(r"C:\Users\maxkl\Downloads\vid_an")
    out = []
    for r in roots:
        if r not in out:
            out.append(r)
    return out


def load_core():
    for root in _candidate_roots():
        if os.path.isdir(os.path.join(root, "core")):
            if root not in sys.path:
                sys.path.insert(0, root)
            try:
                import core  # type: ignore
                return core
            except Exception:  # noqa: BLE001
                continue
    return None


core = load_core()
