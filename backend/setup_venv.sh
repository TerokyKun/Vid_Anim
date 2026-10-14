#!/usr/bin/env bash
# Create the MediaPipe backend virtualenv for vid_an (Linux / WSL).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 -m venv "$HERE/venv"
"$HERE/venv/bin/python" -m pip install --upgrade pip
"$HERE/venv/bin/pip" install -r "$HERE/requirements.txt"
echo
echo "Backend ready."
echo "NOTE: when driving Blender on Windows from WSL, point 'Backend Python' to a"
echo "Windows python.exe with mediapipe installed (pip via setup_windows.bat)."
