import os, subprocess, sys
print("ENV_PYTHONHOME", os.environ.get("PYTHONHOME"))
print("ENV_PYTHONPATH", os.environ.get("PYTHONPATH"))
py = r"C:\Users\maxkl\Downloads\vid_an\backend\venv\Scripts\python.exe"
r = subprocess.run([py, "-c", "import cv2; print('CV2_OK', cv2.__version__)"], capture_output=True, text=True)
print("RC", r.returncode)
print("OUT", r.stdout.strip())
print("ERR", r.stderr.strip()[-300:])
env = dict(os.environ); env.pop("PYTHONHOME", None); env.pop("PYTHONPATH", None)
r2 = subprocess.run([py, "-E", "-c", "import cv2; print('CV2_OK', cv2.__version__)"], capture_output=True, text=True, env=env)
print("RC2", r2.returncode)
print("OUT2", r2.stdout.strip())
print("ERR2", r2.stderr.strip()[-300:])
print("PROBE_DONE")
