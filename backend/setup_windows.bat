@echo off
REM Create the MediaPipe backend virtualenv for vid_an (Windows).
set HERE=%~dp0
where py >nul 2>nul
if %errorlevel%==0 (
  py -3.11 -m venv "%HERE%venv"
) else (
  python -m venv "%HERE%venv"
)
call "%HERE%venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r "%HERE%requirements.txt"
echo.
echo Backend ready.
echo In Blender, set "Backend Python" to:
echo   %HERE%venv\Scripts\python.exe
pause
