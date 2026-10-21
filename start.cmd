@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -c "import tkinter, numpy, cv2, PIL, tifffile" >nul 2>&1
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto failed
)
start "" /b ".venv\Scripts\pythonw.exe" "%~dp0launch.py"
exit /b 0
:failed
echo Setup failed. Check Python installation and internet connection.
pause
exit /b 1
