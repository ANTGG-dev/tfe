@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set "FLOW_PYTHON=python"
if exist "%~dp0revision\.venv\Scripts\python.exe" set "FLOW_PYTHON=%~dp0revision\.venv\Scripts\python.exe"
"%FLOW_PYTHON%" -X utf8 "%~dp0reproduce_flow.py" --prepare --run
set RESULT=%errorlevel%
if not "%RESULT%"=="0" echo FloW stopped with an error. Inspect the log before resuming.
pause
exit /b %RESULT%
