@echo off
setlocal
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
pushd "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m evofact.cli %*
set "EVOFACT_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EVOFACT_EXIT_CODE%
