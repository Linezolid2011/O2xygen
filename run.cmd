@echo off
rem O2xygen dev server launcher. Double-click or run from anywhere.
rem Data lives in the repo: .data\MIST (grids) and .data\worlds (saves,
rem overridable via O2YGEN_WORLDS_DIR).
cd /d %~dp0
set PYTHONIOENCODING=utf-8
".\.venv\Scripts\python.exe" server\app.py
