@echo off
rem O2xygen dev server launcher. Double-click or run from anywhere.
rem Data (MIST grids, worlds) is expected under D:\test3\data (see README).
cd /d %~dp0
set PYTHONIOENCODING=utf-8
".\.venv\Scripts\python.exe" server\app.py
