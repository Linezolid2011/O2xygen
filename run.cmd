@echo off
rem O2xygen dev server launcher. Double-click or run from anywhere.
rem Reference data lives in .data; user saves live in data\worlds\<name>\<name>.oxygen
rem The save root is overridable via O2YGEN_WORLDS_DIR.
cd /d %~dp0
set PYTHONIOENCODING=utf-8
rem Ctrl+C is handled by server/app.py with a [Y/n] confirmation prompt.
".\.venv\Scripts\python.exe" server\app.py
