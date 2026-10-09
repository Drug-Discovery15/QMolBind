#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
$PY scripts/explore_v3.py
$PY scripts/confirm_v3.py
echo V3_DONE
