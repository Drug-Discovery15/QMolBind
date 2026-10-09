#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
$PY scripts/v4.py explore
$PY scripts/v4.py confirm
echo V4_DONE
