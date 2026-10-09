"""Start the local QuMolBind web app: python scripts/serve.py [--port 8765] [--open]"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="local-only by default; the app has no authentication")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()
    import uvicorn

    if a.open:
        webbrowser.open(f"http://{a.host}:{a.port}")
    uvicorn.run("qumolbind.app.server:app", host=a.host, port=a.port, log_level="info")
