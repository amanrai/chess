#!/usr/bin/env python3
"""Serve the PGN visualizer locally to avoid browser file:// restrictions."""
from __future__ import annotations

import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 8765
ROOT = Path(__file__).resolve().parents[1]

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

if __name__ == "__main__":
    url = f"http://localhost:{PORT}/tools/pgn_visualizer.html"
    print(f"Serving {ROOT}")
    print(f"Open {url}")
    webbrowser.open(url)
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        httpd.serve_forever()
