#!/usr/bin/env python3
"""Download Lumbra's Gigabase OTB PGN 7z archives.

The official Lumbras page redirects archive downloads to MEGA file URLs. This
script discovers the current OTB PGN package links, follows the redirect, and
streams/decrypts MEGA files to data/raw/lumbras/otb.
"""
from __future__ import annotations

import base64
import html
import json
import os
import re
import struct
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

PAGE_URL = "https://lumbrasgigabase.com/en/download-in-pgn-format-en/"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "lumbras" / "otb"
CHUNK_SIZE = 1024 * 1024


@dataclass
class Package:
    title: str
    size_label: str
    lumbras_url: str

    @property
    def filename(self) -> str:
        slug = self.title.lower().replace("–", "-").replace(">", "gt")
        slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
        return f"lumbras-{slug}.7z"


def request_bytes(url: str, *, data: bytes | None = None, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "chess-data-downloader/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def discover_packages() -> list[Package]:
    page = request_bytes(PAGE_URL).decode("utf-8", errors="replace")
    soup = BeautifulSoup(page, "html.parser")
    packages: list[Package] = []
    for card in soup.select(".card.card-default"):
        title_el = card.select_one(".ptitle")
        link_el = card.select_one("a[data-downloadurl]")
        footer_el = card.select_one(".card-footer")
        if not title_el or not link_el:
            continue
        title = title_el.get_text(" ", strip=True)
        # Full OTB corpus only. Exclude pre-filtered elite and online packages.
        if not title.startswith("OTB") or "Elite" in title:
            continue
        size_label = " ".join(footer_el.get_text(" ", strip=True).split()) if footer_el else ""
        packages.append(Package(title, size_label, html.unescape(link_el["data-downloadurl"])))
    return packages


def resolve_mega_url(lumbras_url: str) -> str:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    req = urllib.request.Request(lumbras_url, headers={"User-Agent": "chess-data-downloader/0.1"})
    try:
        opener.open(req, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code in (301, 302, 303, 307, 308):
            location = exc.headers.get("Location")
            if location and "mega.nz/file/" in location:
                return location
        raise
    raise RuntimeError(f"No MEGA redirect found for {lumbras_url}")


def b64url_decode(value: str) -> bytes:
    value += "=" * ((4 - len(value) % 4) % 4)
    return base64.urlsafe_b64decode(value)


def mega_file_info(mega_url: str) -> tuple[str, int, bytes, bytes]:
    handle = mega_url.split("/file/", 1)[1].split("#", 1)[0]
    key_fragment = mega_url.split("#", 1)[1]
    key_bytes = b64url_decode(key_fragment)
    words = list(struct.unpack(f">{len(key_bytes) // 4}I", key_bytes))
    if len(words) < 8:
        raise ValueError("Unexpected MEGA file key length")
    file_key_words = [words[i] ^ words[i + 4] for i in range(4)]
    file_key = struct.pack(">4I", *file_key_words)
    iv = struct.pack(">4I", words[4], words[5], 0, 0)

    body = json.dumps([{"a": "g", "g": 1, "p": handle}]).encode()
    response = request_bytes("https://g.api.mega.co.nz/cs?id=1", data=body)
    payload = json.loads(response)
    if isinstance(payload[0], int):
        raise RuntimeError(f"MEGA API error: {payload[0]}")
    return payload[0]["g"], int(payload[0]["s"]), file_key, iv


def download_mega(mega_url: str, dest: Path) -> None:
    dl_url, size, file_key, iv = mega_file_info(mega_url)
    if dest.exists() and dest.stat().st_size == size and dest.read_bytes()[:6] == bytes.fromhex("377abcaf271c"):
        print(f"skip existing {dest.name} ({size:,} bytes)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    cipher = Cipher(algorithms.AES(file_key), modes.CTR(iv), backend=default_backend()).decryptor()
    req = urllib.request.Request(dl_url, headers={"User-Agent": "chess-data-downloader/0.1"})
    with urllib.request.urlopen(req, timeout=180) as response, tmp.open("wb") as out:
        done = 0
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            out.write(cipher.update(chunk))
            done += len(chunk)
            pct = done / size * 100 if size else 0
            print(f"  {dest.name}: {done:,}/{size:,} bytes ({pct:5.1f}%)", end="\r", flush=True)
        out.write(cipher.finalize())
    print()
    if tmp.read_bytes()[:6] != bytes.fromhex("377abcaf271c"):
        raise RuntimeError(f"Downloaded file does not look like a 7z archive: {tmp}")
    os.replace(tmp, dest)


def main() -> int:
    packages = discover_packages()
    if not packages:
        print("No OTB packages found", file=sys.stderr)
        return 1
    print(f"Found {len(packages)} OTB packages")
    for pkg in packages:
        print(f"- {pkg.title} [{pkg.size_label}] -> {pkg.filename}")

    for pkg in packages:
        dest = OUT_DIR / pkg.filename
        print(f"\nDownloading {pkg.title}")
        mega_url = resolve_mega_url(pkg.lumbras_url)
        download_mega(mega_url, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
