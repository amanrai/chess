#!/usr/bin/env bash
set -euo pipefail

# Install or stage a Stockfish binary at the repo-local path expected by the
# BSE neighbor explorer backend.
#
# Default target:
#   tools/stockfish/stockfish
#
# Usage:
#   bash install-stockfish.sh
#   bash install-stockfish.sh /custom/path/to/stockfish

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$ROOT/tools/stockfish/stockfish}"
TARGET_DIR="$(dirname "$TARGET")"
mkdir -p "$TARGET_DIR"

find_stockfish() {
  if command -v stockfish >/dev/null 2>&1; then
    command -v stockfish
    return 0
  fi
  for p in \
    /usr/games/stockfish \
    /usr/local/bin/stockfish \
    /opt/homebrew/bin/stockfish \
    /usr/bin/stockfish; do
    if [[ -x "$p" ]]; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

install_with_package_manager() {
  if command -v brew >/dev/null 2>&1; then
    brew install stockfish
    return 0
  fi
  if command -v apt-get >/dev/null 2>&1; then
    if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
      apt-get update
      apt-get install -y stockfish
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install -y stockfish
    else
      echo "apt-get is available but sudo is not; install stockfish manually." >&2
      return 1
    fi
    return 0
  fi
  echo "No supported package manager found. Install stockfish manually, then rerun this script." >&2
  return 1
}

if ! FOUND="$(find_stockfish)"; then
  install_with_package_manager
  FOUND="$(find_stockfish)"
fi

# Prefer a symlink so package-manager updates apply automatically. Fall back to copy.
rm -f "$TARGET"
if ln -s "$FOUND" "$TARGET" 2>/dev/null; then
  :
else
  cp "$FOUND" "$TARGET"
  chmod +x "$TARGET"
fi

echo "Stockfish staged at: $TARGET"
"$TARGET" bench 1 >/dev/null 2>&1 || echo "Warning: staged binary did not pass a quick bench smoke test." >&2
