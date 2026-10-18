#!/usr/bin/env bash
# Fetch the official doScenes repository (loader, metrics.py, annotations) at the pinned commit.
# Usage: scripts/setup_third_party.sh
# The doScenes code and annotations belong to their authors and are not redistributed here.
set -euo pipefail

URL="https://github.com/rossgreer/doScenes"
COMMIT="a5c3c2545a88525337f380363a19c4d1b0297119"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/doScenes"

if [ -d "$DEST/.git" ] || [ -f "$DEST/.git" ]; then
  git -C "$DEST" fetch --quiet origin
else
  mkdir -p "$(dirname "$DEST")"
  git clone --quiet "$URL" "$DEST"
fi
git -C "$DEST" checkout --quiet "$COMMIT"
for f in dataloader.py metrics.py Annotations; do
  [ -e "$DEST/$f" ] || { echo "ERROR: $DEST/$f missing at $COMMIT"; exit 1; }
done
echo "doScenes at $(git -C "$DEST" rev-parse --short HEAD) in $DEST"
