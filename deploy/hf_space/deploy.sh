#!/usr/bin/env bash
# Assembles a clean copy of exactly what streamlit_app/app.py needs and
# uploads it to the Hugging Face Space. Run from the repo root:
#   bash deploy/hf_space/deploy.sh
#
# Uses `git archive` rather than the working tree so stray local files
# (__pycache__, logs, scratch output) never end up on the Space.
set -euo pipefail

SPACE_ID="${HF_SPACE_ID:-Shuvamadh/fpl-ml-assistant}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

git archive HEAD | (mkdir -p "$STAGE/src_tree" && tar -x -C "$STAGE/src_tree")

mkdir -p "$STAGE/upload/gui"
cp -r "$STAGE/src_tree/src" "$STAGE/upload/src"
cp -r "$STAGE/src_tree/streamlit_app" "$STAGE/upload/streamlit_app"
cp "$STAGE/src_tree/gui/league_extras.py" "$STAGE/upload/gui/league_extras.py"
cp -r "$STAGE/src_tree/assets" "$STAGE/upload/assets"
cp -r "$STAGE/src_tree/models" "$STAGE/upload/models"
cp -r "$STAGE/src_tree/data" "$STAGE/upload/data"

cp "$STAGE/src_tree/streamlit_app/requirements.txt" "$STAGE/upload/requirements.txt"
cp deploy/hf_space/README.md "$STAGE/upload/README.md"
cp deploy/hf_space/packages.txt "$STAGE/upload/packages.txt"

hf upload "$SPACE_ID" "$STAGE/upload" . --repo-type space --commit-message "Deploy $(git rev-parse --short HEAD)"
