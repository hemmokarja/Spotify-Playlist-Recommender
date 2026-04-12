#!/bin/bash
set -euo pipefail

data/scripts/load-playlists.sh
data/scripts/load-tracks.sh
uv run python data/scripts/prepare_data.py

rm -rf data/raw-datasets
rm -rf data/datasets
