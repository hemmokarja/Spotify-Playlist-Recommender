#!/bin/bash

# https://www.kaggle.com/datasets/himanshuwagh/spotify-million/

set -a
source .env
set +a

DATASET="himanshuwagh/spotify-million"
TARGET_DIR="data/raw-datasets/playlists"

uv run kaggle datasets download -d $DATASET -p $TARGET_DIR --unzip

uv run python data/scripts/playlists_to_parquet.py

rm -rf $TARGET_DIR
