#!/bin/bash

# https://www.kaggle.com/datasets/krishsharma0413/2-million-songs-from-mpd-with-audio-features

set -a
source .env
set +a

DATASET="krishsharma0413/2-million-songs-from-mpd-with-audio-features"
TARGET_DIR="data/raw-datasets/tracks"

uv run kaggle datasets download -d $DATASET -p $TARGET_DIR --unzip

uv run python data/scripts/tracks_to_parquet.py

rm -rf $TARGET_DIR
