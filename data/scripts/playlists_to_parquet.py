import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

INPUT_DIR = Path("data/raw-datasets/playlists/data")
OUTPUT_DIR = Path("data/datasets/playlists/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    filepaths = sorted(list(INPUT_DIR.glob("*.json")))

    for filepath in tqdm(filepaths, desc="Converting MPD JSONs to parquet..."):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        playlists = data["playlists"]

        df = pd.json_normalize(playlists, record_path=["tracks"], meta=["pid", "name"])
        df = df.rename(columns={"pid": "playlist_id", "name": "playlist_name"})
        df = df[["playlist_id", "playlist_name", "pos", "track_uri"]]

        out_filepath = OUTPUT_DIR / filepath.with_suffix(".parquet").name
        df.to_parquet(out_filepath, index=False, engine="pyarrow")

        del data
        del df

        filepath.unlink()  # remove original JSON file right away to free up disk space


if __name__ == "__main__":
    main()
