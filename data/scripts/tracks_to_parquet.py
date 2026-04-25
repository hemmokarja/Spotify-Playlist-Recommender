from pathlib import Path

import pandas as pd
import sqlite3

OUTPUT_DIR = Path("data/datasets/tracks/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SQL = """
SELECT
    track_uri,
    track_name,
    artist_name,
    danceability,
    energy,
    loudness,
    speechiness,
    acousticness,
    instrumentalness,
    liveness,
    valence,
    tempo,
    mode,
    key,
    time_signature
FROM extracted
"""


def main():
    conn = sqlite3.connect("data/raw-datasets/tracks/extracted.sqlite")
    df = pd.read_sql_query(SQL, conn)
    out_filepath = OUTPUT_DIR / "tracks.parquet"
    df.to_parquet(out_filepath, engine="pyarrow", index=False)
    conn.close()


if __name__ == "__main__":
    main()
