from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_SIZE = 0.9

DATASET_DIR = Path("data/datasets")
OUTPUT_DIR = Path("data/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _filter_shared_tracks(playlists, tracks):
    common_uris = set(tracks.track_uri) & set(playlists.track_uri)
    tracks = tracks[tracks.track_uri.isin(common_uris)]
    playlists = playlists[playlists.track_uri.isin(common_uris)]
    return playlists, tracks


def _train_test_split(playlists, train_size: float):
    playlist_ids = playlists.playlist_id.unique()
    rng = np.random.default_rng(seed=42)
    rng.shuffle(playlist_ids)

    split = int(train_size * len(playlist_ids))
    train_ids = playlist_ids[:split]
    test_ids = playlist_ids[split:]

    train_playlists = playlists[playlists.playlist_id.isin(set(train_ids))]
    test_playlists = playlists[playlists.playlist_id.isin(set(test_ids))]

    return train_playlists, test_playlists


def _map_to_index(values):
    return {v: i for i, v in enumerate(sorted(values.unique()), 1)}


def _make_padding_row(df):
    pad_values = {}

    for col, dtype in df.dtypes.items():
        if np.issubdtype(dtype, np.number):
            pad_values[col] = 0
        else:
            pad_values[col] = "<PAD>"
    
    return pd.DataFrame([pad_values])


def _minmax_scale(tracks, train_tracks, features):
    for feature in features:
        min_val = train_tracks[feature].min()
        max_val = train_tracks[feature].max()
        tracks[feature] = (tracks[feature] - min_val) / (max_val - min_val)
    return tracks


def _count_train_observations(train_playlists, tracks):
    n_obs = train_playlists.groupby("track_uri").size().reset_index(name="n_obs")
    tracks = tracks.merge(n_obs, on="track_uri", how="left")
    tracks["n_obs"] = tracks["n_obs"].fillna(0).astype(int)
    return tracks


def _playlists_to_sequences(playlists):
    return (
        playlists
        .sort_values(["playlist_id", "pos"])
        .groupby(["playlist_id", "playlist_name"])
        .track_id
        .apply(list)
        .reset_index()
        .rename(columns={"track_id": "playlist"})
    )


def main():
    playlists = pd.read_parquet(DATASET_DIR / "playlists")
    tracks = pd.read_parquet(DATASET_DIR / "tracks")

    playlists, tracks = _filter_shared_tracks(playlists, tracks)

    train_playlists, test_playlists = _train_test_split(playlists, TRAIN_SIZE)
    train_tracks = tracks[tracks.track_uri.isin(train_playlists.track_uri.unique())]

    track_uri_to_id = _map_to_index(tracks.track_uri)

    artist_name_to_index = _map_to_index(train_tracks.artist_name)  # train only

    mode_to_index = _map_to_index(tracks["mode"])
    key_to_index = _map_to_index(tracks.key)
    time_signature_to_index = _map_to_index(tracks.time_signature)

    tracks["track_id"] = tracks.track_uri.map(track_uri_to_id)
    train_playlists["track_id"] = train_playlists.track_uri.map(track_uri_to_id)
    test_playlists["track_id"] = test_playlists.track_uri.map(track_uri_to_id)

    # artist who are not seen during training are mapped to padding zero-index
    tracks["artist_index"] = tracks.artist_name.map(artist_name_to_index)
    tracks.artist_index = tracks.artist_index.fillna(0).astype(int)

    tracks["mode_index"] = tracks["mode"].map(mode_to_index)
    tracks["key_index"] = tracks.key.map(key_to_index)
    tracks["time_signature_index"] = tracks.time_signature.map(time_signature_to_index)

    tracks = _minmax_scale(tracks, train_tracks, ["loudness", "tempo"])

    pad_row = _make_padding_row(tracks)
    tracks = pd.concat([pad_row, tracks])

    tracks = _count_train_observations(train_playlists, tracks)

    train_playlists = _playlists_to_sequences(train_playlists)
    test_playlists = _playlists_to_sequences(test_playlists)

    train_playlists.to_parquet(OUTPUT_DIR / "train.parquet")
    test_playlists.to_parquet(OUTPUT_DIR / "test.parquet")
    tracks.to_parquet(OUTPUT_DIR / "tracks.parquet")


if __name__ == "__main__":
    main()
