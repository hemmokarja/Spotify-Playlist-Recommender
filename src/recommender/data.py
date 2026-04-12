import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class Tensoriser:
    CONT_FEATURES = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]
    CAT_FEATURES = [
        "mode_index",
        "key_index",
        "time_signature_index",
    ]

    def __init__(self, tracks: pd.DataFrame):
        self.tracks = tracks.sort_values("track_id").reset_index(drop=True)

    def tensorise(
        self, playlist_name: str, playlist: np.ndarray, inference: bool = False
    ) -> dict:
        x = playlist[:-1] if not inference else playlist
        sample = {"name": playlist_name, "x": torch.from_numpy(x).to(torch.long)}
        if not inference:
            sample["y"] = torch.from_numpy(playlist).to(torch.long)
        return sample

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        max_x_len = max(s["x"].shape[0] for s in batch)
        x_pads = [max_x_len - s["x"].shape[0] for s in batch]
        collated = {
            "name": [s["name"] for s in batch],
            "x": torch.stack(
                [F.pad(s["x"], (0, pad)) for s, pad in zip(batch, x_pads)]
            ),
        }
        if "y" in batch[0]:
            max_y_len = max(s["y"].shape[0] for s in batch)
            y_pads = [max_y_len - s["y"].shape[0] for s in batch]
            collated["y"] = torch.stack(
                [F.pad(s["y"], (0, pad)) for s, pad in zip(batch, y_pads)]
            )
        return collated

    @property
    def artist_vocab_size(self):
        return int(self.tracks.artist_index.nunique())

    @property
    def cat_vocab_sizes(self):
        return [int(self.tracks[f].nunique()) for f in self.CAT_FEATURES]

    def as_dict(self):
        return self.tracks.to_dict("records")

    @classmethod
    def from_dict(cls, d):
        tracks = pd.DataFrame(d)
        return cls(tracks)


class PlaylistDataset(Dataset):
    def __init__(self, split, tensoriser: Tensoriser):
        self.df = pd.read_parquet(f".data/data/{split}.parquet")
        self.tensoriser = tensoriser

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        sample = self.tensoriser.tensorise(row.playlist_name, row.playlist)
        return sample
