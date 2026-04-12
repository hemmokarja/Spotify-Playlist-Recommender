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

    def __init__(self, tracks):
        self.tracks = tracks.sort_values("track_id").reset_index(drop=True)

        self.cont_feat_mapping = self.tracks[self.CONT_FEATURES].to_numpy()
        self.cat_feat_mapping = self.tracks[self.CAT_FEATURES].to_numpy()
        self.artist_mapping = self.tracks.artist_index.to_numpy()

    def tensorise(
        self, playlist_name: str, playlist: np.ndarray, inference: bool = False
    ) -> dict:
        if not inference:
            x = playlist[:-1]
            y = playlist
        else:
            x = playlist

        x_cont = self.cont_feat_mapping[x]  # [T-1, n_cont]
        x_cat = self.cat_feat_mapping[x]  # [T-1, n_cat]
        x_artist = self.artist_mapping[x]  # [T-1]
        sample = {
            "name": playlist_name,
            "x_artist": torch.from_numpy(x_artist).to(torch.long),
            "x_cont": torch.from_numpy(x_cont).to(torch.float32),
            "x_cat": torch.from_numpy(x_cat).to(torch.long),
        }
        if not inference:
            sample["y"] = torch.from_numpy(y).to(torch.long)  # [T]
        return sample

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        max_x_len = max(sample["x_cont"].shape[0] for sample in batch)
        x_pads = [max_x_len - sample["x_cont"].shape[0] for sample in batch]

        def _pad2d(key, pads):
            return torch.stack(
                [F.pad(sample[key], (0, 0, 0, pad)) for sample, pad in zip(batch, pads)]
            )

        def _pad1d(key, pads):
            return torch.stack(
                [F.pad(sample[key], (0, pad)) for sample, pad in zip(batch, pads)]
            )

        collated_batch = {
            "name": [sample["name"] for sample in batch],
            "x_artist": _pad1d("x_artist", x_pads),
            "x_cont": _pad2d("x_cont", x_pads),
            "x_cat": _pad2d("x_cat", x_pads),
        }

        if "y" in batch[0]:
            max_y_len = max(sample["y"].shape[0] for sample in batch)
            y_pads = [max_y_len - sample["y"].shape[0] for sample in batch]
            collated_batch["y"] = _pad1d("y", y_pads)

        return collated_batch

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
