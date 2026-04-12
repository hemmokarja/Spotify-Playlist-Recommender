import os
from dataclasses import asdict

import pandas as pd
import structlog
import torch
import torch.nn as nn

from recommender.data import Tensoriser
from recommender.layers import PlaylistNameEmbedder, TrackEmbedder
from recommender.model_config import ModelConfig

logger = structlog.get_logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PlaylistRecommender(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        tensoriser: Tensoriser,
        name_embedder: PlaylistNameEmbedder,
        track_embedder = TrackEmbedder,
    ):
        super().__init__()
        self.config = config
        self.tensoriser = tensoriser

        self.name_embedder = name_embedder
        self.track_embedder = track_embedder

    def forward(
        self,
        name: list[str],  # [B]
        x_artist: torch.Tensor,  # [B, T]
        x_cont: torch.Tensor,  # [B, T, n_cont]
        x_cat: torch.Tensor,  # [B, T, n_cat]
        non_pad_mask: torch.Tensor,  # [B, T]
        y: torch.Tensor | None = None,  # [B, T]
    ):
        e_name = self.name_embedder(name)  # [B, T, C]
        e_track = self.track_embedder(x_artist, x_cont, x_cat)  # [B, T, C]

    @classmethod
    def from_config(cls, config):
        tracks = pd.read_parquet("./.data/data/tracks.parquet")
        tensoriser = Tensoriser(tracks)
        
        name_embedder = PlaylistNameEmbedder.from_config(config)
        track_embedder = TrackEmbedder(
            config, tensoriser.artist_vocab_size, tensoriser.cat_vocab_sizes,
        )

        return cls(config, tensoriser, name_embedder, track_embedder)

    def as_dict(self) -> dict:
        states = {
            "name_embedder": self.name_embedder.state_dict(),
            "track_embedder": self.track_embedder.state_dict()
        }
        return {
            "config": asdict(self.config),
            "states": states,
            "tensoriser": self.tensoriser.as_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlaylistRecommender":
        config = ModelConfig(**d["config"])
        tensoriser = Tensoriser.from_dict(d["tensoriser"])

        name_embedder = PlaylistNameEmbedder.from_config(config)
        track_embedder = TrackEmbedder(
            config, tensoriser.artist_vocab_size, tensoriser.cat_vocab_sizes,
        )

        model = cls(config, tensoriser, name_embedder, track_embedder)

        for name, state in d["states"].items():
            getattr(model, name).load_state_dict(state)

        return model
