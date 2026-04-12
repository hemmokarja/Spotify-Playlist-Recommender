from dataclasses import asdict

import pandas as pd
import structlog
import torch
import torch.nn as nn

from recommender.data import Tensoriser
from recommender.head import SampledSoftmaxPredictionHead
from recommender.layers import (
    PlaylistNameEmbedder, TrackEmbedder, TransformerBlockStack
)
from recommender.model_config import ModelConfig

logger = structlog.get_logger(__name__)


class PlaylistRecommender(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        tensoriser: Tensoriser,
        name_embedder: PlaylistNameEmbedder,
        track_embedder: TrackEmbedder,
        block_stack: TransformerBlockStack,
    ):
        super().__init__()
        self.config = config
        self.tensoriser = tensoriser

        self.name_embedder = name_embedder
        self.track_embedder = track_embedder
        self.block_stack = block_stack

        self.head = SampledSoftmaxPredictionHead(
            self.track_embedder,
            tensoriser.vocab_size,
            config.loss_kwargs,
            config.sampler_kwargs,
        )

        logger.info(
            f"Initialized PlaylistRecommender with "
            f"{self.num_params() / 1e6 :.2f} M params "
            f"(of which {self.num_params(trainable_only=True) / 1e6 :.2f} M trainable)"
        )

    def propagate_hidden(self, name: list[str], x: torch.Tensor):
        # name: [B]
        # x: [B, T-1]
        e_name = self.name_embedder(name)  # [B, C]
        e_track = self.track_embedder(x)  # [B, T-1, C]
        e = torch.concat([e_name.unsqueeze(1), e_track], dim=1)  # [B, T, C]
        e = self.block_stack(e)  # [B, T, C]
        return e

    def forward(self, name: list[str], x: torch.Tensor, y: torch.Tensor | None = None):
        # name: [B]
        # x: [B, T-1]
        # y: [B, T] (optional)
        e = self.propagate_hidden(name, x)
        loss = self.head.loss(e, y)
        return loss

    @classmethod
    def from_config(cls, config) -> "PlaylistRecommender":
        tracks = pd.read_parquet("data/data/tracks.parquet")
        tensoriser = Tensoriser(tracks)

        name_embedder = PlaylistNameEmbedder.from_config(config)
        track_embedder = TrackEmbedder.from_config_and_tensoriser(config, tensoriser)
        block_stack = TransformerBlockStack(config)

        return cls(config, tensoriser, name_embedder, track_embedder, block_stack)

    def as_dict(self) -> dict:
        states = {
            "name_embedder": self.name_embedder.state_dict(),
            "track_embedder": self.track_embedder.state_dict(),
            "block_stack": self.block_stack.state_dict(),
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
        track_embedder = TrackEmbedder.from_config_and_tensoriser(config, tensoriser)
        block_stack = TransformerBlockStack(config)

        model = cls(config, tensoriser, name_embedder, track_embedder, block_stack)

        for name, state in d["states"].items():
            getattr(model, name).load_state_dict(state)

        return model

    def num_params(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def get_device(self):
        self.track_embedder.artist_emb.weight.device


def _handle_batching(x, device):
    return x.unsqueeze(0).to(device) if isinstance(x, torch.Tensor) else [x]


class PlaylistRecommenderInference:
    def __init__(self, model: PlaylistRecommender):
        self.model = model
        self.tensoriser = model.tensoriser

    def last_step_probs(
        self, name: list[str], x: torch.Tensor, allowed_mask: torch.Tensor | None = None
    ):
        e = self.model.propagate_hidden(name, x)
        probs = self.model.head.full_probs(e[:, -1, :], allowed_mask)
        return probs

    def get_recommendations(
        self, playlist_name: str,
        playlist: list[int],
        allowed_mask: torch.Tensor | None = None,
    ):
        was_training = self.model.training
        self.model.eval()
        device = self.model.get_device()
        sample = self.tensoriser.tensorise(playlist_name, playlist)
        batch = {
            k: _handle_batching(v, device) for k, v in sample.items()
        }
        probs = self.last_step_probs(**batch)
        # TODO finnish

        self.model.training(was_training)
        return None
