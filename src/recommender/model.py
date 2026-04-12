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
            tensoriser.vocab_size,
            config.loss_kwargs,
            config.sampler_kwargs,
            item_embedding_fn=self.track_embedder
        )

        logger.info(
            f"Initialized PlaylistRecommender with "
            f"{self.num_params() / 1e6 :.2f} M params "
            f"(of which {self.num_params(trainable_only=True) / 1e6 :.2f} M trainable)"
        )

    def forward(
        self,
        name: list[str],
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        inference: bool = False
    ):
        # name: [B]
        # x: [B, T-1]
        # y: [B, T] (optional)
        e_name = self.name_embedder(name)  # [B, C]
        e_track = self.track_embedder(x)  # [B, T-1, C]
        e = torch.concat([e_name.unsqueeze(1), e_track], dim=1)  # [B, T, C]
        e = self.block_stack(e)  # [B, T, C]

        last_step_probs, loss = self.head(e, y, inference)

        return last_step_probs, loss

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
