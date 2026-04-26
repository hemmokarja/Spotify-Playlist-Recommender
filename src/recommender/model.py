import math
from dataclasses import asdict, dataclass

import pandas as pd
import structlog
import torch
import torch.nn as nn

from recommender.data import Tensoriser
from recommender.head import SampledSoftmaxPredictionHead
from recommender.layers import (
    FrozenTrackEmbedder, PlaylistNameEmbedder, TrackEmbedder, TransformerBlockStack
)
from recommender.model_config import ModelConfig

logger = structlog.get_logger(__name__)

_FULL_PROBS_CHUNK_SIZE = 100_000


@torch.no_grad()
def _make_popularity_sampling_distribution(
    tracks: pd.DataFrame,
    smoothing_factor: float = 1.0,
    uniform_mix_factor: float | None = None,
    mask: torch.Tensor | None = None
):
    probs = torch.from_numpy(tracks.n_obs.to_numpy())  # [vocab_size]
    probs = (probs + 1e-10) ** smoothing_factor  # [vocab_size]
    probs /= probs.sum()  # [vocab_size]

    if uniform_mix_factor is not None:
        uniform_probs = torch.ones(len(probs)) / len(probs)  # [vocab_size]
        probs = (
            1.0 - uniform_mix_factor
        ) * probs + uniform_mix_factor * uniform_probs  # [vocab_size]

    if mask is not None:
        probs *= mask

    return probs  # [vocab_size]


class PlaylistRecommender(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        tensoriser: Tensoriser,
        name_embedder: PlaylistNameEmbedder,
        track_embedder: TrackEmbedder | FrozenTrackEmbedder,
        block_stack: TransformerBlockStack,
    ):
        super().__init__()
        self.config = config
        self.tensoriser = tensoriser

        self.name_embedder = name_embedder
        self.track_embedder = track_embedder
        self.block_stack = block_stack

        self.register_buffer("train_mask", tensoriser.get_train_mask(include_pad=True))

        sampling_probs = _make_popularity_sampling_distribution(
            self.tensoriser.tracks,
            config.smoothing_factor,
            config.uniform_mix_factor,
            self.train_mask
        )

        self.head = SampledSoftmaxPredictionHead(
            track_embedder=self.track_embedder,
            sampling_probs=sampling_probs,
            n_neg_samples=config.n_neg_samples,
            temperature=config.loss_temperature,
        )

        self.apply(self._init_weights)
        self._init_skip_proj_weights()

        logger.info(
            f"Initialized PlaylistRecommender with "
            f"{self.num_params() / 1e6 :.2f} M params "
            f"(of which {self.num_params(trainable_only=True) / 1e6 :.2f} M trainable)"
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if not module.weight.requires_grad:
                # avoid overwriting pre-trained text embedding model
                return
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if not module.weight.requires_grad:
                # avoid overwriting pre-trained text embedding model
                return
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def _init_skip_proj_weights(self):
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") and p.requires_grad:
                # avoid overwriting pre-trained text embedding model
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
                )

    def forward(
        self,
        name: list[str],
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        seq_len: torch.Tensor = None,
        loss_mask: torch.Tensor | None = None,  
    ):
        # name: [B]
        # x: [B, T-1]
        # y: [B, T] (optional)
        # seq_len: [B] (not used, allow passing batch directly to forward)
        # loss_mask: [vocab_size], don't consider these items in loss
        e = self.propagate_hidden(name, x)
        loss = self.head.loss(e, y, loss_mask)
        return loss

    def propagate_hidden(self, name: list[str], x: torch.Tensor):
        # name: [B]
        # x: [B, T-1]
        e_name = self.name_embedder(name)  # [B, C]
        e_track = self.track_embedder(x)  # [B, T-1, C]
        e = torch.concat([e_name.unsqueeze(1), e_track], dim=1)  # [B, T, C]
        e = self.block_stack(e)  # [B, T, C]
        return e

    def top_k_indices(
        self,
        name: list[str],
        x: torch.Tensor,
        seq_len: torch.Tensor,
        top_k: int = 10,
        allowed_mask: torch.Tensor | None = None,
        chunk_size: int | None = 100_000,
        precomputed_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        e = self.propagate_hidden(name, x)  # [B, T, C]
        batch_idx = torch.arange(e.shape[0], device=e.device)
        e_last = e[batch_idx, seq_len]  # [B, C]
        return self.head.top_k_indices(
            hidden=e_last,
            k=top_k,
            allowed_mask=allowed_mask,
            chunk_size=chunk_size,
            precomputed_embeddings=precomputed_embeddings
        )  # [B, top_k]

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

        model.name_embedder.model.to(model.get_device())

        return model

    def num_params(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def get_device(self):
        return self.track_embedder.get_device()

    def to_inference_model(
        self, freeze_track_embedder: bool = True
    ) -> "PlaylistRecommenderInference":
        return PlaylistRecommenderInference(self, freeze_track_embedder)


def _handle_batching(x, device):
    return x.unsqueeze(0).to(device) if isinstance(x, torch.Tensor) else [x]


@dataclass
class Recommendation:
    position: int
    track_id: int
    track: str
    artist: str


class PlaylistRecommenderInference:
    def __init__(self, model: PlaylistRecommender, freeze_track_embedder: bool = True):
        self.model = model
        self.tensoriser = model.tensoriser
        self.freeze_track_embedder = freeze_track_embedder
        self.precomputed_embeddings = None

        if freeze_track_embedder:
            was_training = self.model.training
            self.model.eval()
            frozen = FrozenTrackEmbedder.from_track_embedder(model.track_embedder)
            self.model.track_embedder = frozen
            self.model.head.track_embedder = frozen
            self.precomputed_embeddings = frozen.embeddings  # [vocab_size, C]
            self.model.train(was_training)

    @torch.inference_mode()
    def get_recommendations(
        self,
        playlist_name: str,
        playlist: list[int],
        top_k: int = 10,
        allowed_mask: torch.Tensor | None = None,
        chunk_size: int | None = None
    ) -> list[Recommendation]:
        # allowed_mask: [vocab_size]
        was_training = self.model.training
        self.model.eval()

        device = self.model.get_device()
        sample = self.tensoriser.tensorise(playlist_name, playlist, inference=True)
        batch = {
            k: _handle_batching(v, device) for k, v in sample.items()
        }

        indices = self.model.top_k_indices(
            **batch,
            top_k=top_k,
            allowed_mask=allowed_mask,
            chunk_size=chunk_size,
            precomputed_embeddings=self.precomputed_embeddings
        )
        indices = indices.squeeze(0).tolist()  # [top_k]

        self.model.train(was_training)
        return [
            Recommendation(
                position=pos,
                track_id=ix,
                track=self.tensoriser.track_id_to_name[ix],
                artist=self.tensoriser.track_id_to_artist[ix],
            )
            for pos, ix in enumerate(indices, start=1)
        ]
