import collections

import pandas as pd
import structlog
import torch
import torch.nn as nn
from torch.nn import functional as F

from recommender.layers import TrackEmbedder

logger = structlog.get_logger(__name__)


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


_SamplerOutput = collections.namedtuple(
    "SamplerOutput", ["sampled_indices", "true_probs", "sample_probs"]
)


class Sampler(nn.Module):
    def __init__(
        self,
        sampling_probs: torch.Tensor,
        n_samples: int,
        replacement: bool = False,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.replacement = replacement
        self.register_buffer("sampling_probs", sampling_probs)  # [vocab_size]

    def forward(self, y) -> _SamplerOutput:
        # y: [B] positive item indices
        sampled_indices = torch.multinomial(
            self.sampling_probs, self.n_samples, replacement=self.replacement
        )  # [n_samples]

        true_probs = self.sampling_probs[y]  # [B]
        sample_probs = self.sampling_probs[sampled_indices]  # [n_samples]

        return _SamplerOutput(sampled_indices, true_probs, sample_probs)


class SampledSoftmaxLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, pos_logits, neg_logits, true_probs, sample_probs):
        # pos_logits: [B] logits for positive items
        # neg_logits: [B, n_samples] logits for negative items
        # true_probs: [B] sampling probabilities for positive items
        # sample_probs: [n_samples] sampling probabilities for negatives
        pos_logits = pos_logits / self.temperature - torch.log(true_probs + 1e-10)  # [B]
        neg_logits = neg_logits / self.temperature - torch.log(
            sample_probs.unsqueeze(0) + 1e-10
        )  # [B, n_samples]

        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)  # [B, 1+n_samples]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # [B]
        return F.cross_entropy(logits, labels)  # scalar


class SampledSoftmaxPredictionHead(nn.Module):
    def __init__(
        self,
        tracks: pd.DataFrame,
        track_embedder: TrackEmbedder,
        vocab_size: int,
        n_neg_samples: int,
        smoothing_factor: float = 1.0,
        uniform_mix_factor: float | None = None,
        temperature: float = 1.0,
        train_mask: torch.Tensor = None,
    ):
        super().__init__()
        self.track_embedder = track_embedder
        self.vocab_size = vocab_size

        if train_mask is not None:
            self.register_buffer("train_mask", train_mask)
        else:
            self.train_mask = None

        sampling_probs = _make_popularity_sampling_distribution(
            tracks, smoothing_factor, uniform_mix_factor, train_mask
        )

        self.sampler = Sampler(sampling_probs, n_neg_samples, replacement=True)
        self.loss_fn = SampledSoftmaxLoss(temperature)

    def loss(self, hidden, y):
        # hidden: [B, T, C]
        # y: [B, T]
        hidden = hidden.view(-1, hidden.size(-1))  # [B', C]
        y = y.view(-1)  # [B']

        # exclude padding and non-train items from loss
        mask = y != 0
        if self.train_mask is not None:
            mask &= self.train_mask[y]

        hidden = hidden[mask]
        y = y[mask]

        sampler_output = self.sampler(y)
        sampled_indices = sampler_output.sampled_indices  # [n_samples]

        e_pos = self.track_embedder(y)  # [B', C]
        e_neg = self.track_embedder(sampled_indices)  # [n_samples, C]

        pos_logits = (hidden * e_pos).sum(dim=1)  # [B']
        neg_logits = hidden @ e_neg.T  # [B', n_samples]

        # mask false negatives
        collision_mask = y.view(-1, 1) == sampled_indices.view(1, -1)  # [B', n_samples]
        neg_logits = neg_logits.masked_fill(collision_mask, -1e9)

        return self.loss_fn(
            pos_logits,
            neg_logits,
            sampler_output.true_probs,
            sampler_output.sample_probs,
        )

    def full_probs(self, hidden, allowed_mask=None):
        # hidden: [B, C]
        # allowed_mask: [vocab_size] (optional)
        all_indices = torch.arange(self.vocab_size, device=hidden.device)
        e_track = self.track_embedder(all_indices)  # [vocab_size, C]
        logits = hidden @ e_track.T

        if allowed_mask is not None:
            logits = logits.masked_fill(~allowed_mask.unsqueeze(0), float("-inf"))

        return F.softmax(logits, dim=-1)  # [B, vocab_size]
