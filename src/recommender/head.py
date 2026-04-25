from collections import namedtuple

import structlog
import torch
import torch.nn as nn
from torch.nn import functional as F

from recommender.layers import TrackEmbedder

logger = structlog.get_logger(__name__)


_SamplerOutput = namedtuple(
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
        track_embedder: TrackEmbedder,
        sampling_probs: torch.Tensor,
        n_neg_samples: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.track_embedder = track_embedder
        self.vocab_size = len(sampling_probs)

        self.sampler = Sampler(sampling_probs, n_neg_samples, replacement=True)
        self.loss_fn = SampledSoftmaxLoss(temperature)

    def loss(self, hidden, y, loss_mask: torch.Tensor | None = None):
        # hidden: [B, T, C]
        # y: [B, T]
        # loss_mask: [vocab_size]
        hidden = hidden.view(-1, hidden.size(-1))  # [B', C]
        y = y.view(-1)  # [B']

        # exclude padding and non-train items from loss
        mask = y != 0
        if loss_mask is not None:
            mask &= loss_mask[y]

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

    def full_probs(self, hidden, allowed_mask: torch.Tensor | None = None):
        # hidden: [B, C]
        # allowed_mask: [vocab_size] (optional)
        all_indices = torch.arange(self.vocab_size, device=hidden.device)
        e_track = self.track_embedder(all_indices)  # [vocab_size, C]
        logits = hidden @ e_track.T

        if allowed_mask is not None:
            logits = logits.masked_fill(~allowed_mask.unsqueeze(0), float("-inf"))

        return F.softmax(logits, dim=-1)  # [B, vocab_size]

    def top_k_indices(
        self,
        hidden,
        k: int,
        allowed_mask: torch.Tensor | None = None,
        chunk_size: int = 100_000,
    ) -> torch.Tensor:
        # hidden: [B, C]
        # Returns [B, k] top-k vocab indices without ever materializing [B, vocab_size].
        # Peak memory: O(B * chunk_size) instead of O(B * vocab_size).
        best_vals = torch.full(
            (hidden.size(0), k), float("-inf"), device=hidden.device
        )  # [B, k]
        best_idxs = torch.zeros(
            (hidden.size(0), k), dtype=torch.long, device=hidden.device
        )  # [B, k]

        for start in range(0, self.vocab_size, chunk_size):
            end = min(start + chunk_size, self.vocab_size)
            indices = torch.arange(start, end, device=hidden.device)
            e_chunk = self.track_embedder(indices)  # [chunk, C]
            chunk_logits = hidden @ e_chunk.T  # [B, chunk]

            if allowed_mask is not None:
                chunk_logits = chunk_logits.masked_fill(
                    ~allowed_mask[start:end].unsqueeze(0), float("-inf")
                )

            # merge chunk top-k with running top-k
            combined_vals = torch.cat([best_vals, chunk_logits], dim=1)  # [B, k+chunk]
            combined_idxs = torch.cat(
                [best_idxs, indices.unsqueeze(0).expand(hidden.size(0), -1)], dim=1
            )  # [B, k+chunk]
            best_vals, local_idxs = torch.topk(combined_vals, k, dim=1)
            best_idxs = combined_idxs.gather(1, local_idxs)

        return best_idxs  # [B, k]
