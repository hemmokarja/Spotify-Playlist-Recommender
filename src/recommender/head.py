import math
from collections import namedtuple

import structlog
import torch
import torch.nn as nn
from torch.nn import functional as F

from recommender.layers import TrackEmbedder

logger = structlog.get_logger(__name__)

_TEMP_CLAMP = 4.6  # ≈ln(100)


def _cosine_logits(
    hidden: torch.Tensor,
    embeddings: torch.Tensor,
    log_temperature: torch.Tensor,
) -> torch.Tensor:
    h = F.normalize(hidden, dim=-1)
    e = F.normalize(embeddings, dim=-1)
    # clamped at 4.6 ≈ ln(100) to prevent explosion
    return (h @ e.T) * log_temperature.clamp(max=_TEMP_CLAMP).exp()


def _sampled_softmax_loss(pos_logits, neg_logits, true_probs, sample_probs):
    # pos_logits: [B] logits for positive items (already temperature-scaled)
    # neg_logits: [B, n_samples] logits for negative items (already temperature-scaled)
    # true_probs: [B] sampling probabilities for positive items
    # sample_probs: [n_samples] sampling probabilities for negatives
    pos_logits = pos_logits - torch.log(true_probs + 1e-10)  # [B]
    neg_logits = neg_logits - torch.log(sample_probs.unsqueeze(0) + 1e-10)  # [B, n_samples]
    logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)  # [B, 1+n_samples]
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # [B]
    return F.cross_entropy(logits, labels)  # scalar


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


class SampledSoftmaxPredictionHead(nn.Module):
    def __init__(
        self,
        track_embedder: TrackEmbedder,
        sampling_probs: torch.Tensor,
        n_neg_samples: int,
        temperature_init: float = 0.07,
    ):
        super().__init__()
        # NOTE about using track_embedder in the head: we should never apply artist
        # dropout here. Candidate embeddings must be deterministic — stochastically
        # zeroing the artist component during loss computation would send contradictory
        # gradient signals
        self.track_embedder = track_embedder
        self.vocab_size = len(sampling_probs)

        self.sampler = Sampler(sampling_probs, n_neg_samples, replacement=True)

        # Learnable log-temperature, initialised à la CLIP (1/0.07 ≈ 14.3 → log ≈ 2.66).
        # Clamped at ln(100) ≈ 4.6 at use-sites to prevent logit explosion.
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(1.0 / temperature_init))
        )

    def loss(
        self,
        hidden: torch.Tensor,
        y: torch.Tensor,
        loss_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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

        e_pos = self.track_embedder(y, apply_artist_dropout=False)  # [B', C]
        e_neg = self.track_embedder(sampled_indices, apply_artist_dropout=False)  # [n_samples, C]

        hidden_n = F.normalize(hidden, dim=-1)  # [B', C]
        e_pos_n = F.normalize(e_pos, dim=-1)
        temp = self.log_temperature.clamp(max=_TEMP_CLAMP).exp()
        pos_logits = (hidden_n * e_pos_n).sum(dim=1) * temp  # [B']

        neg_logits = _cosine_logits(hidden, e_neg, self.log_temperature)   # [B', n_samples]

        # mask false negatives
        collision_mask = y.view(-1, 1) == sampled_indices.view(1, -1)  # [B', n_samples]
        neg_logits = neg_logits.masked_fill(collision_mask, -1e9)

        return _sampled_softmax_loss(
            pos_logits,
            neg_logits,
            sampler_output.true_probs,
            sampler_output.sample_probs,
        )

    def full_probs(
        self, hidden: torch.Tensor, allowed_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # hidden: [B, C]
        # allowed_mask: [vocab_size] (optional)
        all_indices = torch.arange(self.vocab_size, device=hidden.device)
        e_track = self.track_embedder(all_indices, apply_artist_dropout=False)  # [vocab_size, C]
        logits = _cosine_logits(hidden, e_track, self.log_temperature)  # [B, vocab_size]

        if allowed_mask is not None:
            logits = logits.masked_fill(~allowed_mask.unsqueeze(0), float("-inf"))

        return F.softmax(logits, dim=-1)  # [B, vocab_size]

    def top_k_indices(
        self,
        hidden: torch.Tensor,
        k: int,
        allowed_mask: torch.Tensor | None = None,
        chunk_size: int | None = 100_000,
        precomputed_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # hidden: [B, C]
        # precomputed_embeddings: [vocab_size, C] (optional, for fast inference)
        # vocab_chunk_size: if None, materialize full [B, vocab_size] logits in one shot.
        #   Only pass None when B is small (inference). For large-batch eval, use chunking.

        if chunk_size is None:
            e_all = (
                precomputed_embeddings
                if precomputed_embeddings is not None
                else self.track_embedder(
                    torch.arange(self.vocab_size, device=hidden.device),
                    apply_artist_dropout=False,
                )
            )  # [vocab_size, C]
            logits = _cosine_logits(hidden, e_all, self.log_temperature)  # [B, vocab_size]
            if allowed_mask is not None:
                logits = logits.masked_fill(~allowed_mask.unsqueeze(0), float("-inf"))
            return torch.topk(logits, k, dim=1).indices  # [B, k]

        # chunked path: peak memory O(B * vocab_chunk_size) instead of O(B * vocab_size)
        best_vals = torch.full(
            (hidden.size(0), k), float("-inf"), device=hidden.device
        )  # [B, k]
        best_idxs = torch.zeros(
            (hidden.size(0), k), dtype=torch.long, device=hidden.device
        )  # [B, k]

        for start in range(0, self.vocab_size, chunk_size):
            end = min(start + chunk_size, self.vocab_size)
            indices = torch.arange(start, end, device=hidden.device)
            e_chunk = (
                precomputed_embeddings[start:end]
                if precomputed_embeddings is not None
                else self.track_embedder(indices, apply_artist_dropout=False)
            )  # [chunk, C]
            chunk_logits = _cosine_logits(hidden, e_chunk, self.log_temperature)  # [B, chunk]

            if allowed_mask is not None:
                chunk_logits = chunk_logits.masked_fill(
                    ~allowed_mask[start:end].unsqueeze(0), float("-inf")
                )

            combined_vals = torch.cat([best_vals, chunk_logits], dim=1)  # [B, k+chunk]
            combined_idxs = torch.cat(
                [best_idxs, indices.unsqueeze(0).expand(hidden.size(0), -1)], dim=1
            )  # [B, k+chunk]
            best_vals, local_idxs = torch.topk(combined_vals, k, dim=1)
            best_idxs = combined_idxs.gather(1, local_idxs)

        return best_idxs  # [B, k]
