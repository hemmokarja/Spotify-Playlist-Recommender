import collections

import structlog
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = structlog.get_logger(__name__)


@torch.no_grad()
def _build_popularity_sampling_distribution(
    item_mapping, smoothing_factor=1.0, uniform_mix_factor=None
):
    item_counts = item_mapping.groupby("item_index").num_obs_token.sum()
    probs = torch.from_numpy(item_counts.to_numpy(dtype="float32"))  # [vocab_size]
    probs = (probs + 1e-10) ** smoothing_factor  # [vocab_size]
    probs /= probs.sum()  # [vocab_size]

    if uniform_mix_factor is not None:
        uniform_probs = torch.ones(len(probs)) / len(probs)  # [vocab_size]
        probs = (
            1.0 - uniform_mix_factor
        ) * probs + uniform_mix_factor * uniform_probs  # [vocab_size]

    return probs  # [vocab_size]


_SamplerOutput = collections.namedtuple(
    "SamplerOutput", ["sampled_indices", "true_probs", "sample_probs"]
)


class PopularitySampler(nn.Module):
    def __init__(
        self,
        item_mapping,
        n_samples,
        smoothing_factor=1.0,
        uniform_mix_factor=None,
        replacement=False,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.replacement = replacement

        sampling_probs = _build_popularity_sampling_distribution(
            item_mapping, smoothing_factor, uniform_mix_factor
        )  # [vocab_size]
        self.register_buffer("sampling_probs", sampling_probs)  # [vocab_size]

    def forward(self, y):
        # y: [B] positive item indices
        sampled_indices = torch.multinomial(
            self.sampling_probs, self.n_samples, replacement=self.replacement
        )  # [n_samples]

        true_probs = self.sampling_probs[y]  # [B]
        sample_probs = self.sampling_probs[sampled_indices]  # [n_samples]

        return _SamplerOutput(sampled_indices, true_probs, sample_probs)


class SampledSoftmaxLoss(nn.Module):
    def __init__(self, temperature=1.0):
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
        vocab_size,
        loss_kwargs=None,
        sampler_kwargs=None,
        item_embedding_fn=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.item_embedding_fn = item_embedding_fn
        self.sampler = PopularitySampler(**sampler_kwargs)
        self.loss_fn = SampledSoftmaxLoss(**loss_kwargs)

    def _compute_loss(self, emb, y):
        # emb: [B, T, C]
        # y: [B, T]
        emb = emb.view(-1, emb.size(-1))  # [B', C]
        y = y.view(-1)  # [B']

        mask = y != 0
        emb = emb[mask]
        y = y[mask]

        sampler_output = self.sampler(y)
        sampled_indices = sampler_output.sampled_indices  # [n_samples]

        pos_item_emb = self.item_embedding_fn(y)  # [B', C]
        neg_item_emb = self.item_embedding_fn(sampled_indices)  # [n_samples, C]

        pos_logits = (emb * pos_item_emb).sum(dim=1)  # [B']
        neg_logits = emb @ neg_item_emb.T  # [B', n_samples]

        # mask false negatives
        collision_mask = y.view(-1, 1) == sampled_indices.view(1, -1)  # [B', n_samples]
        neg_logits = neg_logits.masked_fill(collision_mask, -1e9)

        return self.loss_fn(
            pos_logits,
            neg_logits,
            sampler_output.true_probs,
            sampler_output.sample_probs,
        )

    def _get_full_probs(self, emb):
        # emb: [B, C]
        all_indices = torch.arange(self.vocab_size, device=emb.device)
        all_item_embs = self.item_embedding_fn(all_indices)  # [vocab_size, C]
        return F.log_softmax(emb @ all_item_embs.T, dim=-1)  # [B, vocab_size]

    def forward(self, emb, y=None, inference=False):
        # emb: [B, T, C]
        # y: [B, T]
        if inference:
            with torch.no_grad():
                last_step_probs = self._get_full_probs(emb[:, -1, :])  # [B, vocab_size]
        else:
            last_step_probs = None

        loss = self._compute_loss(emb, y) if y is not None else None
        return last_step_probs, loss
