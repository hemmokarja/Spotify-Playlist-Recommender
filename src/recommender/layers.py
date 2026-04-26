import os

import huggingface_hub
import requests
import structlog
import torch
import torch.nn as nn
from dotenv import load_dotenv
from huggingface_hub.utils import GatedRepoError
from sentence_transformers import SentenceTransformer

from recommender.data import Tensoriser
from recommender.model_config import ModelConfig

load_dotenv()

logger = structlog.get_logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PlaylistNameEmbedder(nn.Module):
    MODEL_ID = "google/embeddinggemma-300m"

    def __init__(self, config: ModelConfig, model: SentenceTransformer):
        super().__init__()
        self.config = config

        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.proj = nn.Linear(config.d_name, config.d_model)

    @torch._dynamo.disable
    def forward(self, names: list[str]) -> torch.Tensor:
        with torch.no_grad():
            e_name = self.model.encode(names, convert_to_tensor=True)  # [B, d_name]
        e_name = e_name.clone().detach()
        return self.proj(e_name)  # [B, d_model]

    @classmethod
    def from_config(cls, config: ModelConfig) -> "PlaylistNameEmbedder":
        try:
            emb_model = SentenceTransformer(cls.MODEL_ID)
        except (OSError, GatedRepoError, requests.exceptions.HTTPError):
            logger.info("Loading a model that requires huggingface login...")
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                raise RuntimeError("Missing HF_TOKEN environment variable!")
            huggingface_hub.login(hf_token)
            emb_model = SentenceTransformer(cls.MODEL_ID)
        return cls(config, emb_model)

    def state_dict(self, **kwargs) -> dict:
        return self.proj.state_dict(**kwargs)

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        self.proj.load_state_dict(state_dict, strict=strict)


def _artist_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Zeroes out entire embeddings probabilistically (one mask per position)."""
    # x: [B, T, C]
    if not training or p == 0.0:
        return x

    B, T, _ = x.size()
    keep_mask = (torch.rand((B, T, 1), device=x.device) > p).float()
    return (x * keep_mask) / (1.0 - p)


class TrackEmbedder(nn.Module):
    N_CONT = 9

    def __init__(
        self,
        config: ModelConfig,
        cont_feat_mapping: torch.Tensor,
        cat_feat_mapping: torch.Tensor,
        artist_mapping: torch.Tensor,
        cat_vocab_sizes: list[int],
    ):
        super().__init__()
        self.config = config

        d_model = config.d_model
        d_artist = config.d_artist or d_model // 4
        d_cont = config.d_cont or d_model // 2

        self.register_buffer("cont_feat_mapping", cont_feat_mapping, persistent=False)  # [vocab_size, n_cont]
        self.register_buffer("cat_feat_mapping", cat_feat_mapping, persistent=False)  # [vocab_size, n_cat]
        self.register_buffer("artist_mapping", artist_mapping, persistent=False)  # [vocab_size]

        artist_vocab_size = int(artist_mapping.max().item()) + 1

        self.artist_emb = nn.Embedding(artist_vocab_size, d_artist, padding_idx=0)
        self.cont_mlp = nn.Sequential(
            nn.Linear(self.N_CONT, d_cont),
            nn.ReLU(),
            nn.Linear(d_cont, d_cont),
        )

        # give each categorical feature a proportionally-sized embedding,
        # with a minimum of 4 dims so tiny vocab features still have some capacity
        cat_budget = d_model - d_artist - d_cont
        d_per_cat = max(4, cat_budget // len(cat_vocab_sizes))
        self.cat_embs = nn.ModuleList(
            [
                nn.Embedding(vocab_size, d_per_cat, padding_idx=0)
                for vocab_size in cat_vocab_sizes
            ]
        )
        d_cat = d_per_cat * len(cat_vocab_sizes)

        d_concat = d_artist + d_cont + d_cat
        self.proj = nn.Linear(d_concat, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(config.dropout)

    @classmethod
    def from_config_and_tensoriser(
        cls, config: ModelConfig, tensoriser: Tensoriser
    ) -> "TrackEmbedder":
        tracks = tensoriser.tracks
        cont_feat_mapping = torch.tensor(
            tracks[tensoriser.CONT_FEATURES].to_numpy(), dtype=torch.float32
        )
        cat_feat_mapping = torch.tensor(
            tracks[tensoriser.CAT_FEATURES].to_numpy(), dtype=torch.long
        )
        artist_mapping = torch.tensor(tracks.artist_index.to_numpy(), dtype=torch.long)
        return cls(
            config,
            cont_feat_mapping,
            cat_feat_mapping,
            artist_mapping,
            tensoriser.cat_vocab_sizes,
        )

    def forward(self, x, apply_artist_dropout: bool = True):
        # x: [B] or [B, T] — T may be 0 for empty playlists
        is_1d = x.dim() == 1
        if is_1d:
            x = x.unsqueeze(1)  # [B, 1]

        if x.shape[1] == 0:
            B = x.shape[0]
            return torch.zeros(
                B, 0, self.config.d_model, device=x.device, dtype=self.proj.weight.dtype
            )

        x_cont = self.cont_feat_mapping[x]  # [B, T, n_cont]
        x_cat = self.cat_feat_mapping[x]  # [B, T, n_cat]
        x_artist = self.artist_mapping[x]  # [B, T]

        e_artist = self.artist_emb(x_artist)  # [B, T, d_artist]
        if apply_artist_dropout:
            e_artist = _artist_dropout(
                e_artist, self.config.artist_dropout, self.training
            )  # [B, T, d_artist]

        e_cont = self.cont_mlp(x_cont)  # [B, T, d_cont]

        e_cats = [emb(x_cat[..., i]) for i, emb in enumerate(self.cat_embs)]  # [B, T, d_per_cat] each
        e_cat = torch.cat(e_cats, dim=-1)  # [B, T, d_cat]

        e = torch.cat([e_artist, e_cont, e_cat], dim=-1)  # [B, T, d_concat]
        e = self.ln(self.proj(self.dropout(e)))  # [B, T, d_model]
        
        if is_1d:
            e = e.squeeze(1)  # [B, d_model]

        return e

    def get_device(self):
        return self.artist_emb.weight.device


class FrozenTrackEmbedder(nn.Module):
    """
    A materialized, frozen version of TrackEmbedder.
    Runs the full TrackEmbedder forward pass once over the entire vocabulary and caches
    the result as a non-trainable buffer. Lookups are then simple index operations - no
    MLP, no proj, no dropout.

    Use for inference only. Not suitable for training (gradients won't flow back to the
    original TrackEmbedder parameters).
    """
    def __init__(self, embeddings: torch.Tensor):
        super().__init__()
        self.register_buffer("embeddings", embeddings)  # [vocab_size, d_model]

    @classmethod
    def from_track_embedder(
        cls, track_embedder: TrackEmbedder
    ) -> "FrozenTrackEmbedder":
        logger.info("Freezing TrackEmbedder...")
        vocab_size = track_embedder.cont_feat_mapping.shape[0]
        device = track_embedder.artist_emb.weight.device
        all_indices = torch.arange(vocab_size, device=device)

        with torch.no_grad():
            embeddings = track_embedder(all_indices, apply_artist_dropout=False)  # [vocab_size, C]

        return cls(embeddings)

    def forward(self, x, apply_artist_dropout: bool = False) -> torch.Tensor:
        # x: [B] or [B, T]
        return self.embeddings[x]

    def get_device(self):
        return self.embeddings.device


class CausalSelfAttentionWithROPE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0
        assert (config.d_model // config.n_head) % 2 == 0, "RoPE requires even head size"
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.head_size = config.d_model // config.n_head
        self.d_model = config.d_model
        self.dropout = config.dropout

        self.rope_base = config.rope_base
        self.rope_dim = self.head_size // 2
        dim_indices = torch.arange(0, self.rope_dim).float()
        theta = 1.0 / (self.rope_base ** (2 * dim_indices / self.head_size))
        self.register_buffer("theta", theta)

    @staticmethod
    def _apply_rope(x, cos, sin):
        # x: [B, nh, T, hs]
        # cos: [1, 1, T, d]
        # sin: [1, 1, T, d]
        x_ = x.reshape(*x.shape[:-1], -1, 2)  # [B, nh, T, d, 2] (split head size into two)
        x0 = x_[..., 0]
        x1 = x_[..., 1]
        x0_rot = x0 * cos - x1 * sin
        x1_rot = x0 * sin + x1 * cos
        x_rot = torch.stack([x0_rot, x1_rot], dim=-1).flatten(start_dim=-2)
        return x_rot

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # [B, nh, T, hs]
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # [B, nh, T, hs]
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # [B, nh, T, hs]

        theta = self.theta.to(x.dtype)
        t = torch.arange(T, device=x.device, dtype=x.dtype)
        freqs = torch.outer(t, theta)  # [T, d]
        freqs_cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # [1, 1, T, d]
        freqs_sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)  # [1, 1, T, d]

        q = self._apply_rope(q, freqs_cos, freqs_sin)  # [B, nh, T, hs]
        k = self._apply_rope(k, freqs_cos, freqs_sin)  # [B, nh, T, hs]

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )  # [B, nh, T, hs]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: [B, T, C]
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttentionWithROPE(config)
        self.ln_2 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # x: [B, T, C]
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerBlockStack(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln = nn.LayerNorm(config.d_model)
    
    def forward(self, x):
        # x: [B, T, C]
        for block in self.blocks:
            x = block(x)
        return self.ln(x)
