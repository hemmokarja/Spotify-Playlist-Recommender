import torch
import torch.nn as nn

from recommender.model_config import ModelConfig


class TrackEmbedder(nn.Module):
    """
    Args:
        artist_vocab_size: vocabulary size for artist IDs
        cat_vocab_sizes:   list of vocab sizes for each categorical feature
                           (must match order of Tensoriser.CAT_FEATURES)
        total_dim:         total output embedding dimension D (default 128)
        artist_dim:        artist embedding dim (default D // 4)
        cont_dim:          continuous projection output dim (default D // 2)
        dropout:           dropout probability applied before final projection
    """
    N_CONT = 9

    def __init__(
        self, artist_vocab_size: int, cat_vocab_sizes: list[int], config: ModelConfig,
    ):
        super().__init__()

        self.config = config

        d_model = config.d_model
        d_artist = config.d_artist or d_model // 4
        d_cont = config.d_cont or d_model // 2

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

    def forward(
        self,
        x_artist: torch.Tensor,  # [B, T]
        x_cont: torch.Tensor,  # [B, T, n_cont]
        x_cat: torch.Tensor,  # [B, T, n_cat]
    ) -> torch.Tensor:  # [B, T, d_model]

        e_artist = self.artist_emb(x_artist)  # [B, T, d_artist]
        e_cont = self.cont_mlp(x_cont)  # [B, T, d_cont]

        e_cats = [emb(x_cat[..., i]) for i, emb in enumerate(self.cat_embs)]  # [B, T, d_per_cat] each
        e_cat = torch.cat(e_cats, dim=-1)   # [B, T, d_cat]

        e = torch.cat([e_artist, e_cont, e_cat], dim=-1)  # [B, T, d_concat]
        return self.ln(self.proj(self.dropout(e)))  # [B, T, d_model]
