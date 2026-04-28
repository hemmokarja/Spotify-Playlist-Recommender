from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_layer: int
    d_name: int  # frozen embedder
    d_model: int
    d_artist: int | None = None
    d_cont: int | None = None
    n_head: int = 8
    dropout: float = 0.0
    artist_dropout: float = 0.0
    bias: bool = True
    rope_base: float = 10_000.0
    n_neg_samples: int = 1000
    smoothing_factor: float = 1.0
    uniform_mix_factor: float | None = None
