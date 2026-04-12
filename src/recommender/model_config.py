from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int
    d_artist: int | None = None
    d_cat: int | None = None
    dropout: float = 0.0
