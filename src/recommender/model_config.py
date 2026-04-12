from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_name: int  # frozen embedder
    d_model: int
    d_artist: int | None = None
    d_cont: int | None = None
    dropout: float = 0.0
