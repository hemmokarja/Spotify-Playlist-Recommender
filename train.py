import torch
import structlog

from recommender.data import DataConfig, PlaylistDataset, ColdStartTransform
from recommender.model import PlaylistRecommender
from recommender.model_config import ModelConfig
from recommender.trainer import Trainer, TrainerConfig

logger = structlog.get_logger(__name__)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

CHECKPOINT_PATH = None  # if None, start from scratch
N_SAMPLES_TRAIN = 2_000_000

MODEL_CONFIG = ModelConfig(
    n_layer=5,
    d_name=768,
    d_model=256,
    d_artist=None,
    d_cont=None,
    n_head=8,
    dropout=0.0,
    artist_dropout=0.3,
    bias=True,
    rope_base=10_000.0,
    n_neg_samples=15_000,
    smoothing_factor=0.75,
    uniform_mix_factor=0.1,
    loss_temperature=1.0
)
TRAINER_CONFIG = TrainerConfig(
    batch_size=2048,
    gradient_acc_steps=8,
    log_interval=2048,
    compile=True,
    base_learning_rate=3e-4,
    min_learning_rate=1e-6,
    lr_step_size=1_000_000,
    lr_gamma=0.75,
    weight_decay=1e-5,
    betas=(0.9, 0.95),
    grad_clip=1.0,
    num_workers=2,
    prefetch_factor=4,
    pin_memory=True,
    validation_samples=50_000,
    validation_interval=200_000,
    checkpoint_filepath="checkpoints/model.pt"
)
DATA_CONFIG = DataConfig(p_sample_cold_start=0.01)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_trainer_from_scratch():
    model = PlaylistRecommender.from_config(MODEL_CONFIG)

    transforms = [ColdStartTransform(DATA_CONFIG.p_sample_cold_start)]
    train_dataset = PlaylistDataset("train", transforms)
    validation_dataset = PlaylistDataset("test")
    
    trainer =  Trainer(
        TRAINER_CONFIG,
        model,
        train_dataset,
        validation_dataset,
        DEVICE,
    )
    return trainer


def initalize_trainer_from_checkpoint():
    logger.info("Continuing SFT run from a checkpoint")

    transforms = [ColdStartTransform(DATA_CONFIG.p_sample_cold_start)]
    train_dataset = PlaylistDataset("train", transforms)
    validation_dataset = PlaylistDataset("test")

    trainer = Trainer.from_checkpoint(
        CHECKPOINT_PATH,
        train_dataset,
        validation_dataset,
        DEVICE,
    )
    return trainer


def main():
    if CHECKPOINT_PATH is None:
        trainer = initialize_trainer_from_scratch()
    else:
        trainer = initalize_trainer_from_checkpoint()
    trainer.train(N_SAMPLES_TRAIN)


if __name__ == "__main__":
    main()