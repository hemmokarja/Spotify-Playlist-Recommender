import torch
import structlog

from recommender.data import PlaylistDataset
from recommender.model import PlaylistRecommender
from recommender.model_config import ModelConfig
from recommender.trainer import Trainer, TrainerConfig

logger = structlog.get_logger(__name__)

CHECKPOINT_PATH = None  # if None, start from scratch
N_SAMPLES_TRAIN = 280_000

MODEL_CONFIG = ModelConfig(
    n_layer=3,
    d_name=768,
    d_model=128,
    d_artist=None,
    d_cont=None,
    n_head=8,
    dropout=0.0,
    artist_dropout=0.01,
    bias=True,
    rope_base=10_000.0,
    n_neg_samples=1000,
    smoothing_factor=0.75,
    uniform_mix_factor=0.1,
    loss_temperature=1.0
)
TRAINER_CONFIG = TrainerConfig(
    batch_size=128,
    gradient_acc_steps=16,
    log_interval=8,
    compile=False,
    base_learning_rate=1e-4,
    min_learning_rate=1e-6,
    lr_step_size=20_000,
    lr_gamma=0.75,
    weight_decay=0.01,
    betas=(0.9, 0.95),
    grad_clip=1.0,
    num_workers=1,
    prefetch_factor=2,
    pin_memory=False,
    validation_samples=5_000,
    validation_interval=10_000,
    checkpoint_filepath="checkpoints/model.pt"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_trainer_from_scratch():
    config = ModelConfig(n_layer=3, d_model=128, d_name=768)
    model = PlaylistRecommender.from_config(config)
    train_dataset = PlaylistDataset("train", model.tensoriser)
    validation_dataset = PlaylistDataset("test", model.tensoriser)
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

    tensoriser = PlaylistRecommender.tensoriser_from_checkpoint(CHECKPOINT_PATH)
    train_dataset = PlaylistDataset("train", tensoriser)
    validation_dataset = PlaylistDataset("test", tensoriser)

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