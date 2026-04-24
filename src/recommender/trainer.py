import collections
import contextlib
import datetime
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from recommender.data import PlaylistDataset
from recommender.model import PlaylistRecommender

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    batch_size: int  # split into micro steps
    gradient_acc_steps: int = 10
    log_interval: int = 100
    compile: bool = True
    base_learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    lr_step_size: int = 50_000_000
    lr_gamma: float = 0.33
    weight_decay: float = 1e-5
    betas: Tuple[float] = (0.9, 0.95)
    grad_clip: float = 1.2
    num_workers: Optional[int] = 0
    prefetch_factor: int = None
    pin_memory: bool = False
    validation_samples: int = 1000
    validation_interval: int = 1000
    checkpoint_filepath: Optional[str] = None  # don't save if None

    def __post_init__(self):
        if self.batch_size % self.gradient_acc_steps != 0:
            raise ValueError("batch_size must be divisible by gradient_acc_steps")


def _get_learning_rate_stepwise(
    samples_seen: int,
    base_lr: float = 3e-4,
    min_lr: float = 1e-6,
    step_size: int = 50_000_000,
    gamma: float = 0.33,
) -> float:
    # reduce lr by a factor of gamma every step_size samples
    # set base_lr==min_lr to effectively turn scheduling off
    if not base_lr >= min_lr:
        raise ValueError(f"Set base_lr {base_lr} equal or greater than min_lr {min_lr}")
    num_steps = samples_seen // step_size
    lr = base_lr * (gamma**num_steps)
    return max(lr, min_lr)


def _configure_optimizer(
    model: PlaylistRecommender,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
) -> tuple[torch.optim.Optimizer, list[torch.nn.Parameter]]:
    params = [p for p in model.parameters() if p.requires_grad]

    # only >=2D parameters will be weight decayed, i.e. all weight tensors in
    # matmuls + embeddings decay, but biases and layernorms won"t.
    decay_params = [p for p in params if p.dim() >= 2]
    nodecay_params = [p for p in params if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer, params


def _compute_batch_metrics(
    probs: torch.Tensor,  # [B, vocab_size]
    y_last: torch.Tensor,  # [B]
    k: int = 10,
) -> dict[str, float]:
    top_k_indices = torch.topk(probs, k=k, dim=-1).indices  # [B, k]
    hits = (top_k_indices == y_last.unsqueeze(-1)).any(dim=-1).float()  # [B]
    return {"hit_rate": hits.mean().item()}


def _aggregate_metrics(all_batch_metrics: list[dict[str, float]]) -> dict[str, float]:
    avg_loss = float(np.mean([m["loss"] for m in all_batch_metrics]))
    avg_hit_rate = float(np.mean([m["hit_rate"] for m in all_batch_metrics]))
    return {"loss": avg_loss, "hit_rate": avg_hit_rate}


def _to_hms(took: float) -> tuple[int, int, int]:
    took = int(took)
    hours = took // 3600
    minutes = (took % 3600) // 60
    seconds = took % 60
    return hours, minutes, seconds


def _print_train_results(
    iter_: int,
    samples_seen: int,
    avg_loss: float,
    lr: float,
    took_hms: tuple[int, int, int],
    samples_per_sec: float,
) -> None:
    h, m, s = took_hms
    print(
        f"🔄 iter: {iter_:>6,} │ "
        f"📊 samples: {samples_seen:>8,} │ "
        f"📉 loss: {avg_loss:>7.4f} │ "
        f"📈 lr: {lr:>9.2e} │ "
        f"⏳ time: {h:02}:{m:02}:{s:02} | "
        f"⚡ {int(samples_per_sec):>4,} samples/s"
    )


def _print_validation_results(
    metrics: dict[str, float],
    samples_seen: int,
    took_hms: tuple[int, int, int],
) -> None:
    h, m, s = took_hms

    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    print(f"📊 METRICS (samples seen: {samples_seen:,}, {h:02}:{m:02}:{s:02})")
    print("-" * 40)
    print(f"  Loss:     {metrics['loss']:.4f}")
    print(f"  Hit rate: {metrics['hit_rate']:.1%}")

    print("\n" + "="*80 + "\n")


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        model: PlaylistRecommender,
        train_dataset: PlaylistDataset,
        validation_dataset: PlaylistDataset,
        device: torch.device
    ):
        self.config = config
        self.model = model.to(device)
        self.device = device

        self.micro_batch_size = config.batch_size // config.gradient_acc_steps
        self.samples_seen = 0
        self.best_loss = float("inf")

        optimizer, trainable_params = _configure_optimizer(
            model, config.weight_decay, config.base_learning_rate, config.betas
        )
        self.optimizer = optimizer
        self.trainable_params = trainable_params

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            pin_memory=config.pin_memory,
            shuffle=True,
            collate_fn=model.tensoriser.collate_fn
        )
        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.micro_batch_size,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            pin_memory=config.pin_memory,
            shuffle=False,
            collate_fn=model.tensoriser.collate_fn
        )
        self.train_iterator = iter(self.train_loader)

        if config.compile:
            self.model = torch.compile(self.model)

        self.ctx = (
            contextlib.nullcontext()
            if device.type == "cpu"
            else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        )

    def _get_next_batch(self, mode: str = "train"):
        if mode == "train":
            iterator = self.train_iterator
        elif mode == "validation":
            iterator = self.validation_iterator
        else:
            raise ValueError(f"Unknown mode '{mode}', expected 'train' or 'validation'")

        try:
            return next(iterator)
        except StopIteration:
            logger.info(f"Exhausted {mode} iterator epoch, restarting from beginning")

            if mode == "train":
                self.train_iterator = iter(self.train_loader)
                return next(self.train_iterator)
            else:
                self.validation_iterator = iter(self.validation_loader)
                return next(self.validation_iterator)

    def _prepare_batch(
        self, batch: dict[str, torch.Tensor | str]
    ) -> dict[str, torch.Tensor | str]:

        def _prepare(element: torch.Tensor | str):
            return (
                element.to(self.device, non_blocking=True)
                if isinstance(element, torch.Tensor)
                else element
            )

        return {k: _prepare(v) for k, v in batch.items()}

    def _samples_in_batch(self, batch: dict[str, torch.Tensor | str]):
        return batch["x"].size(0)

    def _set_optimizer_lr(self):
        lr = _get_learning_rate_stepwise(
            self.samples_seen,
            base_lr=self.config.base_learning_rate,
            min_lr=self.config.min_learning_rate,
            step_size=self.config.lr_step_size,
            gamma=self.config.lr_gamma,
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def _take_optimisation_step(self):
        total_loss = 0

        self._set_optimizer_lr()

        for _ in range(self.config.gradient_acc_steps):
            batch = self._get_next_batch("train")
            batch = self._prepare_batch(batch)

            with self.ctx:
                loss = self.model(**batch)
                loss /= self.config.gradient_acc_steps
                loss.backward()
                total_loss += loss.item()
            
            self.samples_seen += self._samples_in_batch(batch)

        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.trainable_params, self.config.grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return total_loss

    def _crossed_interval(self, interval: int):
        this_iter = self.samples_seen // interval
        prev_iter = (self.samples_seen - self.config.batch_size) // interval
        prev_iter = max(prev_iter, 0)
        return this_iter > prev_iter

    def train(self, n_samples: int):
        logger.info(f"Staring model training for {n_samples} samples...")

        self.model.train()

        n_iter = n_samples // self.config.batch_size + 1

        recent_losses = collections.deque(
            maxlen=max(self.config.log_interval // self.config.batch_size, 1)
        )
        samples_seen_prev = 0
        t0 = time.time()
        t_start = t0

        for i in range(n_iter):
            loss = self._take_optimisation_step()
            recent_losses.append(loss)

            if self._crossed_interval(self.config.log_interval):
                t1 = time.time()
                took = t1 - t0
                t0 = t1

                samples_per_sec = (self.samples_seen - samples_seen_prev) / took
                samples_seen_prev = self.samples_seen

                took_total = t1 - t_start
                took_hms = _to_hms(took_total)

                _print_train_results(
                    iter=i,
                    samples_seen=self.samples_seen,
                    avg_loss=np.mean(recent_losses),
                    lr=self.get_current_lr(),
                    took_hms=took_hms,
                    samples_per_sec=samples_per_sec
                )

            if self._crossed_interval(self.config.validation_interval):
                metrics = self._validate()
                took_total = time.time() - t_start
                took_hms = _to_hms(took_total)
                _print_validation_results(metrics, self.samples_seen, took_hms)
                if self.config.checkpoint_filepath and metrics["loss"] < self.best_loss:
                    self.best_loss = metrics["loss"]
                    self._save_checkpoint(metrics)
                t0 = time.time()

        logger.info("Finished model training.")

    def _validate(self) -> dict:
        self.model.eval()

        self.validation_iterator = iter(self.validation_loader)
        all_batch_metrics = []
        n_iter = self.config.validation_samples // self.micro_batch_size

        with torch.no_grad():
            for _ in range(n_iter):
                batch = self._get_next_batch("validation")
                batch = self._prepare_batch(batch)

                loss = self.model(**batch)
                probs = self.model.last_step_probs(
                    **{k: v for k, v in batch.items() if k != "y"}
                )

                batch_idx = torch.arange(batch["x"].size(0), device=self.device)
                y_last = batch["y"][batch_idx, batch["seq_len"]]  # [B]

                batch_metrics = _compute_batch_metrics(probs, y_last)
                batch_metrics["loss"] = loss.item()

                all_batch_metrics.append(batch_metrics)

        metrics = _aggregate_metrics(all_batch_metrics)
        self.model.train()
        return metrics

    def _save_checkpoint(self, validation_metrics: dict | None = None) -> None:
        cp_dir = os.path.dirname(self.config.checkpoint_filepath)
        if cp_dir and cp_dir != ".":
            os.makedirs(cp_dir, exist_ok=True)

        raw_model = getattr(self.model, "_orig_mod", self.model)
        checkpoint = {
            "datetime": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": raw_model.as_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "trainer_config": asdict(self.config),
            "samples_seen": self.samples_seen,
            "validation_metrics": validation_metrics,
        }
        torch.save(checkpoint, self.config.checkpoint_filepath)
        logger.info(f"Checkpoint saved to '{self.config.checkpoint_filepath}'")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_filepath: str,
        train_dataset: PlaylistDataset,
        validation_dataset: PlaylistDataset,
        device: torch.device,
    ) -> "Trainer":
        checkpoint = torch.load(
            checkpoint_filepath, map_location=device, weights_only=False
        )
        model = PlaylistRecommender.from_dict(checkpoint["model"])
        config = TrainerConfig(**checkpoint["trainer_config"])

        trainer = cls(config, model, train_dataset, validation_dataset, device)
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.samples_seen = checkpoint["samples_seen"]

        if checkpoint.get("validation_metrics") is not None:
            trainer.best_loss = (
                checkpoint["validation_metrics"].get("loss", float("inf"))
            )

        logger.info(
            f"Loaded checkpoint from '{checkpoint_filepath}' "
            f"(samples seen: {trainer.samples_seen:,})"
        )
        return trainer
