"""Training layer: UBM, LDA, total variability, NBC and PLDA fitting."""

from __future__ import annotations

from .trainer import MAPTrainer, TrainedBundle, fit_ubm, train_bundle

__all__ = ["MAPTrainer", "TrainedBundle", "fit_ubm", "train_bundle"]
