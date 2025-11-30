"""
Fusion models for price prediction combining multiple embeddings
"""

from .loss_functions import (
    WeightedHuberLoss,
    QuantileHuberLoss,
    AdaptiveHuberLoss,
    FocalHuberLoss,
    SMAPELoss,
    FocalSMAPELoss,
    HybridFocalSMAPELoss,
    WeightedSMAPELoss,
    get_loss_function
)

from .utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    calculate_smape,
    calculate_metrics,
    EarlyStopping
)

__all__ = [
    # Loss functions
    'WeightedHuberLoss',
    'QuantileHuberLoss',
    'AdaptiveHuberLoss',
    'FocalHuberLoss',
    'SMAPELoss',
    'FocalSMAPELoss',
    'HybridFocalSMAPELoss',
    'WeightedSMAPELoss',
    'get_loss_function',
    # Utils
    'set_seed',
    'save_checkpoint',
    'load_checkpoint',
    'calculate_smape',
    'calculate_metrics',
    'EarlyStopping',
]

