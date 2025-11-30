"""
Utility functions for training and evaluation
"""
import numpy as np
import torch
import json
from pathlib import Path


def compute_smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
    
    Formula: SMAPE = (1/n) * Σ |predicted - actual| / ((|actual| + |predicted|)/2)
    
    Args:
        y_true: Ground truth prices
        y_pred: Predicted prices
        
    Returns:
        SMAPE score (0-100)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Ensure non-negative
    y_pred = np.maximum(y_pred, 0)
    
    # Calculate SMAPE: |pred - true| / ((|true| + |pred|)/2)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Avoid division by zero
    mask = denominator != 0
    smape = np.zeros_like(numerator)
    smape[mask] = numerator[mask] / denominator[mask]
    
    return 100 * np.mean(smape)


def compute_metrics(y_true, y_pred):
    """
    Compute various regression metrics
    
    Returns:
        dict with metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Ensure non-negative predictions
    y_pred = np.maximum(y_pred, 0)
    
    # Calculate metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = 100 * np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))
    smape = compute_smape(y_true, y_pred)
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'smape': smape,
        'r2': r2
    }


class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience=7, min_delta=0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metrics']


def save_history(history, filepath):
    """Save training history to JSON"""
    # Convert numpy types to Python types
    history_json = {}
    for key, values in history.items():
        if isinstance(values, list):
            history_json[key] = [float(v) if not isinstance(v, (list, dict)) else v for v in values]
        else:
            history_json[key] = values
    
    with open(filepath, 'w') as f:
        json.dump(history_json, f, indent=2)


def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_seed(seed):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Example usage
if __name__ == "__main__":
    # Test SMAPE calculation
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 310, 380, 520])
    
    smape = compute_smape(y_true, y_pred)
    print(f"SMAPE: {smape:.2f}%")
    
    metrics = compute_metrics(y_true, y_pred)
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=3, mode='min')
    scores = [10, 9, 8.5, 8.4, 8.5, 8.6, 8.7]
    
    print("\nEarly stopping test:")
    for i, score in enumerate(scores):
        stop = early_stopping(score)
        print(f"  Epoch {i+1}: score={score:.2f}, stop={stop}")
        if stop:
            break

