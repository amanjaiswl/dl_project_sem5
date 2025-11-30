"""
Custom loss functions with special handling for high-priced items
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedHuberLoss(nn.Module):
    """
    Huber loss with sample-specific weights
    Gives more importance to high-priced items
    """
    
    def __init__(self, delta=1.0, weight_power=0.5, high_price_threshold=100.0, 
                 high_price_multiplier=2.0):
        """
        Args:
            delta: Huber delta parameter
            weight_power: Power for price-based weighting (weight = price^power)
            high_price_threshold: Threshold for high-price boost
            high_price_multiplier: Extra multiplier for high-priced items
        """
        super().__init__()
        self.delta = delta
        self.weight_power = weight_power
        self.high_price_threshold = high_price_threshold
        self.high_price_multiplier = high_price_multiplier
    
    def forward(self, predictions, targets, prices=None):
        """
        Args:
            predictions: Model predictions (batch,)
            targets: Ground truth targets (batch,)
            prices: Original prices for weighting (batch,)
            
        Returns:
            loss: Weighted Huber loss
        """
        # Compute element-wise Huber loss
        diff = predictions - targets
        abs_diff = torch.abs(diff)
        
        huber_loss = torch.where(
            abs_diff <= self.delta,
            0.5 * diff ** 2,
            self.delta * (abs_diff - 0.5 * self.delta)
        )
        
        # Compute sample weights
        if prices is not None:
            # Base weight from price
            weights = torch.pow(prices + 1e-6, self.weight_power)
            
            # Extra weight for high-priced items
            high_price_mask = prices > self.high_price_threshold
            weights = torch.where(
                high_price_mask,
                weights * self.high_price_multiplier,
                weights
            )
            
            # Normalize weights to have mean=1 (keeps loss scale similar)
            weights = weights / (weights.mean() + 1e-8)
        else:
            weights = torch.ones_like(huber_loss)
        
        # Apply weights
        weighted_loss = huber_loss * weights
        
        return weighted_loss.mean()


class QuantileHuberLoss(nn.Module):
    """
    Huber loss that focuses on different quantiles
    Helps with imbalanced distribution of prices
    """
    
    def __init__(self, delta=1.0, quantiles=[0.1, 0.5, 0.9]):
        """
        Args:
            delta: Huber delta parameter
            quantiles: List of quantiles to optimize for
        """
        super().__init__()
        self.delta = delta
        self.quantiles = torch.tensor(quantiles)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model predictions (batch,)
            targets: Ground truth targets (batch,)
            
        Returns:
            loss: Quantile Huber loss
        """
        predictions = predictions.unsqueeze(1)  # (batch, 1)
        targets = targets.unsqueeze(1)  # (batch, 1)
        quantiles = self.quantiles.to(predictions.device).unsqueeze(0)  # (1, n_quantiles)
        
        errors = targets - predictions  # (batch, 1)
        
        # Quantile loss
        quantile_loss = torch.max(
            quantiles * errors,
            (quantiles - 1) * errors
        )
        
        # Add Huber smoothing
        abs_errors = torch.abs(errors)
        huber_loss = torch.where(
            abs_errors <= self.delta,
            0.5 * errors ** 2,
            self.delta * (abs_errors - 0.5 * self.delta)
        )
        
        # Combine
        loss = (quantile_loss + huber_loss).mean()
        
        return loss


class AdaptiveHuberLoss(nn.Module):
    """
    Huber loss with adaptive delta based on price range
    Uses smaller delta (more sensitive) for high-priced items
    """
    
    def __init__(self, base_delta=1.0, min_delta=0.5, max_delta=2.0,
                 adapt_on_log=True):
        """
        Args:
            base_delta: Base delta value
            min_delta: Minimum delta for high-priced items
            max_delta: Maximum delta for low-priced items
            adapt_on_log: Whether to adapt based on log-price
        """
        super().__init__()
        self.base_delta = base_delta
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.adapt_on_log = adapt_on_log
    
    def forward(self, predictions, targets, prices=None):
        """
        Args:
            predictions: Model predictions (batch,)
            targets: Ground truth targets (batch,)
            prices: Original prices for adaptation (batch,)
            
        Returns:
            loss: Adaptive Huber loss
        """
        diff = predictions - targets
        abs_diff = torch.abs(diff)
        
        if prices is not None:
            # Compute adaptive delta based on price
            if self.adapt_on_log:
                price_values = torch.log1p(prices)
            else:
                price_values = prices
            
            # Normalize to [0, 1]
            price_norm = (price_values - price_values.min()) / (price_values.max() - price_values.min() + 1e-8)
            
            # Higher prices get smaller delta (more sensitive)
            delta = self.max_delta - price_norm * (self.max_delta - self.min_delta)
        else:
            delta = self.base_delta
        
        # Huber loss with adaptive delta
        huber_loss = torch.where(
            abs_diff <= delta,
            0.5 * diff ** 2,
            delta * (abs_diff - 0.5 * delta)
        )
        
        return huber_loss.mean()


class FocalHuberLoss(nn.Module):
    """
    Focal loss variant of Huber loss
    Focuses training on hard examples (large errors)
    """
    
    def __init__(self, delta=1.0, gamma=2.0, alpha=1.0):
        """
        Args:
            delta: Huber delta parameter
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Scaling factor
        """
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model predictions (batch,)
            targets: Ground truth targets (batch,)
            
        Returns:
            loss: Focal Huber loss
        """
        diff = predictions - targets
        abs_diff = torch.abs(diff)
        
        # Huber loss
        huber_loss = torch.where(
            abs_diff <= self.delta,
            0.5 * diff ** 2,
            self.delta * (abs_diff - 0.5 * self.delta)
        )
        
        # Focal weighting (focus on hard examples)
        # Normalize loss to [0, 1] for focal weighting
        huber_norm = huber_loss / (huber_loss.max() + 1e-8)
        focal_weight = torch.pow(huber_norm, self.gamma)
        
        focal_loss = self.alpha * focal_weight * huber_loss
        
        return focal_loss.mean()


class SMAPELoss(nn.Module):
    """
    Direct SMAPE (Symmetric Mean Absolute Percentage Error) loss
    
    SMAPE penalizes relative errors more than absolute errors.
    Naturally focuses on low-value predictions where SMAPE is most sensitive.
    
    IMPORTANT: SMAPE must be computed on original price scale, not log-space!
    """
    
    def __init__(self, eps=1e-8, predict_log=True):
        """
        Args:
            eps: Epsilon for numerical stability (prevents division by zero)
            predict_log: If True, converts predictions/targets from log-space to original scale
        """
        super().__init__()
        self.eps = eps
        self.predict_log = predict_log
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model predictions (batch,) - may be in log-space
            targets: Ground truth targets (batch,) - may be in log-space
            
        Returns:
            loss: Mean SMAPE loss (range: 0-200, but typically 0-100)
        """
        # Convert from log-space to original scale if needed
        if self.predict_log:
            predictions = torch.expm1(predictions)
            targets = torch.expm1(targets)
            predictions = torch.clamp(predictions, min=0.0)
            targets = torch.clamp(targets, min=0.0)
        
        numerator = torch.abs(predictions - targets)
        denominator = torch.abs(predictions) + torch.abs(targets) + self.eps
        smape = 200.0 * numerator / denominator
        
        return smape.mean()


class FocalSMAPELoss(nn.Module):
    """
    Focal SMAPE Loss - Penalizes high SMAPE errors more heavily
    
    Instead of linearly penalizing by SMAPE, applies a power function
    to amplify the penalty for samples with high SMAPE.
    
    This makes the model focus more on worst predictions:
    - Sample with SMAPE=10% → loss ≈ 0.01 (with gamma=2)
    - Sample with SMAPE=50% → loss ≈ 0.25 (25x worse, not just 5x!)
    
    Perfect for imbalanced distributions where you want to avoid
    terrible predictions on any sample (low or high price).
    
    IMPORTANT: SMAPE must be computed on original price scale, not log-space!
    """
    
    def __init__(self, gamma=2.0, eps=1e-8, normalize=True, predict_log=True):
        """
        Args:
            gamma: Focusing parameter (higher = more aggressive)
                   - gamma=1.0: Same as regular SMAPE
                   - gamma=2.0: Quadratic penalty (recommended)
                   - gamma=3.0: Cubic penalty (very aggressive)
            eps: Epsilon for numerical stability
            normalize: If True, normalize SMAPE to [0,2] before applying power
            predict_log: If True, converts predictions/targets from log-space to original scale
        """
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.normalize = normalize
        self.predict_log = predict_log
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model predictions (batch,) - may be in log-space
            targets: Ground truth targets (batch,) - may be in log-space
            
        Returns:
            loss: Mean focal SMAPE loss (computed on original scale)
        """
        # Convert from log-space to original scale if needed
        if self.predict_log:
            predictions = torch.expm1(predictions)  # inverse of log1p
            targets = torch.expm1(targets)
            # Clamp to avoid negative prices from numerical issues
            predictions = torch.clamp(predictions, min=0.0)
            targets = torch.clamp(targets, min=0.0)
        
        # Compute per-sample SMAPE on original price scale
        numerator = torch.abs(predictions - targets)
        denominator = torch.abs(predictions) + torch.abs(targets) + self.eps
        smape = 200.0 * numerator / denominator  # Range: [0, 200]
        
        if self.normalize:
            # Normalize to [0, 2] for more stable gradients
            smape = smape / 100.0
        
        # Apply focal penalty: amplify high SMAPE samples
        focal_loss = torch.pow(smape, self.gamma)
        
        return focal_loss.mean()


class HybridFocalSMAPELoss(nn.Module):
    """
    Hybrid loss combining stable base loss with Focal SMAPE
    
    Early training: More weight on stable loss (Huber/MSE)
    Later training: More weight on Focal SMAPE
    
    This provides:
    - Stable gradients early in training
    - Direct SMAPE optimization as training progresses
    """
    
    def __init__(self, base_loss='huber', huber_delta=1.0, 
                 smape_gamma=2.0, smape_weight=0.3, eps=1e-8, predict_log=True):
        """
        Args:
            base_loss: Base loss type ('huber', 'mse', 'mae')
            huber_delta: Delta for Huber loss
            smape_gamma: Gamma for Focal SMAPE
            smape_weight: Weight for SMAPE component (0-1)
                         - 0.0: Pure base loss
                         - 0.5: Equal weight
                         - 1.0: Pure SMAPE loss
            eps: Epsilon for SMAPE
            predict_log: If True, converts predictions/targets for SMAPE calculation
        """
        super().__init__()
        self.smape_weight = smape_weight
        self.base_weight = 1.0 - smape_weight
        
        # Base loss (operates in log-space, no conversion needed)
        if base_loss == 'huber':
            self.base_criterion = nn.HuberLoss(delta=huber_delta)
        elif base_loss == 'mse':
            self.base_criterion = nn.MSELoss()
        elif base_loss == 'mae':
            self.base_criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")
        
        # SMAPE loss (needs to operate in original scale)
        self.smape_criterion = FocalSMAPELoss(gamma=smape_gamma, eps=eps, predict_log=predict_log)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model predictions (batch,)
            targets: Ground truth targets (batch,)
            
        Returns:
            loss: Weighted combination of base loss and focal SMAPE
        """
        base_loss = self.base_criterion(predictions, targets)
        smape_loss = self.smape_criterion(predictions, targets)
        
        # Weighted combination
        total_loss = self.base_weight * base_loss + self.smape_weight * smape_loss
        
        return total_loss


class WeightedSMAPELoss(nn.Module):
    """
    SMAPE loss with tiered weights based on price ranges.
    Higher weights for lower prices to focus on cheap items.
    """
    def __init__(self, eps=1e-8, predict_log=True):
        super().__init__()
        self.eps = eps
        self.predict_log = predict_log
        
        # Tiered weighting: lower prices get higher weights
        # 0-5: 10x, 5-10: 8x, 10-30: 6x, 30-60: 4x, 60-200: 2x, 200+: 1x
        self.price_thresholds = [5, 30, 200]
        self.weights = [10.0, 3.0, 1.0]
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model predictions (in log space if predict_log=True)
            targets: Ground truth targets (in log space if predict_log=True)
        """
        # Convert from log space if needed
        if self.predict_log:
            predictions = torch.expm1(predictions)  # exp(x) - 1
            targets = torch.expm1(targets)
            predictions = torch.clamp(predictions, min=0.0)
            targets = torch.clamp(targets, min=0.0)
        
        # Compute SMAPE for each sample
        numerator = torch.abs(predictions - targets)
        denominator = torch.abs(predictions) + torch.abs(targets) + self.eps
        smape = 200.0 * numerator / denominator  # [0, 200] range
        
        # Compute tiered weights based on target price
        weights = torch.ones_like(targets) * self.weights[-1]  # Default: >100
        
        for i, threshold in enumerate(self.price_thresholds):
            mask = targets < threshold
            weights = torch.where(mask, torch.full_like(targets, self.weights[i]), weights)
        
        # Weighted SMAPE
        weighted_smape = smape * weights
        
        return weighted_smape.mean()


def get_loss_function(config):
    """
    Get loss function based on config
    
    Args:
        config: Configuration object
        
    Returns:
        loss_fn: Loss function
        requires_prices: Whether loss function needs original prices
    """
    loss_type = config.LOSS_TYPE.lower()
    
    if loss_type == "weighted_huber":
        return WeightedHuberLoss(
            delta=config.HUBER_DELTA,
            weight_power=config.PRICE_WEIGHT_POWER,
            high_price_threshold=config.HIGH_PRICE_THRESHOLD,
            high_price_multiplier=config.HIGH_PRICE_WEIGHT_MULTIPLIER
        ), True
    
    elif loss_type == "quantile_huber":
        return QuantileHuberLoss(delta=config.HUBER_DELTA), False
    
    elif loss_type == "adaptive_huber":
        return AdaptiveHuberLoss(
            base_delta=config.HUBER_DELTA,
            adapt_on_log=config.PREDICT_LOG
        ), True
    
    elif loss_type == "focal_huber":
        return FocalHuberLoss(delta=config.HUBER_DELTA), False
    
    elif loss_type == "smape":
        eps = getattr(config, 'SMAPE_EPS', 1e-8)
        predict_log = getattr(config, 'PREDICT_LOG', False)
        return SMAPELoss(eps=eps, predict_log=predict_log), False
    
    elif loss_type == "focal_smape":
        gamma = getattr(config, 'SMAPE_GAMMA', 2.0)
        eps = getattr(config, 'SMAPE_EPS', 1e-8)
        normalize = getattr(config, 'SMAPE_NORMALIZE', True)
        predict_log = getattr(config, 'PREDICT_LOG', False)
        return FocalSMAPELoss(gamma=gamma, eps=eps, normalize=normalize, predict_log=predict_log), False
    
    elif loss_type == "hybrid_focal_smape":
        base_loss = getattr(config, 'HYBRID_BASE_LOSS', 'huber')
        smape_gamma = getattr(config, 'SMAPE_GAMMA', 2.0)
        smape_weight = getattr(config, 'SMAPE_WEIGHT', 0.3)
        eps = getattr(config, 'SMAPE_EPS', 1e-8)
        predict_log = getattr(config, 'PREDICT_LOG', False)
        return HybridFocalSMAPELoss(
            base_loss=base_loss,
            huber_delta=config.HUBER_DELTA,
            smape_gamma=smape_gamma,
            smape_weight=smape_weight,
            eps=eps,
            predict_log=predict_log
        ), False
    
    elif loss_type == "huber":
        return nn.HuberLoss(delta=config.HUBER_DELTA), False
    
    elif loss_type == "mse":
        return nn.MSELoss(), False
    
    elif loss_type == "mae":
        return nn.L1Loss(), False

    elif loss_type == "weighted_smape":
        return WeightedSMAPELoss(
            predict_log=config.PREDICT_LOG
        ), True  # requires_prices=True
    
    
    else:
        raise ValueError(f"Unknown loss type: {config.LOSS_TYPE}")


# Testing
if __name__ == "__main__":
    # Test weighted Huber loss
    predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    targets = torch.tensor([1.1, 2.2, 3.3, 4.4, 5.5])
    prices = torch.tensor([10.0, 20.0, 50.0, 100.0, 200.0])
    
    print("Testing Weighted Huber Loss:")
    loss_fn = WeightedHuberLoss(delta=1.0, weight_power=0.5, 
                                high_price_threshold=100.0,
                                high_price_multiplier=2.0)
    loss = loss_fn(predictions, targets, prices)
    print(f"Loss: {loss.item():.4f}")
    
    print("\nTesting Quantile Huber Loss:")
    loss_fn = QuantileHuberLoss(delta=1.0)
    loss = loss_fn(predictions, targets)
    print(f"Loss: {loss.item():.4f}")
    
    print("\nTesting Adaptive Huber Loss:")
    loss_fn = AdaptiveHuberLoss(base_delta=1.0)
    loss = loss_fn(predictions, targets, prices)
    print(f"Loss: {loss.item():.4f}")
    
    print("\nTesting Focal Huber Loss:")
    loss_fn = FocalHuberLoss(delta=1.0, gamma=2.0)
    loss = loss_fn(predictions, targets)
    print(f"Loss: {loss.item():.4f}")
    
    # Test new SMAPE losses
    print("\n" + "="*60)
    print("TESTING NEW SMAPE LOSSES")
    print("="*60)
    
    # Simulate different error scenarios
    # Good predictions vs bad predictions
    good_preds = torch.tensor([10.0, 20.0, 100.0])
    good_targets = torch.tensor([11.0, 21.0, 105.0])
    
    bad_preds = torch.tensor([10.0, 20.0, 100.0])
    bad_targets = torch.tensor([15.0, 30.0, 150.0])
    
    print("\nScenario: Good predictions (5-10% error)")
    print(f"Predictions: {good_preds.tolist()}")
    print(f"Targets:     {good_targets.tolist()}")
    
    # Regular SMAPE
    smape_loss = SMAPELoss()
    loss_good = smape_loss(good_preds, good_targets)
    print(f"\nSMAPE Loss: {loss_good.item():.4f}")
    
    # Focal SMAPE (gamma=2)
    focal_smape = FocalSMAPELoss(gamma=2.0)
    loss_good_focal = focal_smape(good_preds, good_targets)
    print(f"Focal SMAPE (γ=2): {loss_good_focal.item():.6f}")
    
    print("\n" + "-"*60)
    print("\nScenario: Bad predictions (33-50% error)")
    print(f"Predictions: {bad_preds.tolist()}")
    print(f"Targets:     {bad_targets.tolist()}")
    
    loss_bad = smape_loss(bad_preds, bad_targets)
    print(f"\nSMAPE Loss: {loss_bad.item():.4f}")
    
    loss_bad_focal = focal_smape(bad_preds, bad_targets)
    print(f"Focal SMAPE (γ=2): {loss_bad_focal.item():.6f}")
    
    print("\n" + "-"*60)
    print("\nAmplification effect:")
    print(f"Regular SMAPE:  Bad/Good ratio = {loss_bad.item()/loss_good.item():.2f}x")
    print(f"Focal SMAPE:    Bad/Good ratio = {loss_bad_focal.item()/loss_good_focal.item():.2f}x")
    print("→ Focal SMAPE amplifies the penalty for worse predictions!")
    
    # Test hybrid loss
    print("\n" + "="*60)
    print("Testing Hybrid Focal SMAPE Loss")
    print("="*60)
    hybrid = HybridFocalSMAPELoss(base_loss='huber', smape_weight=0.3)
    loss_hybrid = hybrid(bad_preds, bad_targets)
    print(f"Hybrid Loss (30% SMAPE, 70% Huber): {loss_hybrid.item():.4f}")

