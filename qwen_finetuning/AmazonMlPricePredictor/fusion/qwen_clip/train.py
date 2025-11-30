"""
Training script for Qwen-CLIP fusion model
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

from .config import Config
from .dataset import QwenClipDataset, create_data_splits
from .model import TwoBranchFusionModel, AdvancedMoEFusionModel, count_parameters
from ..loss_functions import get_loss_function
from ..utils import (compute_smape, compute_metrics,
                   save_checkpoint, save_history, get_lr, set_seed)


def train_epoch(model, dataloader, criterion, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    use_advanced = config.USE_ADVANCED_MODEL
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        # Move to device
        image_emb = batch['image_emb'].to(device)
        text_emb = batch['text_emb'].to(device)
        prices = batch['price'].to(device)
        
        # Prepare targets
        if config.PREDICT_LOG:
            targets = torch.log1p(prices)
        else:
            targets = prices
        
        # Forward pass
        optimizer.zero_grad()
        model_output = model(image_emb, text_emb)
        predictions = model_output[0] if use_advanced else model_output
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config.GRADIENT_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device, config):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    num_batches = 0
    use_advanced = config.USE_ADVANCED_MODEL
    
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc="Validation", leave=False)
    for batch in pbar:
        # Move to device
        image_emb = batch['image_emb'].to(device)
        text_emb = batch['text_emb'].to(device)
        prices = batch['price'].to(device)
        
        # Prepare targets
        if config.PREDICT_LOG:
            targets = torch.log1p(prices)
        else:
            targets = prices
        
        # Forward pass
        model_output = model(image_emb, text_emb)
        predictions = model_output[0] if use_advanced else model_output
        loss = criterion(predictions, targets)
        
        # Track loss
        total_loss += loss.item()
        num_batches += 1
        
        # Convert predictions back to original scale
        if config.PREDICT_LOG:
            pred_prices = torch.expm1(predictions)
        else:
            pred_prices = predictions
        
        pred_prices = torch.clamp(pred_prices, min=0.0)
        
        all_predictions.extend(pred_prices.cpu().numpy())
        all_targets.extend(prices.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches
    
    # Compute metrics
    metrics = compute_metrics(all_targets, all_predictions)
    metrics['val_loss'] = avg_loss
    
    return metrics


def train_model(train_dataset, val_dataset, config, device):
    """Train the model"""
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Create model
    if config.USE_ADVANCED_MODEL:
        model = AdvancedMoEFusionModel(config).to(device)
        print(f"\nModel: Advanced MoE with Cross-Attention")
        print(f"  - Num experts: {config.NUM_EXPERTS}")
        print(f"  - Attention heads: {config.NUM_ATTENTION_HEADS}")
    else:
        model = TwoBranchFusionModel(config).to(device)
        print(f"\nModel: Base Two-Branch Fusion")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Loss function
    criterion, requires_prices = get_loss_function(config)
    print(f"Loss: {config.LOSS_TYPE}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=config.BETAS,
        eps=config.EPS
    )
    
    # Scheduler
    if config.USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config.SCHEDULER_PATIENCE,
            factor=config.SCHEDULER_FACTOR,
            min_lr=config.SCHEDULER_MIN_LR,
        )
    else:
        scheduler = None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_smape': [],
        'val_mae': [],
        'val_rmse': [],
        'val_r2': [],
        'learning_rate': []
    }
    
    best_smape = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    # Training loop
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, config)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_smape'].append(val_metrics['smape'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_r2'].append(val_metrics['r2'])
        history['learning_rate'].append(get_lr(optimizer))
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_metrics['val_loss']:.4f}")
        print(f"Val SMAPE:  {val_metrics['smape']:.2f}%")
        print(f"Val MAE:    {val_metrics['mae']:.4f}")
        print(f"Val RMSE:   {val_metrics['rmse']:.4f}")
        print(f"Val R²:     {val_metrics['r2']:.4f}")
        print(f"LR:         {get_lr(optimizer):.6f}")
        
        # Save best model
        if val_metrics['smape'] < best_smape:
            best_smape = val_metrics['smape']
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint_path = config.CHECKPOINT_DIR / 'best_model.pt'
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics, checkpoint_path
            )
            print(f"✓ Saved best model (SMAPE: {best_smape:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best SMAPE: {best_smape:.2f}% at epoch {best_epoch + 1}")
            break
        
        # Save periodic checkpoints
        if (epoch + 1) % 100 == 0:
            checkpoint_path = config.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch + 1}.pt'
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics, checkpoint_path
            )
            print(f"✓ Saved checkpoint at epoch {epoch + 1}")
        
        # Learning rate scheduler
        if scheduler:
            scheduler.step(val_metrics['smape'])
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best SMAPE: {best_smape:.2f}% (epoch {best_epoch + 1})")
    print(f"{'='*60}")
    
    # Save history
    history_path = config.OUTPUT_DIR / 'training_history.json'
    save_history(history, history_path)
    
    return best_smape, history


def main():
    """Main training function"""
    # Setup
    config = Config()
    set_seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("QWEN-CLIP FUSION MODEL TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Predict log: {config.PREDICT_LOG}")
    print(f"Train/Val split: {config.TRAIN_VAL_SPLIT:.0%}/{1-config.TRAIN_VAL_SPLIT:.0%}")
    print("="*60)
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Load CSV
    df = pd.read_csv(config.CSV_PATH)
    print(f"Loaded {len(df)} samples from CSV")
    
    # Extract prices and sample IDs
    prices = df['price'].values.astype(np.float32)
    sample_ids = df['sample_id'].values
    
    print(f"\nPrice statistics:")
    print(f"  Mean: ${np.mean(prices):.2f}")
    print(f"  Median: ${np.median(prices):.2f}")
    print(f"  Min: ${np.min(prices):.2f}")
    print(f"  Max: ${np.max(prices):.2f}")
    
    # Create train/val split (80:20)
    print(f"\n{'='*60}")
    print(f"CREATING TRAIN/VAL SPLIT ({config.TRAIN_VAL_SPLIT:.0%}:{1-config.TRAIN_VAL_SPLIT:.0%})")
    print(f"{'='*60}")
    
    # Stratified split by price bins
    from sklearn.preprocessing import KBinsDiscretizer
    binner = KBinsDiscretizer(n_bins=config.STRATIFY_BINS, encode='ordinal', strategy='quantile')
    price_bins = binner.fit_transform(prices.reshape(-1, 1)).ravel()
    
    train_idx, val_idx = train_test_split(
        np.arange(len(prices)),
        test_size=1 - config.TRAIN_VAL_SPLIT,
        random_state=config.RANDOM_SEED,
        stratify=price_bins
    )
    
    print(f"  Train: {len(train_idx)} samples ({100*len(train_idx)/len(prices):.1f}%)")
    print(f"  Val:   {len(val_idx)} samples ({100*len(val_idx)/len(prices):.1f}%)")
    
    # Create datasets
    train_dataset, val_dataset = create_data_splits(
        clip_embeddings_dir=config.CLIP_EMBEDDINGS_DIR,
        qwen_embeddings_path=config.QWEN_EMBEDDINGS_PATH,
        csv_path=config.CSV_PATH,
        prices=prices,
        sample_ids=sample_ids,
        train_idx=train_idx,
        val_idx=val_idx
    )
    
    # Train model
    best_smape, history = train_model(
        train_dataset, val_dataset, config, device
    )
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best SMAPE: {best_smape:.2f}%")
    print(f"{'='*60}")
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()

