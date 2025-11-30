"""
Fine-tuning script for Qwen3-Embedding-0.6B with regression head using QLoRA.
This script fine-tunes the entire model (backbone + regression head) on the price prediction task.
"""

import sys
import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import DataParallel
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
from tqdm import tqdm
import time
import math

# Add pipeline config to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import *
from models.qwen3_regression_model import create_model_for_training

class PriceDataset(Dataset):
    """Dataset for price prediction with text inputs"""
    
    def __init__(self, texts, prices, values, units, tokenizer, max_length=512, use_log_price=True, price_offset=1.0):
        self.texts = texts
        self.prices = prices
        self.values = values
        self.units = units
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_log_price = use_log_price
        self.price_offset = price_offset
        
        # Preprocess prices
        if use_log_price:
            self.targets = np.log(prices + price_offset)
        else:
            self.targets = prices
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        price = self.targets[idx]
        value = float(self.values[idx])
        unit = int(self.units[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'price': torch.tensor(price, dtype=torch.float32),
            'value': torch.tensor(value, dtype=torch.float32),
            'unit': torch.tensor(unit, dtype=torch.long)
        }

def smape_loss(y_pred, y_true, epsilon=1e-8):
    """Symmetric Mean Absolute Percentage Error loss"""
    # Convert back to actual prices for SMAPE calculation
    y_pred_actual = torch.exp(y_pred) - PRICE_OFFSET
    y_true_actual = torch.exp(y_true) - PRICE_OFFSET
    
    numerator = torch.abs(y_pred_actual - y_true_actual)
    denominator = (torch.abs(y_true_actual) + torch.abs(y_pred_actual)) / 2 + epsilon
    return torch.mean(numerator / denominator) * 100

def huber_smape_loss(y_pred, y_true, delta=1.0, epsilon=1e-8):
    """Combined Huber + SMAPE loss for robust training"""
    # Huber loss component (on log prices)
    huber_loss = nn.HuberLoss(delta=delta)(y_pred, y_true)
    
    # SMAPE component (on actual prices) - scaled down for better balance
    smape = smape_loss(y_pred, y_true, epsilon) / 200.0  # Scale from ~130% to ~0.65
    
    # Combine (balanced weighting)
    return huber_loss + smape

def train_epoch(model, dataloader, optimizer, criterion, device, scheduler, gradient_accumulation_steps=1, print_interval=50, val_dataloader=None, val_interval=200):
    """Train for one epoch with comprehensive metric printing"""
    model.train()
    total_loss = 0
    total_huber_loss = 0
    total_smape_loss = 0
    num_batches = 0
    running_loss = 0
    running_huber = 0
    running_smape = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        prices = batch['price'].to(device)
        values = batch['value'].to(device)
        units = batch['unit'].to(device)
        
        # Forward pass
        log_prices = model(input_ids, attention_mask, values, units)
        
        # Clamp log_prices to prevent extreme values that cause NaN
        log_prices = torch.clamp(log_prices, min=-10.0, max=10.0)  # Reasonable range for log prices
        
        # Check for NaN in predictions
        if torch.isnan(log_prices).any():
            print(f"    Warning: NaN detected in predictions at batch {batch_idx}")
            log_prices = torch.nan_to_num(log_prices, nan=0.0, posinf=5.0, neginf=-5.0)
        
        loss = criterion(log_prices, prices)
        
        # Calculate individual loss components for monitoring
        with torch.no_grad():
            # Huber loss component
            huber_loss = nn.HuberLoss(delta=1.0)(log_prices, prices)
            
            # SMAPE loss component (scaled for consistency with loss function)
            y_pred_actual = torch.exp(log_prices) - PRICE_OFFSET
            y_true_actual = torch.exp(prices) - PRICE_OFFSET
            numerator = torch.abs(y_pred_actual - y_true_actual)
            denominator = (torch.abs(y_true_actual) + torch.abs(y_pred_actual)) / 2 + 1e-8
            smape_loss = torch.mean(numerator / denominator) * 100 / 200.0  # Scaled down
        
        # Scale loss by accumulation steps
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # Calculate gradient norm before clipping
        grad_norm = 0.0
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Calculate gradient norm before clipping
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** (1. / 2)
            
            # Clip gradients and update
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()  # Update learning rate after each gradient update
            optimizer.zero_grad()
        
        # Track losses
        batch_loss = loss.item() * gradient_accumulation_steps
        batch_huber = huber_loss.item()
        batch_smape = smape_loss.item()
        
        total_loss += batch_loss
        total_huber_loss += batch_huber
        total_smape_loss += batch_smape
        running_loss += batch_loss
        running_huber += batch_huber
        running_smape += batch_smape
        num_batches += 1
        
        # Print comprehensive metrics every print_interval batches
        if (batch_idx + 1) % print_interval == 0:
            avg_running_loss = running_loss / print_interval
            avg_running_huber = running_huber / print_interval
            avg_running_smape = running_smape / print_interval
            current_lr = optimizer.param_groups[0]['lr']
            
            # Get current step for scheduler info
            current_step = (batch_idx + 1) // gradient_accumulation_steps
            
            print(f"    Batch {batch_idx + 1:4d}/{len(dataloader):4d} | "
                  f"Loss: {avg_running_loss:.4f} | "
                  f"Huber: {avg_running_huber:.4f} | "
                  f"SMAPE: {avg_running_smape:.2f}% | "
                  f"LR: {current_lr:.2e} | "
                  f"GradNorm: {grad_norm:.3f} | "
                  f"Step: {current_step}")
            
            running_loss = 0
            running_huber = 0
            running_smape = 0
        
        # Quick validation every val_interval batches
        if val_dataloader is not None and (batch_idx + 1) % val_interval == 0:
            val_loss, val_mae, val_mse, val_rmse, val_r2, val_smape = validate_epoch(
                model, val_dataloader, criterion, device, sample_size=500
            )
            print(f"    [VAL] Loss: {val_loss:.4f} | MAE: ${val_mae:.2f} | SMAPE: {val_smape:.2f}%")
    
    return total_loss / num_batches, total_huber_loss / num_batches, total_smape_loss / num_batches

def validate_epoch(model, dataloader, criterion, device, sample_size=None):
    """Validate for one epoch with optional sampling for faster evaluation"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_prices = []
    num_batches = 0
    
    # Sample validation data if requested for faster evaluation
    if sample_size is not None and len(dataloader.dataset) > sample_size:
        # Create a subset of the validation dataset
        indices = np.random.choice(len(dataloader.dataset), sample_size, replace=False)
        subset = torch.utils.data.Subset(dataloader.dataset, indices)
        dataloader = DataLoader(subset, batch_size=dataloader.batch_size, shuffle=False)
        print(f"    Using {sample_size:,} samples for faster validation")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            prices = batch['price'].to(device)
            values = batch['value'].to(device)
            units = batch['unit'].to(device)
            
            log_prices = model(input_ids, attention_mask, values, units)
            
            # Clamp log_prices to prevent extreme values that cause NaN
            log_prices = torch.clamp(log_prices, min=-10.0, max=10.0)
            
            # Check for NaN in predictions
            if torch.isnan(log_prices).any():
                log_prices = torch.nan_to_num(log_prices, nan=0.0, posinf=5.0, neginf=-5.0)
            
            loss = criterion(log_prices, prices)
            
            total_loss += loss.item()
            all_predictions.extend(log_prices.cpu().numpy())
            all_prices.extend(prices.cpu().numpy())
            num_batches += 1
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_prices = np.array(all_prices)
    
    # Convert back to actual prices for metrics
    pred_actual = np.exp(all_predictions) - PRICE_OFFSET
    true_actual = np.exp(all_prices) - PRICE_OFFSET
    
    # Handle NaN values by filtering them out
    valid_mask = np.isfinite(pred_actual) & np.isfinite(true_actual)
    nan_count = np.sum(~valid_mask)
    
    if nan_count > 0:
        print(f"    Warning: {nan_count}/{len(pred_actual)} predictions contain NaN/inf values")
        print(f"    Log price range: [{np.min(all_predictions):.3f}, {np.max(all_predictions):.3f}]")
        print(f"    Actual price range: [{np.min(pred_actual):.3f}, {np.max(pred_actual):.3f}]")
    
    if not np.any(valid_mask):
        print("    Warning: All predictions are NaN, returning default metrics")
        return total_loss / num_batches, float('inf'), float('inf'), float('inf'), 0.0, float('inf')
    
    pred_actual_clean = pred_actual[valid_mask]
    true_actual_clean = true_actual[valid_mask]
    
    if len(pred_actual_clean) == 0:
        print("    Warning: No valid predictions after NaN filtering")
        return total_loss / num_batches, float('inf'), float('inf'), float('inf'), 0.0, float('inf')
    
    mae = mean_absolute_error(true_actual_clean, pred_actual_clean)
    mse = mean_squared_error(true_actual_clean, pred_actual_clean)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_actual_clean, pred_actual_clean)
    smape = smape_loss(torch.FloatTensor(all_predictions[valid_mask]), torch.FloatTensor(all_prices[valid_mask])).item() / 200.0  # Scaled down
    
    return total_loss / num_batches, mae, mse, rmse, r2, smape

def save_checkpoint(model, optimizer, scheduler, epoch, training_history, config, output_dir, is_best=False):
    """Save model checkpoint"""
    # Handle DataParallel models by saving the underlying module
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'training_history': training_history,
        'config': config,
        'timestamp': time.time()
    }
    
    # Save regular checkpoint
    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model separately
    if is_best:
        best_path = output_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"    ✓ Best model saved: {best_path}")
    
    print(f"    Checkpoint saved: {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint for resuming training"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state - handle DataParallel models
    if hasattr(model, 'module'):
        # DataParallel wrapped model
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Regular model
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    training_history = checkpoint.get('training_history', {
        'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_r2': [], 'val_smape': []
    })
    
    print(f"✓ Checkpoint loaded from epoch {epoch}")
    return epoch, training_history

def create_unit_vocab(units):
    """Create vocabulary mapping for units"""
    unique_units = sorted(list(set(units)))
    unit_to_idx = {unit: idx for idx, unit in enumerate(unique_units)}
    idx_to_unit = {idx: unit for unit, idx in unit_to_idx.items()}
    
    print(f"✓ Created unit vocabulary with {len(unique_units)} unique units")
    print(f"  Unit range: 0 - {len(unique_units)-1}")
    
    return unit_to_idx, idx_to_unit

def load_data(train_processed_path):
    """Load and prepare training data"""
    print(f"\nLoading training data from: {train_processed_path}")
    
    df = pd.read_csv(train_processed_path)
    
    # Filter out rows with missing data
    df = df.dropna(subset=['combined_text', 'price', 'value', 'unit'])
    
    texts = df['combined_text'].tolist()
    prices = df['price'].values
    values = df['value'].values
    units = df['unit'].values
    
    # Create unit vocabulary and convert units to indices
    unit_to_idx, idx_to_unit = create_unit_vocab(units)
    unit_indices = np.array([unit_to_idx[unit] for unit in units])
    
    print(f"✓ Loaded {len(texts):,} samples")
    print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    print(f"  Mean price: ${prices.mean():.2f}")
    print(f"  Value range: {values.min():.2f} - {values.max():.2f}")
    print(f"  Unit indices range: {unit_indices.min()} - {unit_indices.max()}")
    
    return texts, prices, values, unit_indices, unit_to_idx, idx_to_unit

def train_model(model, tokenizer, texts, prices, values, units, config, output_dir, resume_from=None):
    """Train the fine-tuned model with checkpointing and resume capability"""
    print("\n" + "="*80)
    print("FINE-TUNING MODEL")
    print("="*80)
    
    # Set random seeds
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # Create dataset
    dataset = PriceDataset(
        texts, prices, values, units, tokenizer, 
        max_length=config['max_length'],
        use_log_price=config['use_log_price'],
        price_offset=config['price_offset']
    )
    
    # Split data
    val_size = int(len(dataset) * config['val_split_ratio'])
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config['random_seed'])
    )
    
    print(f"\nData split:")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Move model to device and wrap with DataParallel if using multiple GPUs
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Check for multiple GPUs and wrap with DataParallel
    if config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
        print(f"\nUsing {torch.cuda.device_count()} GPUs for training")
        model = DataParallel(model)
        # Update effective batch size
        effective_batch_size = config['batch_size'] * torch.cuda.device_count()
        print(f"  Effective batch size: {effective_batch_size} (per-GPU: {config['batch_size']})")
    else:
        print(f"\nUsing single GPU: {device}")
    
    # Print model info
    if hasattr(model, 'module'):
        # DataParallel wrapped model
        trainable_params, total_params = model.module.get_trainable_parameters()
    else:
        # Regular model
        trainable_params, total_params = model.get_trainable_parameters()
    
    print(f"\nModel info:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable %: {trainable_params/total_params*100:.2f}%")
    print(f"  Device: {device}")
    if config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
        print(f"  GPU count: {torch.cuda.device_count()}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    
    # Create scheduler
    total_steps = len(train_loader) * config['num_epochs'] // config['gradient_accumulation_steps']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Create criterion
    criterion = huber_smape_loss
    
    # Initialize training state
    start_epoch = 0
    best_val_smape = float('inf')
    patience_counter = 0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'val_r2': [],
        'val_smape': []
    }
    
    # Resume from checkpoint if provided
    if resume_from and Path(resume_from).exists():
        start_epoch, training_history = load_checkpoint(
            resume_from, model, optimizer, scheduler
        )
        best_val_smape = min(training_history['val_smape']) if training_history['val_smape'] else float('inf')
        print(f"Resuming training from epoch {start_epoch + 1}")
    
    print(f"\nStarting fine-tuning for {config['num_epochs']} epochs...")
    print(f"Total steps: {total_steps:,}")
    print(f"Gradient accumulation steps: {config['gradient_accumulation_steps']}")
    print(f"Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"Warmup steps: {config['warmup_steps']}")
    print(f"Validation sample size: {config.get('val_sample_size', 'Full dataset')}")
    print(f"\nNote: Learning rate starts at 0 during warmup phase (first {config['warmup_steps']} steps)")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("=" * 80)
        
        # Train with comprehensive monitoring
        train_loss, train_huber, train_smape = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler,
            config['gradient_accumulation_steps'],
            print_interval=config.get('print_interval', 50),
            val_dataloader=val_loader,
            val_interval=config.get('val_interval', 200)
        )
        
        # Validate (with optional sampling for faster evaluation)
        val_sample_size = config.get('val_sample_size', None)
        val_loss, val_mae, val_mse, val_rmse, val_r2, val_smape = validate_epoch(
            model, val_loader, criterion, device, val_sample_size
        )
        
        # Scheduler is updated after each gradient step, not per epoch
        
        # Record history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_mae'].append(val_mae)
        training_history['val_rmse'].append(val_rmse)
        training_history['val_r2'].append(val_r2)
        training_history['val_smape'].append(val_smape)
        
        # Calculate additional metrics
        current_lr = scheduler.get_last_lr()[0]
        
        # Print comprehensive epoch summary
        print(f"\n📊 EPOCH {epoch+1} SUMMARY")
        print("-" * 50)
        print(f"TRAINING METRICS:")
        print(f"  Total Loss:     {train_loss:.4f}")
        print(f"  Huber Loss:     {train_huber:.4f}")
        print(f"  SMAPE Loss:     {train_smape:.2f}%")
        print(f"  Learning Rate:  {current_lr:.2e}")
        print(f"")
        print(f"VALIDATION METRICS:")
        print(f"  Val Loss:       {val_loss:.4f}")
        print(f"  Val MAE:        ${val_mae:.2f}")
        print(f"  Val RMSE:       ${val_rmse:.2f}")
        print(f"  Val R²:         {val_r2:.4f}")
        print(f"  Val SMAPE:      {val_smape:.2f}%")
        
        # Check for best model and early stopping
        is_best = val_smape < best_val_smape
        if is_best:
            best_val_smape = val_smape
            patience_counter = 0
            print(f"")
            print(f"🎉 NEW BEST MODEL!")
            print(f"  Best Val SMAPE: {val_smape:.2f}%")
        else:
            patience_counter += 1
            print(f"")
            print(f"⏳ PATIENCE: {patience_counter}/{config['patience']}")
        
        # Save checkpoint every epoch
        save_checkpoint(
            model, optimizer, scheduler, epoch, training_history, config, 
            output_dir, is_best=is_best
        )
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"")
            print(f"🛑 EARLY STOPPING TRIGGERED!")
            break
    
    training_time = time.time() - start_time
    print(f"\nFine-tuning completed in {training_time/60:.1f} minutes")
    print(f"Best validation SMAPE: {best_val_smape:.2f}%")
    
    # Load best model for final return
    best_checkpoint_path = output_dir / "best_model.pt"
    if best_checkpoint_path.exists():
        if hasattr(model, 'module'):
            # DataParallel wrapped model
            model.module.load_state_dict(torch.load(best_checkpoint_path, map_location='cpu')['model_state_dict'])
        else:
            # Regular model
            model.load_state_dict(torch.load(best_checkpoint_path, map_location='cpu')['model_state_dict'])
        print("✓ Loaded best model for final return")
    
    return model, training_history

def save_model(model, training_history, config, output_dir):
    """Save trained model and metadata"""
    print(f"\nSaving model to: {output_dir}")
    
    # Save model - handle DataParallel models
    model_path = output_dir / "qwen3_regression_finetuned.pt"
    if hasattr(model, 'module'):
        # DataParallel wrapped model - save the underlying module
        model.module.save_model(model_path)
    else:
        # Regular model
        model.save_model(model_path)
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save metadata
    metadata = {
        'model_type': 'qwen3_regression_finetuned',
        'model_name': config['model_name'],
        'use_qlora': config['use_qlora'],
        'lora_r': config['lora_r'],
        'lora_alpha': config['lora_alpha'],
        'regression_hidden_dims': config['regression_hidden_dims'],
        'dropout_rate': config['dropout_rate'],
        'use_log_price': config['use_log_price'],
        'price_offset': config['price_offset'],
        'best_val_smape': min(training_history['val_smape']),
        'num_epochs_trained': len(training_history['train_loss']),
        'training_config': config
    }
    
    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Model saved successfully")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen3-0.6B with regression head')
    parser.add_argument('--train_processed_path', type=str, default=str(TRAIN_PROCESSED),
                        help='Path to processed training data')
    parser.add_argument('--output_dir', type=str, default=str(MODELS_DIR),
                        help='Output directory for trained model')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME,
                        help='Model name to use')
    parser.add_argument('--device', type=str, default=DEVICE,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=PATIENCE,
                        help='Early stopping patience')
    parser.add_argument('--val_ratio', type=float, default=VAL_SPLIT_RATIO,
                        help='Validation split ratio')
    parser.add_argument('--use_qlora', action='store_true', default=USE_QLORA,
                        help='Use QLoRA for efficient fine-tuning')
    parser.add_argument('--lora_r', type=int, default=LORA_R,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=LORA_ALPHA,
                        help='LoRA alpha')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--val_sample_size', type=int, default=None,
                        help='Number of validation samples to use (None for full dataset)')
    parser.add_argument('--print_interval', type=int, default=50,
                        help='Print training loss every N batches')
    parser.add_argument('--val_interval', type=int, default=200,
                        help='Run validation every N batches during training')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                        help='Use multiple GPUs for training (DataParallel)')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Check for multi-GPU setup
    if args.use_multi_gpu and device == "cuda":
        if torch.cuda.device_count() < 2:
            print(f"Warning: Only {torch.cuda.device_count()} GPU(s) available, disabling multi-GPU training")
            args.use_multi_gpu = False
        else:
            print(f"Multi-GPU training enabled with {torch.cuda.device_count()} GPUs")
    
    # Load data
    texts, prices, values, units, unit_to_idx, idx_to_unit = load_data(args.train_processed_path)
    
    # Training configuration
    config = {
        'model_name': args.model_name,
        'use_qlora': args.use_qlora,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'regression_hidden_dims': REGRESSION_HIDDEN_DIMS,
        'dropout_rate': DROPOUT_RATE,
        'use_log_price': USE_LOG_PRICE,
        'price_offset': PRICE_OFFSET,
        'max_length': MAX_LENGTH,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'patience': args.patience,
        'val_split_ratio': args.val_ratio,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
        'max_grad_norm': MAX_GRAD_NORM,
        'warmup_steps': WARMUP_STEPS,
        'device': device,
        'num_workers': NUM_WORKERS,
        'random_seed': RANDOM_SEED,
        'val_sample_size': args.val_sample_size,
        'print_interval': args.print_interval,
        'val_interval': args.val_interval,
        'use_multi_gpu': args.use_multi_gpu
    }
    
    # Create model
    print("\nCreating model...")
    unit_vocab_size = len(unit_to_idx)
    model, tokenizer = create_model_for_training(
        model_name=config['model_name'],
        regression_hidden_dims=config['regression_hidden_dims'],
        dropout_rate=config['dropout_rate'],
        use_qlora=config['use_qlora'],
        lora_r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=LORA_DROPOUT,
        unit_vocab_size=unit_vocab_size,
        unit_embedding_dim=2
    )
    
    # Train model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model, training_history = train_model(
        model, tokenizer, texts, prices, values, units, config, output_dir, args.resume_from
    )
    
    # Save final model
    save_model(model, training_history, config, output_dir)
    
    print("\n" + "="*80)
    print("FINE-TUNING COMPLETE!")
    print("="*80)
    print(f"Model saved to: {output_dir}")
    print(f"Best validation SMAPE: {min(training_history['val_smape']):.2f}%")
    print("Next step: Run evaluate_finetuned.py or predict_finetuned.py")

if __name__ == "__main__":
    main()
