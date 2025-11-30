"""
Main training script for CLIP fine-tuning
"""
import torch
from torch.utils.data import DataLoader, random_split

from config import Config
from dataset import ProductDataset
from model import CLIPFineTuner
from utils import save_training_history


def train():
    """Main training function"""
    config = Config()
    
    print("="*60)
    print("CLIP Fine-tuning for Product Price Prediction")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Unfrozen blocks: {config.NUM_UNFROZEN_BLOCKS}")
    print("="*60)
    
    # Initialize fine-tuner
    print("\nInitializing model...")
    finetuner = CLIPFineTuner(config)
    
    # Create dataset
    print("\nLoading dataset...")
    full_dataset = ProductDataset(split="train", processor=finetuner.processor)
    
    # Split into train/val
    total_size = len(full_dataset)
    val_size = int(total_size * config.VAL_SPLIT)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}")
    
    # Create dataloaders (temporarily disable multiprocessing to debug)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing to debug tensor issues
        pin_memory=False  # Disable pin_memory when num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing to debug tensor issues
        pin_memory=False  # Disable pin_memory when num_workers=0
    )
    
    # Training loop
    print("\nStarting training...")
    print(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE} epochs")
    print(f"Learning rate scheduler: reduce by {config.LR_SCHEDULER_FACTOR}x after {config.LR_SCHEDULER_PATIENCE} epochs")
    print("="*60)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    epochs_no_improve = 0
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"Current LR: {finetuner.optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}")
        
        # Train
        train_loss = finetuner.train_epoch(train_loader, epoch)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = finetuner.validate(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduler step
        finetuner.scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if config.SAVE_CHECKPOINTS:
                finetuner.save_checkpoint(epoch, val_loss, is_best=True)
                print(f"  ✓ Saved best checkpoint (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  ⚠ No improvement for {epochs_no_improve} epoch(s)")
        
        # Save epoch checkpoint
        if config.SAVE_CHECKPOINTS:
            finetuner.save_checkpoint(epoch, val_loss, is_best=False)
            print(f"  ✓ Saved epoch checkpoint")
        
        # Early stopping check
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n{'='*60}")
            print(f"EARLY STOPPING: No improvement for {config.EARLY_STOPPING_PATIENCE} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"{'='*60}")
            break
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    save_training_history(history, config.OUTPUT_DIR / 'training_history.json')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Epochs completed: {len(train_losses)}/{config.NUM_EPOCHS}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
        print(f"Stopped early due to no improvement")
    print(f"Checkpoints saved to: {config.CHECKPOINT_DIR}/")
    print("="*60)
    print("\nNext step: Run extract_embeddings.py to generate embeddings")


if __name__ == "__main__":
    train()

