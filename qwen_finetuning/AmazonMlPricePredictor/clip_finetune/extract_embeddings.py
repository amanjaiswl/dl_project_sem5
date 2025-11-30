"""
Extract embeddings from fine-tuned CLIP model
Run this after training to generate embeddings for all samples
"""
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import ProductDataset
from model import CLIPFineTuner
from utils import extract_embeddings, save_embeddings


def main(split="train"):
    """
    Extract embeddings from fine-tuned model
    
    Args:
        split: "train" or "test"
    """
    config = Config()
    
    print("="*60)
    print(f"Extracting CLIP Embeddings - {split.upper()} set")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Checkpoint: {config.CHECKPOINT_DIR / 'best_model.pt'}")
    print("="*60)
    
    # Initialize fine-tuner
    print("\nLoading model...")
    finetuner = CLIPFineTuner(config)
    
    # Load best checkpoint
    checkpoint_path = config.CHECKPOINT_DIR / 'checkpoint_epoch_5.pt'
    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        epoch, val_loss = finetuner.load_checkpoint(checkpoint_path, strict=True)
        print(f"Loaded model from epoch {epoch+1} (val_loss={val_loss:.4f})")
    else:
        print("Warning: No checkpoint found, using base CLIP model")
    
    # Create dataset
    print(f"\nLoading {split} dataset...")
    dataset = ProductDataset(split=split, processor=finetuner.processor)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Extract embeddings
    print(f"\nExtracting embeddings...")
    embeddings = extract_embeddings(finetuner.model, dataloader, finetuner.device)
    
    # Save embeddings
    output_dir = config.OUTPUT_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)
    save_embeddings(embeddings, output_dir)
    
    print("\n" + "="*60)
    print("EMBEDDING EXTRACTION COMPLETE!")
    print("="*60)
    print(f"Embeddings saved to: {output_dir}/")
    print("\nYou can now use these embeddings to train a regression head")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    # Default to train, but allow "test" as argument
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    
    if not split.startswith("train") and split not in ["test", "train_aug_v1"]:
        print(f"Error: split must be 'train' or 'test', got '{split}'")
        sys.exit(1)
    
    main(split)

