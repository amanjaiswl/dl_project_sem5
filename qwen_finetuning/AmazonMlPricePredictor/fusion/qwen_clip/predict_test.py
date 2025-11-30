"""
Generate predictions on test set using trained Qwen-CLIP fusion model
"""
import sys
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from .config import Config
from .model import TwoBranchFusionModel, AdvancedMoEFusionModel
from ..utils import load_checkpoint


class TestDataset(Dataset):
    """Simple dataset for test predictions"""
    
    def __init__(self, image_emb, text_emb, sample_ids):
        self.image_emb = image_emb
        self.text_emb = text_emb
        self.sample_ids = sample_ids
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        return {
            'image_emb': torch.from_numpy(self.image_emb[idx]).float(),
            'text_emb': torch.from_numpy(self.text_emb[idx]).float(),
            'sample_id': self.sample_ids[idx]
        }


def main():
    config = Config()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("TEST SET INFERENCE - QWEN-CLIP FUSION")
    print("="*60)
    print(f"Device: {device}")
    
    # ========================================================================
    # Load Test Data
    # ========================================================================
    print("\n" + "="*60)
    print("LOADING TEST DATA")
    print("="*60)
    
    # Load test CSV
    test_csv_path = config.CSV_PATH.parent / "test.csv"
    test_df = pd.read_csv(test_csv_path)
    print(f"Loaded {len(test_df)} test samples")
    
    # Load CLIP test embeddings
    print("\nLoading CLIP test embeddings...")
    clip_test_dir = Path("/data/utk/amazon_ml_2025/embeddings_clip/test")
    clip_image = np.load(clip_test_dir / 'image_embeddings.npy')
    clip_ids = np.load(clip_test_dir / 'sample_ids.npy')
    print(f"  Image: {clip_image.shape}")
    print(f"  Sample IDs: {len(clip_ids)}")
    
    # Load Qwen test embeddings
    print("\nLoading Qwen test embeddings...")
    qwen_test_path = Path("/data/utk/amazon_ml_2025/embeddings_qwen/qwen3_embeddings_test.pkl")
    with open(qwen_test_path, 'rb') as f:
        qwen_data = pickle.load(f)
    qwen_text = qwen_data['embeddings']
    qwen_ids = np.array([sid.item() for sid in qwen_data['sample_ids']])
    print(f"  Text: {qwen_text.shape}")
    print(f"  Sample IDs: {len(qwen_ids)}")
    
    # Verify alignment
    assert len(clip_ids) == len(qwen_ids), "CLIP and Qwen have different number of samples"
    assert np.array_equal(clip_ids, qwen_ids), "CLIP and Qwen sample IDs don't match!"
    print("  ✓ Embeddings are aligned")
    
    # Create test dataset
    test_dataset = TestDataset(clip_image, qwen_text, clip_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # ========================================================================
    # Load Trained Model
    # ========================================================================
    print("\n" + "="*60)
    print("LOADING TRAINED MODEL")
    print("="*60)
    
    if config.USE_ADVANCED_MODEL:
        model = AdvancedMoEFusionModel(config).to(device)
        print(f"Using Advanced MoE model (experts={config.NUM_EXPERTS})")
    else:
        model = TwoBranchFusionModel(config).to(device)
        print(f"Using Base Two-Branch Fusion model")
    
    checkpoint_path = config.CHECKPOINT_DIR / 'best_model.pt'
    
    if not checkpoint_path.exists():
        print(f"Error: No checkpoint found at {checkpoint_path}")
        print("Train the model first with: python train.py")
        return
    
    epoch, metrics = load_checkpoint(checkpoint_path, model)
    print(f"Loaded checkpoint (epoch {epoch+1}, val_smape={metrics['smape']:.2f}%)")
    
    # ========================================================================
    # Generate Predictions
    # ========================================================================
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)
    
    model.eval()
    use_advanced = config.USE_ADVANCED_MODEL
    all_predictions = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            image_emb = batch['image_emb'].to(device)
            text_emb = batch['text_emb'].to(device)
            sample_ids = batch['sample_id']
            
            # Get predictions
            model_output = model(image_emb, text_emb)
            predictions = model_output[0] if use_advanced else model_output
            
            # Convert to original scale
            if config.PREDICT_LOG:
                pred_prices = torch.expm1(predictions)
            else:
                pred_prices = predictions
            
            # Ensure non-negative
            pred_prices = torch.clamp(pred_prices, min=0.0)
            
            all_predictions.extend(pred_prices.cpu().numpy())
            all_sample_ids.extend(sample_ids.numpy())
    
    # ========================================================================
    # Save Predictions
    # ========================================================================
    print("\n" + "="*60)
    print("SAVING PREDICTIONS")
    print("="*60)
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'sample_id': all_sample_ids,
        'price': all_predictions
    })
    
    # Sort by sample_id to match expected order
    output_df = output_df.sort_values('sample_id').reset_index(drop=True)
    
    # Save predictions
    output_path = config.OUTPUT_DIR / 'test_out.csv'
    output_df.to_csv(output_path, index=False)
    
    print(f"Output saved to: {output_path}")
    print(f"Total predictions: {len(output_df)}")
    
    print(f"\nPrediction statistics:")
    print(f"  Mean:   ${output_df['price'].mean():.2f}")
    print(f"  Median: ${output_df['price'].median():.2f}")
    print(f"  Std:    ${output_df['price'].std():.2f}")
    print(f"  Min:    ${output_df['price'].min():.2f}")
    print(f"  Max:    ${output_df['price'].max():.2f}")
    
    print(f"\nFirst 10 predictions:")
    print(output_df.head(10).to_string(index=False))
    
    print(f"\n{'='*60}")
    print("PREDICTIONS COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

