"""
Generate predictions on test set using trained 4-embedding fusion model
"""
import sys
import numpy as np
import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from .config import Config
from .model import FourBranchFusionModel, AdvancedMoEFusionModel
from ..utils import load_checkpoint


class TestDataset(Dataset):
    """Dataset for test predictions with 4 embeddings"""
    
    def __init__(self, clip_image, siglip_image, siglip_text, qwen_text, sample_ids):
        self.clip_image = clip_image
        self.siglip_image = siglip_image
        self.siglip_text = siglip_text
        self.qwen_text = qwen_text
        self.sample_ids = sample_ids
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        return {
            'clip_image': torch.from_numpy(self.clip_image[idx]).float(),
            'siglip_image': torch.from_numpy(self.siglip_image[idx]).float(),
            'siglip_text': torch.from_numpy(self.siglip_text[idx]).float(),
            'qwen_text': torch.from_numpy(self.qwen_text[idx]).float(),
            'sample_id': self.sample_ids[idx]
        }


def main():
    config = Config()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("TEST SET INFERENCE - 4-EMBEDDING FUSION")
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
    
    # Test embedding paths
    clip_test_dir = Path("/data/utk/amazon_ml_2025/embeddings_clip/test")
    siglip_test_dir = Path("/data/utk/amazon_ml_2025/embedding_cache_test")
    qwen_test_path = Path("/data/utk/amazon_ml_2025/embeddings_qwen/qwen3_embeddings_test.pkl")
    
    # Check if test embeddings exist
    missing = []
    if not clip_test_dir.exists():
        missing.append(f"CLIP: {clip_test_dir}")
    if not siglip_test_dir.exists():
        missing.append(f"SigLIP: {siglip_test_dir}")
    if not qwen_test_path.exists():
        missing.append(f"Qwen: {qwen_test_path}")
    
    if missing:
        print(f"\nError: Test embeddings not found:")
        for path in missing:
            print(f"  - {path}")
        print("\nPlease generate test embeddings first using the same models")
        print("\nExpected CLIP files:")
        print("  - image_embeddings.npy")
        print("  - sample_ids.npy")
        print("\nExpected SigLIP files:")
        print("  - image_embeddings.npy")
        print("  - text_embeddings.npy")
        print("  - image_sample_ids.npy")
        print("\nExpected Qwen file:")
        print("  - qwen3_embeddings_test.pkl")
        return
    
    # Load CLIP test embeddings
    print("\nLoading CLIP test embeddings...")
    clip_image = np.load(clip_test_dir / 'image_embeddings.npy')
    clip_ids = np.load(clip_test_dir / 'sample_ids.npy')
    print(f"  CLIP Image: {clip_image.shape}")
    print(f"  Sample IDs: {len(clip_ids)}")
    
    # Load SigLIP test embeddings
    print("\nLoading SigLIP test embeddings...")
    siglip_image = np.load(siglip_test_dir / 'image_embeddings.npy')
    siglip_text = np.load(siglip_test_dir / 'text_embeddings.npy')
    siglip_ids = np.load(siglip_test_dir / 'image_sample_ids.npy')
    print(f"  SigLIP Image: {siglip_image.shape}")
    print(f"  SigLIP Text: {siglip_text.shape}")
    print(f"  Sample IDs: {len(siglip_ids)}")
    
    # Load Qwen test embeddings
    print("\nLoading Qwen test embeddings...")
    with open(qwen_test_path, 'rb') as f:
        qwen_data = pickle.load(f)
    qwen_text = qwen_data['embeddings']
    qwen_ids = np.array([sid.item() for sid in qwen_data['sample_ids']])
    print(f"  Qwen Text: {qwen_text.shape}")
    print(f"  Sample IDs: {len(qwen_ids)}")
    
    # Verify alignment
    if not (np.array_equal(clip_ids, siglip_ids) and 
            np.array_equal(siglip_ids, qwen_ids)):
        print("\nError: Sample IDs don't match across embeddings!")
        return
    
    print("\n✓ All embeddings are aligned")
    sample_ids = clip_ids
    
    # ========================================================================
    # Load Model
    # ========================================================================
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    
    if config.USE_ADVANCED_MODEL:
        model = AdvancedMoEFusionModel(config).to(device)
        print("Using Advanced MoE model")
    else:
        model = FourBranchFusionModel(config).to(device)
        print("Using Base Four-Branch Fusion model")
    
    checkpoint_path = config.CHECKPOINT_DIR / 'best_model.pt'
    epoch, metrics = load_checkpoint(checkpoint_path, model)
    print(f"Loaded checkpoint from epoch {epoch + 1}")
    print(f"Validation SMAPE: {metrics['smape']:.2f}%")
    
    # ========================================================================
    # Make Predictions
    # ========================================================================
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)
    
    # Create test dataset and dataloader
    test_dataset = TestDataset(clip_image, siglip_image, siglip_text, qwen_text, sample_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Generate predictions
    model.eval()
    use_advanced = config.USE_ADVANCED_MODEL
    
    all_predictions = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            clip_img = batch['clip_image'].to(device)
            siglip_img = batch['siglip_image'].to(device)
            siglip_txt = batch['siglip_text'].to(device)
            qwen_txt = batch['qwen_text'].to(device)
            batch_ids = batch['sample_id']
            
            # Get predictions
            model_output = model(clip_img, siglip_img, siglip_txt, qwen_txt)
            predictions = model_output[0] if use_advanced else model_output
            
            # Convert to original scale
            if config.PREDICT_LOG:
                pred_prices = torch.expm1(predictions)
            else:
                pred_prices = predictions
            
            # Clamp to non-negative
            pred_prices = torch.clamp(pred_prices, min=0.0)
            
            all_predictions.extend(pred_prices.cpu().numpy())
            all_sample_ids.extend(batch_ids)
    
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
    
    # Sort by sample_id
    output_df = output_df.sort_values('sample_id').reset_index(drop=True)
    
    # Save to CSV
    output_path = config.OUTPUT_DIR / 'test_out.csv'
    output_df.to_csv(output_path, index=False)
    
    print(f"✓ Saved predictions to {output_path}")
    print(f"  Total predictions: {len(output_df)}")
    print(f"  Price range: ${output_df['price'].min():.2f} - ${output_df['price'].max():.2f}")
    print(f"  Mean price: ${output_df['price'].mean():.2f}")
    print(f"  Median price: ${output_df['price'].median():.2f}")
    
    # Show sample predictions
    print("\nSample predictions:")
    print(output_df.head(10).to_string(index=False))
    
    print("\n✓ Prediction complete!")


if __name__ == "__main__":
    main()
