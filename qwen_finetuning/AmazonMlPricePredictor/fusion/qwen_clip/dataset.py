"""
Dataset class for loading CLIP image + Qwen text embeddings
"""
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
from pathlib import Path


class QwenClipDataset(Dataset):
    """Dataset that loads CLIP image embeddings and Qwen text embeddings"""
    
    def __init__(self, clip_embeddings_dir, qwen_embeddings_path, csv_path, 
                 indices=None, prices=None, sample_ids=None):
        """
        Args:
            clip_embeddings_dir: Path to CLIP embeddings directory
            qwen_embeddings_path: Path to Qwen embeddings pickle file
            csv_path: Path to CSV file
            indices: Optional array of indices to use (for train/val split)
            prices: Optional precomputed price array
            sample_ids: Optional precomputed sample_id array
        """
        self.clip_dir = Path(clip_embeddings_dir)
        self.qwen_path = Path(qwen_embeddings_path)
        self.csv_path = Path(csv_path)
        
        # Load CLIP embeddings
        print(f"Loading CLIP embeddings from {self.clip_dir}...")
        self.clip_image_emb = np.load(self.clip_dir / 'image_embeddings.npy')
        self.clip_sample_ids = np.load(self.clip_dir / 'sample_ids.npy')
        print(f"  CLIP image: {self.clip_image_emb.shape}")
        print(f"  CLIP sample IDs: {len(self.clip_sample_ids)}")
        
        # Load Qwen embeddings
        print(f"Loading Qwen embeddings from {self.qwen_path}...")
        with open(self.qwen_path, 'rb') as f:
            qwen_data = pickle.load(f)
        self.qwen_text_emb = qwen_data['embeddings']  # (N, 1027)
        self.qwen_sample_ids = [sid.item() for sid in qwen_data['sample_ids']]
        print(f"  Qwen text: {self.qwen_text_emb.shape}")
        print(f"  Qwen sample IDs: {len(self.qwen_sample_ids)}")
        
        # Verify alignment
        assert len(self.clip_sample_ids) == len(self.qwen_sample_ids), \
            "CLIP and Qwen have different number of samples"
        assert np.array_equal(self.clip_sample_ids, self.qwen_sample_ids), \
            "CLIP and Qwen sample IDs don't match!"
        print(f"  ✓ Embeddings are aligned")
        
        # Use provided prices or load from CSV
        if prices is not None and sample_ids is not None:
            self.prices = prices
            self.sample_ids = sample_ids
        else:
            # Load from CSV
            df = pd.read_csv(csv_path)
            # Create mapping from sample_id to price
            id_to_price = dict(zip(df['sample_id'], df['price']))
            self.prices = np.array([id_to_price[sid] for sid in self.clip_sample_ids], dtype=np.float32)
            self.sample_ids = self.clip_sample_ids
        
        # Apply index subset if provided (for train/val split)
        if indices is not None:
            self.clip_image_emb = self.clip_image_emb[indices]
            self.qwen_text_emb = self.qwen_text_emb[indices]
            self.prices = self.prices[indices]
            self.sample_ids = self.sample_ids[indices]
        
        print(f"Dataset size: {len(self)}")
    
    def __len__(self):
        return len(self.clip_image_emb)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - image_emb: (768,) float32 - CLIP image
                - text_emb: (1027,) float32 - Qwen text (includes structural features)
                - price: float32 (target)
                - sample_id: int
        """
        return {
            'image_emb': torch.from_numpy(self.clip_image_emb[idx]).float(),
            'text_emb': torch.from_numpy(self.qwen_text_emb[idx]).float(),
            'price': torch.tensor(self.prices[idx], dtype=torch.float32),
            'sample_id': self.sample_ids[idx]
        }


def create_data_splits(clip_embeddings_dir, qwen_embeddings_path, csv_path, 
                       prices, sample_ids, train_idx, val_idx):
    """
    Create train and validation datasets from indices
    
    Args:
        clip_embeddings_dir: Path to CLIP embeddings
        qwen_embeddings_path: Path to Qwen embeddings
        csv_path: Path to CSV
        prices: Full price array
        sample_ids: Full sample_id array
        train_idx: Training indices
        val_idx: Validation indices
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    train_dataset = QwenClipDataset(
        clip_embeddings_dir=clip_embeddings_dir,
        qwen_embeddings_path=qwen_embeddings_path,
        csv_path=csv_path,
        indices=train_idx,
        prices=prices,
        sample_ids=sample_ids
    )
    
    val_dataset = QwenClipDataset(
        clip_embeddings_dir=clip_embeddings_dir,
        qwen_embeddings_path=qwen_embeddings_path,
        csv_path=csv_path,
        indices=val_idx,
        prices=prices,
        sample_ids=sample_ids
    )
    
    return train_dataset, val_dataset


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.append('/home/utkarsh/amazon_ml_2025/fusion_qwen')
    from config import Config
    
    config = Config()
    
    # Load CSV
    df = pd.read_csv(config.CSV_PATH)
    prices = df['price'].values
    sample_ids = df['sample_id'].values
    
    # Create dataset
    dataset = QwenClipDataset(
        clip_embeddings_dir=config.CLIP_EMBEDDINGS_DIR,
        qwen_embeddings_path=config.QWEN_EMBEDDINGS_PATH,
        csv_path=config.CSV_PATH,
        prices=prices,
        sample_ids=sample_ids
    )
    
    # Test loading
    sample = dataset[0]
    print("\nSample structure:")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} ({v.dtype})")
        else:
            print(f"  {k}: {v}")

