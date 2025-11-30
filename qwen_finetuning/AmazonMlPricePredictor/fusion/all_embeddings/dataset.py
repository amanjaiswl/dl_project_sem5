"""
Dataset class for loading all 4 embeddings: CLIP image, SigLIP image, SigLIP text, Qwen text
"""
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
from pathlib import Path


class FourEmbeddingDataset(Dataset):
    """Dataset that loads CLIP image, SigLIP image, SigLIP text, and Qwen text embeddings"""
    
    def __init__(self, clip_dir, siglip_dir, qwen_path, csv_path, 
                 indices=None, prices=None, sample_ids=None):
        """
        Args:
            clip_dir: Path to CLIP image embeddings directory
            siglip_dir: Path to SigLIP embeddings directory
            qwen_path: Path to Qwen embeddings pickle file
            csv_path: Path to CSV file
            indices: Optional array of indices to use (for train/val split)
            prices: Optional precomputed price array
            sample_ids: Optional precomputed sample_id array
        """
        self.clip_dir = Path(clip_dir)
        self.siglip_dir = Path(siglip_dir)
        self.qwen_path = Path(qwen_path)
        self.csv_path = Path(csv_path)
        
        # Load CLIP image embeddings
        print(f"Loading CLIP image from {self.clip_dir}...")
        self.clip_image_emb = np.load(self.clip_dir / 'image_embeddings.npy')
        self.clip_sample_ids = np.load(self.clip_dir / 'sample_ids.npy')
        print(f"  CLIP Image: {self.clip_image_emb.shape}")
        print(f"  Sample IDs: {len(self.clip_sample_ids)}")
        
        # Load SigLIP embeddings (image + text)
        print(f"Loading SigLIP from {self.siglip_dir}...")
        self.siglip_image_emb = np.load(self.siglip_dir / 'image_embeddings.npy')
        self.siglip_text_emb = np.load(self.siglip_dir / 'text_embeddings.npy')
        self.siglip_sample_ids = np.load(self.siglip_dir / 'image_sample_ids.npy')
        
        print(f"  SigLIP Image: {self.siglip_image_emb.shape}")
        print(f"  SigLIP Text: {self.siglip_text_emb.shape}")
        print(f"  Sample IDs: {len(self.siglip_sample_ids)}")
        
        # Load Qwen text embeddings
        print(f"Loading Qwen text from {self.qwen_path}...")
        with open(self.qwen_path, 'rb') as f:
            qwen_data = pickle.load(f)
        self.qwen_text_emb = qwen_data['embeddings']
        self.qwen_sample_ids = np.array([sid.item() for sid in qwen_data['sample_ids']])
        print(f"  Qwen Text: {self.qwen_text_emb.shape}")
        print(f"  Sample IDs: {len(self.qwen_sample_ids)}")
        
        # Verify all sample IDs match
        assert np.array_equal(self.clip_sample_ids, self.siglip_sample_ids), \
            "CLIP and SigLIP sample IDs don't match!"
        assert np.array_equal(self.siglip_sample_ids, self.qwen_sample_ids), \
            "SigLIP and Qwen sample IDs don't match!"
        print(f"  ✓ All embeddings are aligned")
        
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
            self.siglip_image_emb = self.siglip_image_emb[indices]
            self.siglip_text_emb = self.siglip_text_emb[indices]
            self.qwen_text_emb = self.qwen_text_emb[indices]
            self.prices = self.prices[indices]
            self.sample_ids = self.sample_ids[indices]
        
        print(f"Dataset size: {len(self)}")
    
    def __len__(self):
        return len(self.clip_image_emb)
    
    def __getitem__(self, idx):
        return {
            'clip_image': torch.from_numpy(self.clip_image_emb[idx]).float(),
            'siglip_image': torch.from_numpy(self.siglip_image_emb[idx]).float(),
            'siglip_text': torch.from_numpy(self.siglip_text_emb[idx]).float(),
            'qwen_text': torch.from_numpy(self.qwen_text_emb[idx]).float(),
            'price': torch.tensor(self.prices[idx], dtype=torch.float32),
            'sample_id': self.sample_ids[idx]
        }


def create_data_splits(clip_dir, siglip_dir, qwen_path, csv_path, 
                       prices, sample_ids, train_idx, val_idx):
    """
    Create train and validation datasets from indices
    
    Args:
        clip_dir: Path to CLIP image embeddings
        siglip_dir: Path to SigLIP embeddings
        qwen_path: Path to Qwen embeddings
        csv_path: Path to CSV
        prices: Full price array
        sample_ids: Full sample_id array
        train_idx: Training indices
        val_idx: Validation indices
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    train_dataset = FourEmbeddingDataset(
        clip_dir=clip_dir,
        siglip_dir=siglip_dir,
        qwen_path=qwen_path,
        csv_path=csv_path,
        indices=train_idx,
        prices=prices,
        sample_ids=sample_ids
    )
    
    val_dataset = FourEmbeddingDataset(
        clip_dir=clip_dir,
        siglip_dir=siglip_dir,
        qwen_path=qwen_path,
        csv_path=csv_path,
        indices=val_idx,
        prices=prices,
        sample_ids=sample_ids
    )
    
    return train_dataset, val_dataset


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Test loading
    clip_dir = Path("/data/utk/amazon_ml_2025/embeddings_clip/train")
    siglip_dir = Path("/data/utk/amazon_ml_2025/embedding_cache")
    qwen_path = Path("/data/utk/amazon_ml_2025/embeddings_qwen/qwen3_embeddings_train.pkl")
    csv_path = Path("/data/utk/amazon_ml_2025/dataset/train.csv")
    
    dataset = FourEmbeddingDataset(clip_dir, siglip_dir, qwen_path, csv_path)
    
    print(f"\nDataset created successfully!")
    print(f"Total samples: {len(dataset)}")
    
    # Test getting a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  CLIP image shape: {sample['clip_image'].shape}")
    print(f"  SigLIP image shape: {sample['siglip_image'].shape}")
    print(f"  SigLIP text shape: {sample['siglip_text'].shape}")
    print(f"  Qwen text shape: {sample['qwen_text'].shape}")
    print(f"  Price: ${sample['price']:.2f}")
    print(f"  Sample ID: {sample['sample_id']}")
