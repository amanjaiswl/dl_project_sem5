"""
Script to extract embeddings from the Qwen3-0.6B model without the regression head.
Extracts 1027-dimensional embeddings (1024 text + 1 value + 2 unit) and saves them with sample_id.
"""

import sys
import argparse
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import pickle
from torch.nn import DataParallel

# Add pipeline config to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *
from models.qwen3_regression_model import Qwen3RegressionModel

class EmbeddingDataset:
    """Dataset for embedding extraction with text inputs"""
    
    def __init__(self, texts, values, units, sample_ids, tokenizer, max_length=512):
        self.texts = texts
        self.values = values
        self.units = units
        self.sample_ids = sample_ids
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        value = float(self.values[idx])
        unit = int(self.units[idx])
        sample_id = self.sample_ids[idx]
        
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
            'value': torch.tensor(value, dtype=torch.float32),
            'unit': torch.tensor(unit, dtype=torch.long),
            'sample_id': sample_id
        }

def load_model(model_dir, device, unit_vocab_size=97):
    """Load the fine-tuned model"""
    print(f"\nLoading model from: {model_dir}")
    
    # Load model metadata
    metadata_path = model_dir / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Model metadata: {metadata}")
    else:
        print("Warning: No model metadata found, using default config")
        metadata = {}
    
    # Use the provided unit_vocab_size or from metadata, default to 97 (from training)
    actual_unit_vocab_size = unit_vocab_size or metadata.get('unit_vocab_size', 97)
    print(f"Using unit vocabulary size: {actual_unit_vocab_size}")
    
    # Create model with same config as training
    model = Qwen3RegressionModel(
        model_name=metadata.get('model_name', MODEL_NAME),
        regression_hidden_dims=metadata.get('regression_hidden_dims', REGRESSION_HIDDEN_DIMS),
        dropout_rate=metadata.get('dropout_rate', DROPOUT_RATE),
        use_qlora=metadata.get('use_qlora', USE_QLORA),
        unit_vocab_size=actual_unit_vocab_size,
        unit_embedding_dim=metadata.get('unit_embedding_dim', 2)
    )
    
    # Load model weights - try different possible names
    possible_model_paths = [
        model_dir / "qwen3_regression_finetuned.pt",
        model_dir / "best_model.pt",
        model_dir / "model.pt"
    ]
    
    model_path = None
    for path in possible_model_paths:
        if path.exists():
            model_path = path
            break
    
    if model_path is not None:
        print(f"Loading model weights from: {model_path}")
        
        # Try loading with weights_only=False first (for older checkpoints)
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"Failed to load with weights_only=False: {e}")
            print("Trying with weights_only=True...")
            try:
                # Add safe globals for numpy objects
                torch.serialization.add_safe_globals([np._core.multiarray.scalar])
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            except Exception as e2:
                print(f"Failed to load with weights_only=True: {e2}")
                print("Falling back to weights_only=False (less secure but should work)")
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("✓ Model weights loaded successfully")
    else:
        print("Warning: No fine-tuned weights found, using pretrained model only")
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        metadata.get('model_name', MODEL_NAME), 
        trust_remote_code=True
    )
    
    print("✓ Model and tokenizer loaded successfully")
    
    return model, tokenizer

def create_unit_vocab(units):
    """Create unit vocabulary from units list"""
    unique_units = sorted(list(set(units)))
    unit_to_idx = {unit: idx for idx, unit in enumerate(unique_units)}
    idx_to_unit = {idx: unit for unit, idx in unit_to_idx.items()}
    
    print(f"Created unit vocabulary with {len(unique_units)} unique units")
    
    return unit_to_idx, idx_to_unit

def load_data(data_path, data_type="train"):
    """Load and prepare data for embedding extraction"""
    print(f"\nLoading {data_type} data from: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Check if we have the enhanced columns
    if 'combined_text' in df.columns and 'value' in df.columns and 'unit' in df.columns:
        # Use enhanced data
        texts = df['combined_text'].fillna('').astype(str).tolist()
        values = df['value'].fillna(1.0).values  # Default value of 1.0 for missing values
        units = df['unit'].fillna('unknown').values  # Default unit for missing values
        sample_ids = df['sample_id'].tolist()
        
        # Create unit vocabulary and convert units to indices
        unit_to_idx, idx_to_unit = create_unit_vocab(units)
        unit_indices = np.array([unit_to_idx[str(unit)] for unit in units])
        
        print(f"✓ Loaded {len(texts):,} {data_type} samples with enhanced features")
        print(f"  Value range: {values.min():.2f} - {values.max():.2f}")
        print(f"  Unit indices range: {unit_indices.min()} - {unit_indices.max()}")
        
        return texts, values, unit_indices, sample_ids, unit_to_idx, idx_to_unit
    else:
        # Fallback to basic data (catalog_content only)
        texts = df['catalog_content'].fillna('').astype(str).tolist()
        sample_ids = df['sample_id'].tolist()
        
        # Create dummy values and units
        values = np.ones(len(texts))  # Default value of 1
        unit_indices = np.zeros(len(texts), dtype=int)  # Default unit index 0
        
        print(f"✓ Loaded {len(texts):,} {data_type} samples (basic mode)")
        print("  Using default values (1.0) and units (index 0)")
        
        # Create dummy unit vocabulary
        unit_to_idx = {'unknown': 0}
        idx_to_unit = {0: 'unknown'}
        
        return texts, values, unit_indices, sample_ids, unit_to_idx, idx_to_unit

def extract_embeddings(model, tokenizer, texts, values, units, sample_ids, device, batch_size=8, use_multi_gpu=False):
    """Extract 1027-dimensional embeddings from the model"""
    
    print(f"\nExtracting embeddings...")
    
    # Create dataset
    dataset = EmbeddingDataset(texts, values, units, sample_ids, tokenizer, MAX_LENGTH)
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Check for multiple GPUs and wrap with DataParallel
    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for embedding extraction")
        model = DataParallel(model)
        # Update effective batch size
        effective_batch_size = batch_size * torch.cuda.device_count()
        print(f"  Effective batch size: {effective_batch_size} (per-GPU: {batch_size})")
        
        # Adjust batch size if it might cause memory issues
        if effective_batch_size > 32:
            print(f"  Warning: Large effective batch size ({effective_batch_size}) might cause memory issues")
            print(f"  Consider reducing --batch_size if you encounter OOM errors")
    else:
        print(f"Using single GPU: {device}")
    
    all_embeddings = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            values_batch = batch['value'].to(device)
            units_batch = batch['unit'].to(device)
            sample_ids_batch = batch['sample_id']
            
            # Get embeddings (1027 dimensions: 1024 text + 1 value + 2 unit)
            # We need to manually construct the combined features since return_embeddings=True
            # only returns text embeddings, but we want the full 1027-dim feature vector
            
            # Get text embeddings from backbone
            if hasattr(model, 'module'):
                # DataParallel wrapped model
                outputs = model.module.backbone(input_ids=input_ids, attention_mask=attention_mask)
                unit_embeddings = model.module.unit_embedding(units_batch)
            else:
                # Regular model
                outputs = model.backbone(input_ids=input_ids, attention_mask=attention_mask)
                unit_embeddings = model.unit_embedding(units_batch)
            
            # Use mean pooling to get sentence embeddings
            if attention_mask is not None:
                # Mean pooling (exclude padding tokens)
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                text_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            else:
                # Use [CLS] token if no attention mask
                text_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Concatenate all features: text_embeddings + value + unit_embeddings
            combined_features = torch.cat([
                text_embeddings,  # Shape: [batch_size, 1024]
                values_batch.unsqueeze(-1),  # Shape: [batch_size, 1]
                unit_embeddings  # Shape: [batch_size, 2]
            ], dim=-1)  # Shape: [batch_size, 1027]
            
            all_embeddings.append(combined_features.cpu().numpy())
            all_sample_ids.extend(sample_ids_batch)
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    print(f"✓ Extracted {len(all_embeddings):,} embeddings")
    print(f"  Embedding shape: {all_embeddings.shape}")
    print(f"  Expected shape: (n_samples, 1027)")
    
    return all_embeddings, all_sample_ids

def save_embeddings(embeddings, sample_ids, output_path, data_type):
    """Save embeddings with sample_id mapping"""
    
    print(f"\nSaving embeddings to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy array with sample_id mapping
    embedding_data = {
        'embeddings': embeddings,
        'sample_ids': sample_ids,
        'data_type': data_type,
        'embedding_dim': embeddings.shape[1],
        'num_samples': len(embeddings)
    }
    
    # Save as pickle for easy loading
    with open(output_path, 'wb') as f:
        pickle.dump(embedding_data, f)
    
    # Also save as CSV for easy inspection (first few dimensions)
    csv_path = output_path.with_suffix('.csv')
    df_embeddings = pd.DataFrame(embeddings)
    df_embeddings['sample_id'] = sample_ids
    df_embeddings.to_csv(csv_path, index=False)
    
    print(f"✓ Embeddings saved successfully")
    print(f"  Pickle file: {output_path}")
    print(f"  CSV file: {csv_path}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Number of samples: {len(embeddings)}")

def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from Qwen3-0.6B model')
    parser.add_argument('--data_type', type=str, choices=['train', 'test'], required=True,
                        help='Type of data to process (train or test)')
    parser.add_argument('--model_dir', type=str, default=str(MODELS_DIR),
                        help='Directory containing the fine-tuned model')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data CSV file (default: auto-detect from data_type)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path for embeddings (default: auto-generate)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for embedding extraction')
    parser.add_argument('--device', type=str, default=DEVICE,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                        help='Use multiple GPUs for embedding extraction (DataParallel)')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
        args.use_multi_gpu = False
    
    # Check for multi-GPU setup
    if args.use_multi_gpu and device == "cuda":
        if torch.cuda.device_count() < 2:
            print(f"Warning: Only {torch.cuda.device_count()} GPU(s) available, disabling multi-GPU extraction")
            args.use_multi_gpu = False
        else:
            print(f"Multi-GPU extraction enabled with {torch.cuda.device_count()} GPUs")
    
    # Set default data path based on data_type
    if args.data_path is None:
        if args.data_type == 'train':
            args.data_path = str(TRAIN_PROCESSED)
        else:  # test
            args.data_path = str(TEST_PROCESSED)
    
    # Set default output path
    if args.output_path is None:
        output_filename = f"qwen3_embeddings_{args.data_type}.pkl"
        args.output_path = str(OUTPUTS_DIR / output_filename)
    
    print("="*80)
    print("QWEN3 EMBEDDING EXTRACTION")
    print("="*80)
    print(f"Data type: {args.data_type}")
    print(f"Data path: {args.data_path}")
    print(f"Model dir: {args.model_dir}")
    print(f"Output path: {args.output_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    if args.use_multi_gpu and device == "cuda":
        print(f"Multi-GPU: {torch.cuda.device_count()} GPUs")
    else:
        print("Multi-GPU: Disabled")
    
    # Check if data file exists
    if not Path(args.data_path).exists():
        print(f"Error: Data file not found: {args.data_path}")
        return
    
    # Check if model directory exists
    if not Path(args.model_dir).exists():
        print(f"Error: Model directory not found: {args.model_dir}")
        return
    
    try:
        # Load data first to get unit vocabulary
        texts, values, units, sample_ids, unit_to_idx, idx_to_unit = load_data(
            args.data_path, args.data_type
        )
        
        # Use the model's trained unit vocabulary size (97) instead of data size
        # This ensures compatibility with the trained model
        model_unit_vocab_size = 97
        print(f"Using model's unit vocabulary size: {model_unit_vocab_size}")
        print(f"Data has {len(unit_to_idx)} unique units, will map to model's {model_unit_vocab_size} units")
        
        # Map data units to model's unit vocabulary (0 to model_unit_vocab_size-1)
        # This ensures we don't exceed the model's unit embedding size
        units_mapped = np.clip(units, 0, model_unit_vocab_size - 1)
        print(f"Unit indices after mapping: {units_mapped.min()} - {units_mapped.max()}")
        
        # Load model with the trained unit vocabulary size
        model, tokenizer = load_model(Path(args.model_dir), device, model_unit_vocab_size)
        
        # Extract embeddings using mapped units
        embeddings, sample_ids = extract_embeddings(
            model, tokenizer, texts, values, units_mapped, sample_ids, 
            device, args.batch_size, args.use_multi_gpu
        )
        
        # Save embeddings
        save_embeddings(embeddings, sample_ids, Path(args.output_path), args.data_type)
        
        print("\n" + "="*80)
        print("EMBEDDING EXTRACTION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Extracted {len(embeddings):,} embeddings of dimension {embeddings.shape[1]}")
        print(f"Saved to: {args.output_path}")
        
    except Exception as e:
        print(f"\nError during embedding extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
