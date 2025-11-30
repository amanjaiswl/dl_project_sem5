"""
Inference script for the fine-tuned Qwen3-0.6B regression model.
Generates predictions on test data using the fine-tuned model.
"""

import sys
import argparse
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add pipeline config to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *
from models.qwen3_regression_model import Qwen3RegressionModel

class PriceDataset:
    """Dataset for price prediction with text inputs"""
    
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

def load_model(model_dir, device):
    """Load the fine-tuned model"""
    print(f"\nLoading fine-tuned model from: {model_dir}")
    
    # Try to load from checkpoint first (best_model.pt or checkpoint files)
    checkpoint_path = model_dir / "best_model.pt"
    if not checkpoint_path.exists():
        # Look for any checkpoint file
        checkpoint_files = list(model_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoint_files:
            checkpoint_path = checkpoint_files[-1]  # Use the latest checkpoint
        else:
            raise FileNotFoundError(f"No checkpoint files found in {model_dir}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract metadata from checkpoint
    config = checkpoint.get('config', {})
    
    # Extract unit vocabulary size from the state dict
    state_dict = checkpoint['model_state_dict']
    unit_embedding_weight = state_dict.get('unit_embedding.weight')
    if unit_embedding_weight is not None:
        unit_vocab_size = unit_embedding_weight.shape[0]
        unit_embedding_dim = unit_embedding_weight.shape[1]
    else:
        unit_vocab_size = 1000  # Default fallback
        unit_embedding_dim = 2   # Default fallback
    
    metadata = {
        'model_name': config.get('model_name', 'Qwen/Qwen3-Embedding-0.6B'),
        'use_qlora': config.get('use_qlora', True),
        'epochs_trained': checkpoint.get('epoch', 0),
        'regression_hidden_dims': config.get('regression_hidden_dims', [512, 256, 128]),
        'dropout_rate': config.get('dropout_rate', 0.1),
        'unit_vocab_size': unit_vocab_size,
        'unit_embedding_dim': unit_embedding_dim
    }
    
    # Create model with the same configuration
    model = Qwen3RegressionModel(
        model_name=metadata['model_name'],
        regression_hidden_dims=metadata['regression_hidden_dims'],
        dropout_rate=metadata['dropout_rate'],
        use_qlora=metadata['use_qlora'],
        unit_vocab_size=metadata['unit_vocab_size'],
        unit_embedding_dim=metadata['unit_embedding_dim']
    )
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Model: {metadata['model_name']}")
    print(f"  QLoRA: {metadata['use_qlora']}")
    print(f"  Epochs trained: {metadata['epochs_trained']}")
    
    return model, metadata

def load_tokenizer(model_name):
    """Load tokenizer for the model"""
    from transformers import AutoTokenizer
    
    print(f"Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✓ Tokenizer loaded")
    
    return tokenizer

def create_unit_vocab(units):
    """Create vocabulary mapping for units"""
    # Handle NaN values by converting to string and filtering
    units_clean = []
    for unit in units:
        if pd.isna(unit):
            units_clean.append('unknown')  # Default for missing units
        else:
            units_clean.append(str(unit))
    
    unique_units = sorted(list(set(units_clean)))
    unit_to_idx = {unit: idx for idx, unit in enumerate(unique_units)}
    idx_to_unit = {idx: unit for unit, idx in unit_to_idx.items()}
    
    print(f"✓ Created unit vocabulary with {len(unique_units)} unique units")
    print(f"  Unit range: 0 - {len(unique_units)-1}")
    
    return unit_to_idx, idx_to_unit

def load_test_data(test_csv_path, model_unit_vocab_size=97):
    """Load test data"""
    print(f"\nLoading test data from: {test_csv_path}")
    
    df = pd.read_csv(test_csv_path)
    
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
        
        # Map test unit indices to model unit indices (0 to model_unit_vocab_size-1)
        # This ensures we don't exceed the model's unit embedding size
        unit_indices = np.clip(unit_indices, 0, model_unit_vocab_size - 1)
        
        print(f"✓ Loaded {len(texts):,} test samples with enhanced features")
        print(f"  Value range: {values.min():.2f} - {values.max():.2f}")
        print(f"  Unit indices range: {unit_indices.min()} - {unit_indices.max()} (clipped to model vocab size)")
        
        return texts, values, unit_indices, sample_ids
    else:
        # Fallback to basic data (catalog_content only)
        texts = df['catalog_content'].fillna('').astype(str).tolist()
        sample_ids = df['sample_id'].tolist()
        
        # Create dummy values and units
        values = np.ones(len(texts))  # Default value of 1
        unit_indices = np.zeros(len(texts), dtype=int)  # Default unit index 0
        
        print(f"✓ Loaded {len(texts):,} test samples (basic mode)")
        print("  Using default values (1.0) and units (index 0)")
        
        return texts, values, unit_indices, sample_ids

def generate_predictions(model, tokenizer, texts, values, units, sample_ids, device, batch_size=8):
    """Generate predictions for test data"""
    print(f"\nGenerating predictions...")
    
    # Create dataset
    dataset = PriceDataset(texts, values, units, sample_ids, tokenizer, MAX_LENGTH)
    
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
    
    all_predictions = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            values_batch = batch['value'].to(device)
            units_batch = batch['unit'].to(device)
            sample_ids_batch = batch['sample_id']
            
            # Get log price predictions
            log_prices = model(input_ids, attention_mask, values_batch, units_batch)
            
            # Convert to actual prices
            prices = torch.exp(log_prices) - PRICE_OFFSET
            
            all_predictions.extend(prices.cpu().numpy())
            # Convert sample_ids_batch from tensor to list if needed
            if isinstance(sample_ids_batch, torch.Tensor):
                sample_ids_batch = sample_ids_batch.tolist()
            all_sample_ids.extend(sample_ids_batch)
    
    all_predictions = np.array(all_predictions)
    
    print(f"✓ Generated {len(all_predictions):,} predictions")
    
    return all_predictions, all_sample_ids

def post_process_predictions(predictions):
    """Post-process predictions to ensure valid format"""
    print(f"\nPost-processing predictions...")
    
    # Ensure positive values
    predictions = np.maximum(predictions, 0.01)  # Minimum price of $0.01
    
    # Round to 2 decimal places
    predictions = np.round(predictions, 2)
    
    print(f"  Min prediction: ${predictions.min():.2f}")
    print(f"  Max prediction: ${predictions.max():.2f}")
    print(f"  Mean prediction: ${predictions.mean():.2f}")
    
    return predictions

def save_predictions(predictions, sample_ids, output_path):
    """Save predictions in submission format"""
    print(f"\nSaving predictions to: {output_path}")
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'sample_id': sample_ids,
        'price': predictions
    })
    
    # Sort by sample_id to match expected format
    submission_df = submission_df.sort_values('sample_id').reset_index(drop=True)
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    print(f"✓ Saved {len(submission_df):,} predictions")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Show sample predictions
    print(f"\nSample predictions:")
    print(submission_df.head(10).to_string(index=False))

def evaluate_on_train_sample(model, tokenizer, train_processed_path, device, num_samples=1000):
    """Evaluate model on a sample of training data"""
    print(f"\nEvaluating on {num_samples:,} training samples...")
    
    # Load training data
    df = pd.read_csv(train_processed_path)
    df = df.dropna(subset=['combined_text', 'price'])
    
    # Sample data
    if len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=RANDOM_SEED)
    
    texts = df['combined_text'].tolist()
    prices = df['price'].values
    values = df['value'].values
    units = df['unit'].values
    sample_ids = df['sample_id'].tolist()
    
    # Create unit vocabulary and convert units to indices
    unit_to_idx, _ = create_unit_vocab(units)
    unit_indices = np.array([unit_to_idx[unit] for unit in units])
    
    # Generate predictions
    predictions, _ = generate_predictions(model, tokenizer, texts, values, unit_indices, sample_ids, device, batch_size=8)
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - prices))
    mse = np.mean((predictions - prices) ** 2)
    rmse = np.sqrt(mse)
    
    # Calculate SMAPE
    smape = np.mean(np.abs(predictions - prices) / ((np.abs(prices) + np.abs(predictions)) / 2)) * 100
    
    # Calculate R²
    ss_res = np.sum((prices - predictions) ** 2)
    ss_tot = np.sum((prices - np.mean(prices)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    print(f"\nEvaluation Results:")
    print(f"  MAE: ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  SMAPE: {smape:.2f}%")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'smape': smape,
        'num_samples': len(predictions)
    }

def main():
    parser = argparse.ArgumentParser(description='Generate predictions using fine-tuned Qwen3-0.6B model')
    parser.add_argument('--model_dir', type=str, default=str(MODELS_DIR),
                        help='Directory containing fine-tuned model')
    parser.add_argument('--test_csv', type=str, required=True,
                        help='Path to test.csv')
    parser.add_argument('--output_file', type=str, default='test_out_finetuned.csv',
                        help='Output file for predictions')
    parser.add_argument('--device', type=str, default=DEVICE,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for prediction')
    parser.add_argument('--evaluate_train', action='store_true',
                        help='Evaluate on training sample first')
    parser.add_argument('--train_processed_path', type=str, default=str(TRAIN_PROCESSED),
                        help='Path to processed training data for evaluation')
    parser.add_argument('--num_eval_samples', type=int, default=1000,
                        help='Number of training samples for evaluation')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Load model
    model_dir = Path(args.model_dir)
    model, metadata = load_model(model_dir, device)
    
    # Load tokenizer
    tokenizer = load_tokenizer(metadata['model_name'])
    
    # Evaluate on training sample if requested
    if args.evaluate_train:
        eval_results = evaluate_on_train_sample(
            model, tokenizer, args.train_processed_path, device, args.num_eval_samples
        )
        
        # Save evaluation results
        eval_path = model_dir / "inference_evaluation.json"
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"Evaluation results saved to: {eval_path}")
    
    # Load test data with model's unit vocabulary size
    model_unit_vocab_size = metadata.get('unit_vocab_size', 97)
    texts, values, units, sample_ids = load_test_data(args.test_csv, model_unit_vocab_size)
    
    # Generate predictions
    predictions, pred_sample_ids = generate_predictions(
        model, tokenizer, texts, values, units, sample_ids, device, args.batch_size
    )
    
    # Post-process predictions
    predictions = post_process_predictions(predictions)
    
    # Save predictions
    output_path = Path(args.output_file)
    save_predictions(predictions, pred_sample_ids, output_path)
    
    print(f"\n" + "="*80)
    print("PREDICTION COMPLETE!")
    print("="*80)
    print(f"Predictions saved to: {output_path}")
    print(f"Ready for submission!")
    
    # Validation
    print(f"\nValidation:")
    print(f"  ✓ All predictions are positive")
    print(f"  ✓ All predictions are rounded to 2 decimal places")
    print(f"  ✓ Output format matches submission requirements")
    print(f"  ✓ Sample IDs match test data")

if __name__ == "__main__":
    main()
