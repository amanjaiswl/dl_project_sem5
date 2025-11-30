"""
Evaluation script for the fine-tuned Qwen3-0.6B regression model.
Evaluates model performance on training data with comprehensive metrics.
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
    
    def __init__(self, texts, prices, values, units, sample_ids, tokenizer, max_length=512):
        self.texts = texts
        self.prices = prices
        self.values = values
        self.units = units
        self.sample_ids = sample_ids
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        price = self.prices[idx]
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
            'price': torch.tensor(price, dtype=torch.float32),
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
    unique_units = sorted(list(set(units)))
    unit_to_idx = {unit: idx for idx, unit in enumerate(unique_units)}
    idx_to_unit = {idx: unit for unit, idx in unit_to_idx.items()}
    
    print(f"✓ Created unit vocabulary with {len(unique_units)} unique units")
    print(f"  Unit range: 0 - {len(unique_units)-1}")
    
    return unit_to_idx, idx_to_unit

def load_training_data(train_processed_path, num_samples=None):
    """Load training data for evaluation"""
    print(f"\nLoading training data from: {train_processed_path}")
    
    df = pd.read_csv(train_processed_path)
    df = df.dropna(subset=['combined_text', 'price', 'value', 'unit'])
    
    # Sample data if requested
    if num_samples is not None and len(df) > num_samples:
        df = df.sample(n=num_samples) #, random_state=RANDOM_SEED)
        print(f"  Sampled {num_samples:,} samples for evaluation")
    
    texts = df['combined_text'].tolist()
    prices = df['price'].values
    values = df['value'].values
    units = df['unit'].values
    sample_ids = df['sample_id'].tolist()
    
    # Create unit vocabulary and convert units to indices
    unit_to_idx, idx_to_unit = create_unit_vocab(units)
    unit_indices = np.array([unit_to_idx[unit] for unit in units])
    
    print(f"✓ Loaded {len(texts):,} samples")
    print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    print(f"  Mean price: ${prices.mean():.2f}")
    print(f"  Value range: {values.min():.2f} - {values.max():.2f}")
    print(f"  Unit indices range: {unit_indices.min()} - {unit_indices.max()}")
    
    return texts, prices, values, unit_indices, sample_ids

def evaluate_model(model, tokenizer, texts, prices, values, units, sample_ids, device, batch_size=8):
    """Evaluate model on data"""
    print(f"\nEvaluating model...")
    
    # Create dataset
    dataset = PriceDataset(texts, prices, values, units, sample_ids, tokenizer, MAX_LENGTH)
    
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
    all_prices = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            prices_batch = batch['price'].to(device)
            values_batch = batch['value'].to(device)
            units_batch = batch['unit'].to(device)
            sample_ids_batch = batch['sample_id']
            
            # Get log price predictions
            log_prices = model(input_ids, attention_mask, values_batch, units_batch)
            
            # Convert to actual prices
            predictions = torch.exp(log_prices) - PRICE_OFFSET
            
            all_predictions.extend(predictions.cpu().numpy())
            all_prices.extend(prices_batch.cpu().numpy())
            all_sample_ids.extend(sample_ids_batch)
    
    all_predictions = np.array(all_predictions)
    all_prices = np.array(all_prices)
    
    print(f"✓ Generated {len(all_predictions):,} predictions")
    
    return all_predictions, all_prices, all_sample_ids

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # Calculate R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Calculate SMAPE
    smape = np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'smape': smape,
        'mape': mape
    }

def print_metrics(metrics):
    """Print evaluation metrics"""
    print(f"\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Mean Absolute Error (MAE):     ${metrics['mae']:.2f}")
    print(f"Root Mean Square Error (RMSE): ${metrics['rmse']:.2f}")
    print(f"R² Score:                      {metrics['r2']:.4f}")
    print(f"SMAPE:                         {metrics['smape']:.2f}%")
    print(f"MAPE:                          {metrics['mape']:.2f}%")
    print("="*60)

def show_sample_predictions(predictions, prices, sample_ids, num_samples=10):
    """Show sample predictions vs actual prices"""
    print(f"\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)
    
    # Get random samples
    indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)
    
    print(f"{'Sample ID':<12} {'Actual':<10} {'Predicted':<10} {'Error':<10} {'Error %':<10}")
    print("-" * 60)
    
    for idx in indices:
        actual = prices[idx]
        predicted = predictions[idx]
        error = predicted - actual
        error_pct = (error / actual) * 100 if actual != 0 else 0
        
        print(f"{sample_ids[idx]:<12} ${actual:<9.2f} ${predicted:<9.2f} ${error:<9.2f} {error_pct:<9.1f}%")

def analyze_errors(predictions, prices):
    """Analyze prediction errors"""
    errors = predictions - prices
    abs_errors = np.abs(errors)
    
    print(f"\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    print(f"Mean Error:           ${np.mean(errors):.2f}")
    print(f"Std Error:            ${np.std(errors):.2f}")
    print(f"Mean Absolute Error:  ${np.mean(abs_errors):.2f}")
    print(f"Median Absolute Error: ${np.median(abs_errors):.2f}")
    print(f"Max Absolute Error:   ${np.max(abs_errors):.2f}")
    
    # Error distribution
    print(f"\nError Distribution:")
    print(f"  < $10:    {np.sum(abs_errors < 10):,} ({np.sum(abs_errors < 10)/len(errors)*100:.1f}%)")
    print(f"  < $50:    {np.sum(abs_errors < 50):,} ({np.sum(abs_errors < 50)/len(errors)*100:.1f}%)")
    print(f"  < $100:   {np.sum(abs_errors < 100):,} ({np.sum(abs_errors < 100)/len(errors)*100:.1f}%)")
    print(f"  >= $100:  {np.sum(abs_errors >= 100):,} ({np.sum(abs_errors >= 100)/len(errors)*100:.1f}%)")

def analyze_price_ranges(predictions, prices):
    """Analyze performance across different price ranges"""
    print(f"\n" + "="*60)
    print("PERFORMANCE BY PRICE RANGE")
    print("="*60)
    
    # Define price ranges with more granular buckets
    ranges = [
        (0, 15, "Very Low ($0-$15)"),
        (15, 30, "Low ($15-$30)"),
        (30, 50, "Low-Medium ($30-$50)"),
        (50, 200, "Medium-Low ($50-$75)"),
        (200, 500, "High ($200-$300)"),
        (500, float('inf'), "Very High ($500+)"),
    ]
    
    for min_price, max_price, label in ranges:
        mask = (prices >= min_price) & (prices < max_price)
        if np.sum(mask) == 0:
            continue
            
        range_predictions = predictions[mask]
        range_prices = prices[mask]
        
        # Calculate SMAPE for this range
        smape = np.mean(np.abs(range_predictions - range_prices) / ((np.abs(range_prices) + np.abs(range_predictions)) / 2)) * 100
        mae = np.mean(np.abs(range_predictions - range_prices))
        
        print(f"{label:<20} {np.sum(mask):>6,} samples  SMAPE: {smape:>6.1f}%  MAE: ${mae:>6.1f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned Qwen3-0.6B model')
    parser.add_argument('--model_dir', type=str, default=str(MODELS_DIR),
                        help='Directory containing fine-tuned model')
    parser.add_argument('--train_processed_path', type=str, default=str(TRAIN_PROCESSED),
                        help='Path to processed training data')
    parser.add_argument('--device', type=str, default=DEVICE,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (None for all)')
    parser.add_argument('--show_samples', type=int, default=10,
                        help='Number of sample predictions to show')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUTS_DIR),
                        help='Output directory for evaluation results')
    
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
    
    # Load data
    texts, prices, values, units, sample_ids = load_training_data(args.train_processed_path, args.num_samples)
    
    # Evaluate model
    predictions, actual_prices, eval_sample_ids = evaluate_model(
        model, tokenizer, texts, prices, values, units, sample_ids, device, args.batch_size
    )
    
    # Calculate metrics
    metrics = calculate_metrics(actual_prices, predictions)
    print_metrics(metrics)
    
    # Show sample predictions
    show_sample_predictions(predictions, actual_prices, eval_sample_ids, args.show_samples)
    
    # Analyze errors
    analyze_errors(predictions, actual_prices)
    
    # Analyze performance by price range
    analyze_price_ranges(predictions, actual_prices)
    
    # Convert NumPy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Save results
    results = {
        'metrics': convert_numpy_types(metrics),
        'num_samples_evaluated': len(predictions),
        'model_metadata': convert_numpy_types(metadata),
        'evaluation_config': {
            'device': device,
            'batch_size': args.batch_size,
            'num_samples': args.num_samples
        }
    }
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "finetuned_evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"Results saved to: {results_path}")
    print(f"Key metric - SMAPE: {metrics['smape']:.2f}%")
    
    if metrics['smape'] < 25:
        print("🎉 Excellent performance! SMAPE < 25%")
    elif metrics['smape'] < 30:
        print("✅ Good performance! SMAPE < 30%")
    else:
        print("⚠️  Consider model improvements - SMAPE > 30%")

if __name__ == "__main__":
    main()
