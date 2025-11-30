"""
Evaluate 4-embedding fusion model by price bins and ranges
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from .config import Config
from .dataset import FourEmbeddingDataset
from .model import FourBranchFusionModel, AdvancedMoEFusionModel
from ..utils import load_checkpoint, compute_smape, compute_metrics


def create_price_bins(prices):
    """Create price bins for analysis"""
    bins = [0, 5, 10, 15, 20, 30, 50, 100, 200, 500, np.inf]
    labels = ['$0-5', '$5-10', '$10-15', '$15-20', '$20-30', 
              '$30-50', '$50-100', '$100-200', '$200-500', '$500+']
    return pd.cut(prices, bins=bins, labels=labels, include_lowest=True)


def create_price_ranges(prices):
    """Create broader price ranges"""
    bins = [0, 10, 30, 100, 300, np.inf]
    labels = ['Very Low     ($  0-$  10)', 
              'Low          ($ 10-$  30)',
              'Medium       ($ 30-$ 100)',
              'High         ($100-$ 300)',
              'Very High    ($300-$10000)']
    return pd.cut(prices, bins=bins, labels=labels, include_lowest=True)


def main():
    config = Config()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("4-EMBEDDING FUSION MODEL EVALUATION")
    print("="*60)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(config.CSV_PATH)
    print(f"Loaded {len(df)} samples")
    print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    
    # Create dataset
    print("Creating dataset...")
    dataset = FourEmbeddingDataset(
        clip_dir=config.CLIP_EMBEDDINGS_DIR,
        siglip_dir=config.SIGLIP_EMBEDDINGS_DIR,
        qwen_path=config.QWEN_EMBEDDINGS_PATH,
        csv_path=config.CSV_PATH
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Load model
    print("Loading model...")
    if config.USE_ADVANCED_MODEL:
        model = AdvancedMoEFusionModel(config).to(device)
        print(f"Using Advanced MoE model (experts={config.NUM_EXPERTS})")
    else:
        model = FourBranchFusionModel(config).to(device)
        print(f"Using Base Fusion model")
    
    checkpoint_path = config.CHECKPOINT_DIR / 'best_model.pt'
    epoch, metrics = load_checkpoint(checkpoint_path, model)
    print(f"Loaded checkpoint (epoch {epoch+1}, val_smape={metrics['smape']:.2f}%)")
    
    # Make predictions
    print("Making predictions on full dataset...")
    model.eval()
    use_advanced = config.USE_ADVANCED_MODEL
    
    all_predictions = []
    all_prices = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            clip_image = batch['clip_image'].to(device)
            siglip_image = batch['siglip_image'].to(device)
            siglip_text = batch['siglip_text'].to(device)
            qwen_text = batch['qwen_text'].to(device)
            prices = batch['price']
            sample_ids = batch['sample_id']
            
            # Get predictions
            model_output = model(clip_image, siglip_image, siglip_text, qwen_text)
            predictions = model_output[0] if use_advanced else model_output
            
            # Convert to original scale
            if config.PREDICT_LOG:
                pred_prices = torch.expm1(predictions)
            else:
                pred_prices = predictions
            
            pred_prices = torch.clamp(pred_prices, min=0.0)
            
            all_predictions.extend(pred_prices.cpu().numpy())
            all_prices.extend(prices.numpy())
            all_sample_ids.extend(sample_ids)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'sample_id': all_sample_ids,
        'actual_price': all_prices,
        'predicted_price': all_predictions,
        'absolute_error': np.abs(np.array(all_prices) - np.array(all_predictions)),
        'smape_error': 200 * np.abs(np.array(all_prices) - np.array(all_predictions)) / 
                       (np.abs(np.array(all_prices)) + np.abs(np.array(all_predictions)) + 1e-8)
    })
    
    # Overall metrics
    print("\n" + "="*60)
    print("OVERALL METRICS")
    print("="*60)
    overall_metrics = compute_metrics(all_prices, all_predictions)
    print(f"SMAPE: {overall_metrics['smape']:.2f}%")
    print(f"MAE:   ${overall_metrics['mae']:.2f}")
    print(f"RMSE:  ${overall_metrics['rmse']:.2f}")
    print(f"R²:    {overall_metrics['r2']:.4f}")
    
    # Evaluation by price bins
    print("\n" + "="*60)
    print("EVALUATION BY PRICE BINS")
    print("="*60)
    
    results_df['price_bin'] = create_price_bins(results_df['actual_price'])
    
    print(f"\n{'Bin':<15} {'Count':>8} {'Pct':>6} {'SMAPE':>8} {'MAE':>10} {'RMSE':>10}")
    print("-" * 70)
    
    for bin_label in results_df['price_bin'].cat.categories:
        bin_data = results_df[results_df['price_bin'] == bin_label]
        if len(bin_data) > 0:
            bin_metrics = compute_metrics(
                bin_data['actual_price'].values,
                bin_data['predicted_price'].values
            )
            pct = 100 * len(bin_data) / len(results_df)
            print(f"{bin_label:<15} {len(bin_data):>8} {pct:>5.1f}% "
                  f"{bin_metrics['smape']:>7.2f}% ${bin_metrics['mae']:>9.2f} ${bin_metrics['rmse']:>9.2f}")
    
    # Evaluation by price ranges
    print("\n" + "="*60)
    print("EVALUATION BY PRICE RANGES")
    print("="*60)
    
    results_df['price_range'] = create_price_ranges(results_df['actual_price'])
    
    print(f"\n{'Range':<30} {'Count':>8} {'Pct':>6} {'SMAPE':>8} {'MAE':>10} {'RMSE':>10}")
    print("-" * 80)
    
    for range_label in results_df['price_range'].cat.categories:
        range_data = results_df[results_df['price_range'] == range_label]
        if len(range_data) > 0:
            range_metrics = compute_metrics(
                range_data['actual_price'].values,
                range_data['predicted_price'].values
            )
            pct = 100 * len(range_data) / len(results_df)
            print(f"{range_label:<30} {len(range_data):>8} {pct:>5.1f}% "
                  f"{range_metrics['smape']:>7.2f}% ${range_metrics['mae']:>9.2f} ${range_metrics['rmse']:>9.2f}")
    
    # Worst predictions
    print("\n" + "="*60)
    print("TOP 20 WORST PREDICTIONS (by SMAPE)")
    print("="*60)
    worst = results_df.nlargest(20, 'smape_error')
    print(f"\n{'Sample ID':>12} {'Actual':>10} {'Predicted':>10} {'SMAPE':>10} {'Abs Error':>10}")
    print("-" * 60)
    for _, row in worst.iterrows():
        print(f"{row['sample_id']:>12} ${row['actual_price']:>9.2f} ${row['predicted_price']:>9.2f} "
              f"{row['smape_error']:>9.2f}% ${row['absolute_error']:>9.2f}")
    
    # Best predictions
    print("\n" + "="*60)
    print("TOP 20 BEST PREDICTIONS (by SMAPE)")
    print("="*60)
    best = results_df.nsmallest(20, 'smape_error')
    print(f"\n{'Sample ID':>12} {'Actual':>10} {'Predicted':>10} {'SMAPE':>10} {'Abs Error':>10}")
    print("-" * 60)
    for _, row in best.iterrows():
        print(f"{row['sample_id']:>12} ${row['actual_price']:>9.2f} ${row['predicted_price']:>9.2f} "
              f"{row['smape_error']:>9.2f}% ${row['absolute_error']:>9.2f}")
    
    # Save detailed results
    output_path = config.OUTPUT_DIR / 'evaluation_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved detailed results to {output_path}")


if __name__ == "__main__":
    main()
