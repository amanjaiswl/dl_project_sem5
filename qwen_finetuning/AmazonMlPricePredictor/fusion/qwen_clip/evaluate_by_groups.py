"""
Evaluate Qwen-CLIP fusion model - breakdown by price bins
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

from .config import Config
from .dataset import QwenClipDataset
from .model import TwoBranchFusionModel, AdvancedMoEFusionModel
from ..utils import load_checkpoint, compute_smape, compute_metrics


def predict_all(model, dataloader, device, config):
    """Get predictions for entire dataset"""
    model.eval()
    use_advanced = config.USE_ADVANCED_MODEL
    all_preds = []
    all_true = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            image_emb = batch['image_emb'].to(device)
            text_emb = batch['text_emb'].to(device)
            prices = batch['price']
            sample_ids = batch['sample_id']
            
            # Get predictions
            model_output = model(image_emb, text_emb)
            predictions = model_output[0] if use_advanced else model_output
            
            # Convert to original scale
            if config.PREDICT_LOG:
                pred_prices = torch.expm1(predictions)
            else:
                pred_prices = predictions
            
            pred_prices = torch.clamp(pred_prices, min=0.0)
            
            all_preds.extend(pred_prices.cpu().numpy())
            all_true.extend(prices.numpy())
            all_sample_ids.extend(sample_ids.numpy())
    
    return np.array(all_preds), np.array(all_true), np.array(all_sample_ids)


def analyze_errors(df):
    """Analyze worst predictions"""
    df['abs_error'] = (df['predicted_price'] - df['price']).abs()
    df['pct_error'] = 100 * df['abs_error'] / (df['price'] + 1e-8)
    
    # Compute per-sample SMAPE
    numerator = np.abs(df['predicted_price'] - df['price'])
    denominator = (np.abs(df['price']) + np.abs(df['predicted_price'])) / 2
    df['smape'] = 200 * numerator / (denominator + 1e-8)
    
    return df


def main():
    config = Config()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("QWEN-CLIP FUSION MODEL EVALUATION")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(config.CSV_PATH)
    prices = df['price'].values.astype(np.float32)
    sample_ids = df['sample_id'].values
    
    print(f"Loaded {len(df)} samples")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = QwenClipDataset(
        clip_embeddings_dir=config.CLIP_EMBEDDINGS_DIR,
        qwen_embeddings_path=config.QWEN_EMBEDDINGS_PATH,
        csv_path=config.CSV_PATH,
        prices=prices,
        sample_ids=sample_ids
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    print("\nLoading model...")
    if config.USE_ADVANCED_MODEL:
        model = AdvancedMoEFusionModel(config).to(device)
        print(f"Using Advanced MoE model (experts={config.NUM_EXPERTS})")
    else:
        model = TwoBranchFusionModel(config).to(device)
        print(f"Using Base Two-Branch Fusion model")
    
    checkpoint_path = config.CHECKPOINT_DIR / 'best_model.pt'
    
    if not checkpoint_path.exists():
        print(f"Error: No checkpoint at {checkpoint_path}")
        return
    
    epoch, metrics = load_checkpoint(checkpoint_path, model)
    print(f"Loaded checkpoint (epoch {epoch+1}, val_smape={metrics['smape']:.2f}%)")
    
    # Get predictions
    print("\nMaking predictions on full dataset...")
    y_pred, y_true, pred_sample_ids = predict_all(model, dataloader, device, config)
    
    # Add predictions to dataframe
    df['predicted_price'] = y_pred
    
    # Overall metrics
    overall_metrics = compute_metrics(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"OVERALL METRICS")
    print(f"{'='*60}")
    print(f"SMAPE:  {overall_metrics['smape']:.2f}%")
    print(f"MAE:    ${overall_metrics['mae']:.2f}")
    print(f"RMSE:   ${overall_metrics['rmse']:.2f}")
    print(f"R²:     {overall_metrics['r2']:.4f}")
    
    # SMAPE by price bins
    print(f"\n{'='*60}")
    print("SMAPE BY PRICE BIN")
    print(f"{'='*60}")
    
    bins = [0, 5, 10, 15, 20, 30, 50, 100, 200, 500, 10000]
    bin_labels = ['$0-5', '$5-10', '$10-15', '$15-20', '$20-30', '$30-50', '$50-100', '$100-200', '$200-500', '$500+']
    
    df['price_bin'] = pd.cut(df['price'], bins=bins, labels=bin_labels, include_lowest=True)
    
    bin_results = []
    for bin_name in bin_labels:
        bin_df = df[df['price_bin'] == bin_name]
        if len(bin_df) > 0:
            smape = compute_smape(bin_df['price'].values, bin_df['predicted_price'].values)
            mae = (bin_df['predicted_price'] - bin_df['price']).abs().mean()
            rmse = np.sqrt(((bin_df['predicted_price'] - bin_df['price']) ** 2).mean())
            count = len(bin_df)
            mean_price = bin_df['price'].mean()
            mean_pred = bin_df['predicted_price'].mean()
            
            bin_results.append({
                'bin': bin_name,
                'smape': smape,
                'mae': mae,
                'rmse': rmse,
                'count': count,
                'mean_true': mean_price,
                'mean_pred': mean_pred
            })
    
    print(f"\n{'Bin':<12} {'SMAPE':>8} {'MAE':>8} {'RMSE':>9} {'Count':>6} {'True Avg':>9} {'Pred Avg':>9}")
    print("-" * 75)
    for result in bin_results:
        print(f"{result['bin']:<12} {result['smape']:>7.2f}% ${result['mae']:>6.2f} ${result['rmse']:>7.2f} "
              f"{result['count']:>6d} ${result['mean_true']:>8.2f} ${result['mean_pred']:>8.2f}")
    
    # Price range analysis (simplified)
    print(f"\n{'='*60}")
    print("SMAPE BY PRICE RANGE")
    print(f"{'='*60}")
    
    price_ranges = [
        ('Very Low', 0, 10),
        ('Low', 10, 30),
        ('Medium', 30, 100),
        ('High', 100, 300),
        ('Very High', 300, 10000)
    ]
    
    for name, low, high in price_ranges:
        mask = (df['price'] >= low) & (df['price'] < high)
        range_df = df[mask]
        
        if len(range_df) > 0:
            smape = compute_smape(range_df['price'].values, range_df['predicted_price'].values)
            mae = (range_df['predicted_price'] - range_df['price']).abs().mean()
            count = len(range_df)
            mean_price = range_df['price'].mean()
            
            print(f"{name:12s} (${low:>3d}-${high:>4d}): SMAPE={smape:6.2f}% | MAE=${mae:7.2f} | "
                  f"Count={count:5d} | Avg=${mean_price:7.2f}")
    
    # Error analysis
    print(f"\n{'='*60}")
    print("WORST PREDICTIONS (Top 20 by SMAPE)")
    print(f"{'='*60}")
    
    df = analyze_errors(df)
    worst = df.nlargest(20, 'smape')[['sample_id', 'price', 'predicted_price', 'abs_error', 'smape']]
    
    print(f"\n{'Sample ID':>10} {'True':>8} {'Pred':>8} {'Abs Err':>9} {'SMAPE':>8}")
    print("-" * 50)
    for _, row in worst.iterrows():
        print(f"{int(row['sample_id']):>10d} ${row['price']:>7.2f} ${row['predicted_price']:>7.2f} "
              f"${row['abs_error']:>8.2f} {row['smape']:>7.2f}%")
    
    # Best predictions
    print(f"\n{'='*60}")
    print("BEST PREDICTIONS (Top 20 by SMAPE)")
    print(f"{'='*60}")
    
    best = df.nsmallest(20, 'smape')[['sample_id', 'price', 'predicted_price', 'abs_error', 'smape']]
    
    print(f"\n{'Sample ID':>10} {'True':>8} {'Pred':>8} {'Abs Err':>9} {'SMAPE':>8}")
    print("-" * 50)
    for _, row in best.iterrows():
        print(f"{int(row['sample_id']):>10d} ${row['price']:>7.2f} ${row['predicted_price']:>7.2f} "
              f"${row['abs_error']:>8.2f} {row['smape']:>7.2f}%")
    
    # Prediction distribution
    print(f"\n{'='*60}")
    print("PREDICTION STATISTICS")
    print(f"{'='*60}")
    print(f"\nTrue prices:")
    print(f"  Mean:   ${df['price'].mean():.2f}")
    print(f"  Median: ${df['price'].median():.2f}")
    print(f"  Std:    ${df['price'].std():.2f}")
    print(f"  Min:    ${df['price'].min():.2f}")
    print(f"  Max:    ${df['price'].max():.2f}")
    
    print(f"\nPredicted prices:")
    print(f"  Mean:   ${df['predicted_price'].mean():.2f}")
    print(f"  Median: ${df['predicted_price'].median():.2f}")
    print(f"  Std:    ${df['predicted_price'].std():.2f}")
    print(f"  Min:    ${df['predicted_price'].min():.2f}")
    print(f"  Max:    ${df['predicted_price'].max():.2f}")
    
    print(f"\nErrors:")
    print(f"  Mean Abs Error:  ${df['abs_error'].mean():.2f}")
    print(f"  Median Abs Error: ${df['abs_error'].median():.2f}")
    print(f"  Mean % Error:    {df['pct_error'].mean():.2f}%")
    print(f"  Median % Error:  {df['pct_error'].median():.2f}%")
    
    # Save detailed results
    output_path = config.OUTPUT_DIR / 'evaluation_results.csv'
    df[['sample_id', 'price', 'predicted_price', 'abs_error', 'pct_error', 'smape', 'price_bin']].to_csv(
        output_path, index=False
    )
    print(f"\n✓ Saved detailed results to {output_path}")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()

