# Fusion Qwen-CLIP - Two-Branch Price Prediction

A two-branch neural network combining CLIP image embeddings with Qwen text embeddings (includes structural features) for price prediction.

## Overview

This model fuses two complementary embedding sources:
- **CLIP Image** (768-dim) - Visual features from product images
- **Qwen Text** (1027-dim) - Text embeddings (1024) + structural features (3)

Unlike the original fusion models that require manual feature engineering, this model leverages Qwen's pre-embedded structural information.

## Architecture

### Input Embeddings
- **CLIP Image**: 768-dimensional embeddings from product images
- **Qwen Text**: 1027-dimensional embeddings (1024 text + 3 structural features)

### Model Variants

#### Base Model: `TwoBranchFusionModel`
```
Image Branch:  768 → 512
Text Branch:   1027 → 512
Fusion:        Concatenate → 1024 dims
Regression:    1024 → 512 → 256 → 128 → 1
```

#### Advanced Model: `AdvancedMoEFusionModel`
- Same branch structure
- **Cross-Attention**: Self-attention on fused features
- **Gated MoE**: 3 expert networks with learned routing

### Key Features
- L2 normalization + learnable scaling on embeddings
- Log-price prediction for stability
- Weighted SMAPE loss (optimizes competition metric)
- Early stopping with patience

## 📁 Project Structure

```
fusion_qwen_clip/
├── config.py              # Configuration and hyperparameters
├── dataset.py             # Dataset loading (CLIP + Qwen embeddings)
├── model.py               # Neural network architectures
├── train.py               # Training loop
├── evaluate_by_groups.py  # Detailed evaluation by price bins
├── predict_test.py        # Generate test predictions
└── README.md              # This file
```

**Shared components** (in `fusion_common/`):
- `loss_functions.py` - Custom loss functions
- `utils.py` - Utility functions

## 🚀 Quick Start

### 1. Train Model

```bash
cd fusion_qwen_clip
python train.py
```

**Training details:**
- Loads 75,000 samples (CLIP image + Qwen text embeddings)
- Splits 80% train / 20% validation (stratified by price)
- Trains for up to 1000 epochs with early stopping (patience=15)
- Saves best model based on validation SMAPE
- Logs metrics every epoch: SMAPE, MAE, RMSE, R²

**Expected behavior:**
- Training stops early if validation SMAPE doesn't improve for 15 epochs
- Checkpoint saved every 100 epochs + best model
- Uses AdamW optimizer with ReduceLROnPlateau scheduler

**Expected output:**
```
Epoch 29/1000
Val SMAPE: 33.01%
Val MAE:   7.55
Val RMSE:  21.81
Val R²:    0.5673
LR:        0.000063
```

### 2. Evaluate Model

```bash
python evaluate_by_groups.py
```

**Evaluation provides:**
- Overall metrics (SMAPE, MAE, RMSE, R²)
- Performance breakdown by price bins ($0-5, $5-10, etc.)
- Performance breakdown by price ranges (Low, Medium, High)
- Top 20 worst predictions (highest SMAPE)
- Top 20 best predictions (lowest SMAPE)
- Detailed statistics and error analysis
- Saves results to `evaluation_results.csv`

### 3. Generate Test Predictions

```bash
python predict_test.py
```

**What it does:**
1. Loads test embeddings (CLIP image + Qwen text)
2. Loads trained model from checkpoint
3. Generates price predictions for all test samples
4. Converts from log-space to dollar amounts
5. Saves to `test_out.csv` in competition format

**Output format:**
```csv
sample_id,price
100179,12.45
100180,8.99
...
```

## 🔧 Configuration

Edit `config.py` to customize:

### Model Architecture
```python
IMAGE_EMB_DIM = 768          # CLIP image dimension
TEXT_EMB_DIM = 1027          # Qwen text dimension (1024 + 3 structural)
IMAGE_BRANCH_DIMS = [768, 512]
TEXT_BRANCH_DIMS = [1027, 512]
FUSION_DIM = 1024            # Concatenated
REGRESSION_HEAD_DIMS = [1024, 512, 256, 128, 1]
```

### Training Parameters
```python
BATCH_SIZE = 256
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 15
```

### Loss Function
```python
LOSS_TYPE = "weighted_smape"  # Options: weighted_smape, focal_smape, 
                              # hybrid_focal_smape, huber, mse, mae
PREDICT_LOG = True            # Predict log(price) for stability
```

### Advanced Features
```python
USE_ADVANCED_MODEL = True     # Use MoE + Cross-Attention
NUM_EXPERTS = 3               # Mixture-of-Experts
NUM_ATTENTION_HEADS = 8       # Cross-attention heads
```

## 📊 Data Requirements

### Training Data
Located at `/data/utk/amazon_ml_2025/`:
- **CLIP embeddings**: `embeddings_clip/train/`
- **Qwen embeddings**: `embeddings_qwen/qwen3_embeddings_train.pkl`
- **Training CSV**: `dataset/train/train.csv`

**Split**: 80% train, 20% validation (stratified by price)

### Test Data (for predictions)
- **CLIP embeddings**: `embeddings_clip/test/`
- **Qwen embeddings**: `embeddings_qwen/qwen3_embeddings_test.pkl`

## 💡 Key Differences from Original Fusion

1. **No structured feature engineering**
   - Original: Parses value/unit from catalog, creates 5 features
   - Qwen: Structural info already embedded in text embeddings

2. **Two branches instead of three**
   - Original: Image + Text + Structured (3 branches)
   - Qwen: Image + Text (2 branches, structural info in text)

3. **Different text embedding dimensions**
   - Original: 768 (CLIP text)
   - Qwen: 1027 (Qwen text with structural features)

## 📈 Expected Performance

Based on training results:
- **Validation SMAPE**: ~33%
- **Validation MAE**: ~$7.50
- **Validation RMSE**: ~$21.80
- **R² Score**: ~0.57

Model continues improving until early stopping triggers.

## 🐛 Troubleshooting

### GPU out of memory
```python
# In config.py, reduce batch size
BATCH_SIZE = 128  # or 64
```

### Training too slow
```python
# Reduce model capacity
IMAGE_BRANCH_DIMS = [768, 256]
TEXT_BRANCH_DIMS = [1027, 256]
REGRESSION_HEAD_DIMS = [512, 256, 128, 1]
```

### Model not improving
- Try different loss functions: `focal_smape`, `hybrid_focal_smape`
- Adjust learning rate: `LEARNING_RATE = 5e-4` or `2e-3`
- Increase early stopping patience: `EARLY_STOPPING_PATIENCE = 20`

### Resume from Checkpoint
```python
# In train.py, add before training loop:
checkpoint_path = config.CHECKPOINT_DIR / 'checkpoint_epoch_200.pt'
if checkpoint_path.exists():
    epoch, metrics = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
    print(f"Resumed from epoch {epoch+1}")
```

### Embeddings not aligned
```
AssertionError: CLIP and Qwen sample IDs don't match!
```
Check that embeddings were generated from the same dataset.

## 📝 Output Locations

**Training outputs:**
- Checkpoints: `/data/utk/amazon_ml_2025/outputs/fusion_qwen/checkpoints/`
  - `best_model.pt` - Best model by validation SMAPE
  - `checkpoint_epoch_100.pt`, etc. - Periodic checkpoints
- Training history: `training_history.json`

**Evaluation outputs:**
- Detailed results: `evaluation_results.csv`

**Test predictions:**
- Predictions: `test_out.csv` (ready for submission)

## 🎯 Advanced Usage

### Custom Loss Function
```python
# In config.py:
LOSS_TYPE = "focal_smape"     # Focuses on high-error samples
SMAPE_GAMMA = 2.0             # Quadratic penalty (1.0=linear, 3.0=cubic)
```

### Ensemble with Other Models
Average predictions from multiple models for potentially better performance:
```python
# Combine with fusion_all or other models
ensemble_pred = (qwen_pred + siglip_pred + other_pred) / 3
```

## 🎯 Next Steps

After training:
1. **Wait for training to complete** (or trigger early stopping)
2. **Run evaluation**: `python evaluate_by_groups.py`
3. **Analyze results**: Check which price ranges perform worst
4. **Generate test predictions**: `python predict_test.py`
5. **Ensemble**: Combine with other fusion models for better results

## 📚 Performance Tips

- **Start with base model** (`USE_ADVANCED_MODEL=False`) for faster experiments
- **Monitor validation metrics** - Stop if overfitting occurs
- **Try different loss functions** - `weighted_smape` focuses on low prices
- **Use gradient clipping** - Already enabled (helps with stability)
- **Ensemble multiple models** - Often improves final performance

Good luck! 🚀
