# Fusion All - Hierarchical Multi-Modal Fusion

A hierarchical fusion model combining **ALL FOUR embeddings** for price prediction.

## Overview

This model uses a sophisticated hierarchical fusion strategy to combine:
- **CLIP image** (768-dim) - General visual features
- **SigLIP image** (1152-dim) - Specific visual-linguistic alignment  
- **SigLIP text** (1024-dim) - Product descriptions
- **Qwen text** (1027-dim) - Semantic understanding + structural features

## Architecture

### Hierarchical Fusion Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│                      INPUT EMBEDDINGS (75k samples)                  │
├─────────────────────────────────────────────────────────────────────┤
│  CLIP Image (768)  │  SigLIP Image (1152)  │  SigLIP Text (1024)  │  Qwen Text (1027)  │
└──────────┬──────────┴───────────┬────────────┴──────────┬───────────┴──────────┬─────────┘
           │                      │                        │                      │
           ▼                      ▼                        ▼                      ▼
      ┌────────┐            ┌────────┐               ┌────────┐           ┌────────┐
      │  MLP   │            │  MLP   │               │  MLP   │           │  MLP   │
      │ Branch │            │ Branch │               │ Branch │           │ Branch │
      └────┬───┘            └────┬───┘               └────┬───┘           └────┬───┘
           │ (256)               │ (256)                   │ (256)              │ (256)
           │                     │                         │                    │
┌──────────┴─────────────────────┴─────────┐   ┌──────────┴────────────────────┴────────┐
│        STAGE 1: Image-Image Fusion        │   │       STAGE 2: Text-Text Fusion        │
│         (CLIP + SigLIP images)            │   │       (SigLIP + Qwen texts)            │
│    Concat [256+256] → 512                 │   │    Concat [256+256] → 512              │
│           ↓                               │   │           ↓                             │
│    Cross-Attention (8 heads)              │   │    Cross-Attention (8 heads)           │
│           ↓                               │   │           ↓                             │
│      Image Features (512)                 │   │      Text Features (512)               │
└──────────┬────────────────────────────────┘   └──────────┬─────────────────────────────┘
           │                                               │
           └────────────────┬──────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │  STAGE 3: Image-Text Fusion   │
            │  Concat [512+512] → 1024      │
            │           ↓                   │
            │  Cross-Attention (8 heads)    │
            │           ↓                   │
            │  Multimodal Features (1024)   │
            └───────────────┬───────────────┘
                            │
            ┌───────────────┴───────────────┐
            │  STAGE 4: MoE Regression      │
            │  Gating Network (3 experts)   │
            │           ↓                   │
            │  Expert 1: MLP [1024→256→1]   │
            │  Expert 2: MLP [1024→256→1]   │
            │  Expert 3: MLP [1024→256→1]   │
            │           ↓                   │
            │      Price Prediction         │
            └───────────────────────────────┘
```

### Key Features

1. **Hierarchical Fusion**
   - Stage 1: Images understand each other (CLIP ↔ SigLIP)
   - Stage 2: Texts understand each other (SigLIP ↔ Qwen)
   - Stage 3: Images and texts interact (visual ↔ linguistic)

2. **Cross-Attention Mechanism**
   - 8 attention heads per fusion stage
   - Scaled dot-product attention
   - Residual connections + layer normalization

3. **Mixture-of-Experts (MoE)**
   - 3 expert networks with gated routing
   - Experts can specialize on different price ranges

4. **Learnable Scaling**
   - Each embedding has a learnable scale parameter
   - Automatically balances contribution from each modality

## 📁 Project Structure

```
fusion_all/
├── config.py              # Configuration and hyperparameters
├── dataset.py             # Dataset loading (4 embedding sources)
├── model.py               # Hierarchical fusion architecture
├── train.py               # Training script
├── evaluate_by_groups.py  # Detailed evaluation by price bins
├── predict_test.py        # Generate test predictions
└── README.md             # This file
```

**Shared components** (in `fusion_common/`):
- `loss_functions.py` - Custom loss functions
- `utils.py` - Utility functions

## 🚀 Quick Start

### 1. Train Model

```bash
cd fusion_all
python train.py
```

**What it does:**
- Loads all 4 embeddings from their respective sources
- Splits into 80% train / 20% validation (stratified by price)
- Trains hierarchical fusion model with early stopping
- Saves checkpoints and training history

**Expected runtime:** ~10-15 min/epoch on RTX A5000

### 2. Evaluate Model

```bash
python evaluate_by_groups.py
```

Provides detailed metrics by price bins and ranges.

### 3. Generate Test Predictions

```bash
python predict_test.py
```

Creates submission-ready predictions in `test_out.csv`.

## 🔧 Configuration

Edit `config.py` to adjust:

```python
# Data paths
CLIP_EMBEDDINGS_DIR = Path("/data/utk/amazon_ml_2025/embeddings_clip/train")
SIGLIP_EMBEDDINGS_DIR = Path("/data/utk/amazon_ml_2025/embedding_cache")
QWEN_EMBEDDINGS_PATH = Path("/data/utk/amazon_ml_2025/embeddings_qwen/qwen3_embeddings_train.pkl")

# Model architecture
CLIP_IMAGE_BRANCH_DIMS = [768, 512]
SIGLIP_IMAGE_BRANCH_DIMS = [1152, 512]
SIGLIP_TEXT_BRANCH_DIMS = [1024, 512]
QWEN_TEXT_BRANCH_DIMS = [1027, 512]

# Training
BATCH_SIZE = 256
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-3
LOSS_TYPE = "weighted_smape"
EARLY_STOPPING_PATIENCE = 15

# Advanced features
USE_ADVANCED_MODEL = True  # Use MoE + Cross-Attention
NUM_EXPERTS = 3
NUM_ATTENTION_HEADS = 8
```

## 💡 Why Hierarchical Fusion?

### Benefits:

1. **Modality-Specific Interactions First**
   - Images complement each other (CLIP general + SigLIP specific)
   - Texts complement each other (SigLIP captions + Qwen semantic)

2. **Cross-Modal Interactions Second**
   - Combined images interact with combined texts
   - More meaningful multimodal representation

3. **Efficiency**
   - Each fusion stage is 512 dims (manageable)
   - Final fusion is 1024 dims (optimal)

### vs. Flat Fusion:

```
Flat:   [256+256+256+256=1024] → One big fusion → Output
         ❌ All interactions at once
         ❌ May miss modality-specific patterns

Hierarchical: [256+256→512] + [256+256→512] → [512+512→1024] → Output
         ✓ Staged interactions
         ✓ Captures both intra-modal and cross-modal patterns
```

## 📊 Model Statistics

- **Total Parameters**: ~11.8M
- **Embedding Inputs**: 3971 dims (768 + 1152 + 1024 + 1027)
- **Branch Outputs**: 4 × 256 = 1024 dims
- **Fusion Stages**: 3 cross-attention layers
- **Final Representation**: 1024 dims

## 📈 Expected Performance

With hierarchical fusion of 4 embeddings:
- **Target SMAPE**: <30% (better than individual models)
- **Training time**: ~10-15 min/epoch on RTX A5000
- **Convergence**: ~20-30 epochs with early stopping

## 🐛 Troubleshooting

### CUDA out of memory
```python
# In config.py, reduce batch size
BATCH_SIZE = 128  # or 64
```

### NaN loss during training
- Reduce learning rate: `LEARNING_RATE = 5e-4`
- Check for inf/nan in embeddings

### Model not improving
- Try different loss: `LOSS_TYPE = "focal_smape"`
- Increase patience: `EARLY_STOPPING_PATIENCE = 20`
- Adjust learning rate

### Training too slow
```python
# Simplify model
USE_ADVANCED_MODEL = False
# Or reduce branch dimensions
CLIP_IMAGE_BRANCH_DIMS = [768, 256]
```

## 🎯 Next Steps

After training:
1. Evaluate performance across price ranges
2. Generate test predictions
3. Consider ensemble with other fusion models
4. Analyze which embeddings contribute most

## 📝 Notes

- **Requires all 4 embeddings**: Ensure CLIP, SigLIP (image & text), and Qwen embeddings are available
- **Memory intensive**: Uses more GPU memory than 2-branch models
- **Best performance**: Combines strengths of all embedding types
- **Training stability**: Use gradient clipping and early stopping

Good luck! 🚀
