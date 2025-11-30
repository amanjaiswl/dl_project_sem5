# CLIP Fine-tuning for Product Price Prediction

Fine-tune CLIP ViT-Large-Patch14 on product dataset and extract embeddings for downstream regression.

## 📁 Project Structure

```
clip_finetune/
├── config.py              # Configuration parameters
├── dataset.py             # ProductDataset class (supports CLIP mode)
├── model.py               # CLIPFineTuner class
├── utils.py               # Utility functions
├── train.py               # Main training script
├── extract_embeddings.py  # Embedding extraction script
├── checkpoints/           # Saved model checkpoints (created automatically)
└── embeddings/            # Extracted embeddings (created automatically)
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd clip_finetune

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision transformers pillow numpy pandas tqdm scikit-learn
```

### 2. Train CLIP Model

```bash
python train.py
```

**What it does:**
- Loads train split from `student_resource/dataset/train.csv`
- Automatically splits into 90% train / 10% validation
- Fine-tunes CLIP (last 2 vision transformer blocks unfrozen)
- Saves best checkpoint to `checkpoints/best_model.pt`
- Uses early stopping and learning rate scheduling

**Training Configuration:**
- **Early Stopping**: Stops if no improvement for 3 epochs
- **LR Scheduler**: Reduces LR by 0.5× if no improvement for 2 epochs
- **Max Epochs**: 10 (typically stops early at 5-7 epochs)
- **Batch Size**: 32
- **Learning Rate**: 5e-6

**Expected training output:**
```
CLIP Fine-tuning for Product Price Prediction
============================================================
Device: cuda:1
Model: openai/clip-vit-large-patch14
Batch size: 32
Learning rate: 5e-06
Epochs: 10
Unfrozen blocks: 3
============================================================
Epoch 1/10: 100%|████████████| 1777/1777 [18:23<00:00, loss=0.3421]
...
✓ Saved best checkpoint (val_loss=0.3156)
```

### 3. Extract Embeddings

```bash
# Extract train embeddings
python extract_embeddings.py train

# Extract test embeddings
python extract_embeddings.py test
```

**What it does:**
- Loads the best model checkpoint (or base CLIP if not found)
- Extracts embeddings for specified split
- Saves to `embeddings/train/` or `embeddings/test/`

## 📦 Output Files

### Checkpoints (in `checkpoints/`)
- `best_model.pt` - Best model based on validation loss
- `checkpoint_epoch_N.pt` - Checkpoint from each epoch

### Embeddings (in `embeddings/train/` and `embeddings/test/`)
- `sample_ids.npy` - Sample IDs (75k per split)
- `image_embeddings.npy` - Image-only embeddings (N × 768)
- `image_text_embeddings.npy` - Concatenated `[image || text]` embeddings (N × 1536)
- `text_embeddings.npy` - Text-only embeddings (N × 768)
- `all_embeddings.npz` - All embeddings in one file
- `training_history.json` - Training/validation loss per epoch

## 🔧 Configuration

Edit `config.py` to adjust hyperparameters:

```python
# Model
NUM_UNFROZEN_BLOCKS = 3        # Number of last transformer blocks to unfreeze

# Training
BATCH_SIZE = 32                # Adjust based on GPU memory
NUM_EPOCHS = 10                # Max training epochs
LEARNING_RATE = 5e-6          # Learning rate for unfrozen layers
WEIGHT_DECAY = 0.01           # L2 regularization
GRADIENT_CLIP = 1.0           # Gradient clipping threshold
VAL_SPLIT = 0.1               # Validation split (10%)

# Early Stopping
EARLY_STOPPING_PATIENCE = 3    # Stop after N epochs no improvement
LR_SCHEDULER_PATIENCE = 2      # Reduce LR after N epochs no improvement
LR_SCHEDULER_FACTOR = 0.5      # Multiply LR by this
LR_SCHEDULER_MIN_LR = 1e-7    # Minimum LR

# Hardware
DEVICE = "cuda:1"             # GPU device (cuda:0 or cuda:1)
NUM_WORKERS = 4               # DataLoader workers
```

## 💡 Architecture Details

**Base Model:** `openai/clip-vit-large-patch14`
- Vision: ViT-Large with 24 transformer layers
- Text: Transformer with 77 max token length
- Embedding dimension: 768

**Fine-tuning Strategy:**
- Freeze entire model
- Unfreeze last 3 transformer blocks in vision encoder
- Unfreeze vision post-layernorm and projection
- Train with contrastive loss (CLIP's standard objective)

**Why this approach?**
- Preserves CLIP's powerful alignment
- Adapts vision encoder to product images
- Prevents catastrophic forgetting

**Regularization:**
- Weight decay: 0.01
- Gradient clipping: 1.0
- Validation monitoring for early stopping

## 📊 Using the Embeddings

### Load Embeddings

```python
import numpy as np

# Load train embeddings
train_data = np.load('clip_finetune/embeddings/train/all_embeddings.npz')
train_sample_ids = train_data['sample_ids']                      # (75000,)
train_image_embs = train_data['image_embeddings']                # (75000, 768)
train_image_text_embs = train_data['image_text_embeddings']      # (75000, 1536)
```

### Train Regression Head

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

# Load train embeddings
train_data = np.load('clip_finetune/embeddings/train/all_embeddings.npz')
train_embeddings = train_data['image_text_embeddings']  # (75000, 1536)
train_sample_ids = train_data['sample_ids']

# Load prices for training
df_train = pd.read_csv('student_resource/dataset/train.csv')
df_train = df_train.set_index('sample_id')
train_prices = df_train.loc[train_sample_ids, 'price'].values

# Train regression model
model = Ridge(alpha=1.0)
model.fit(train_embeddings, train_prices)

# Load test embeddings and predict
test_data = np.load('clip_finetune/embeddings/test/all_embeddings.npz')
test_embeddings = test_data['image_text_embeddings']
test_sample_ids = test_data['sample_ids']

predictions = model.predict(test_embeddings)

# Create submission
submission = pd.DataFrame({
    'sample_id': test_sample_ids,
    'price': predictions
})
submission.to_csv('test_out.csv', index=False)
```

## 🗂️ Dataset Format

```python
from dataset import ProductDataset

# Usage: just specify split and processor
dataset = ProductDataset(
    split="train",        # or "test"
    processor=clip_processor,
    use_augmentation=False  # default, no augmentation
)
```

**Auto-detected paths:**
- CSV: `student_resource/dataset/{split}.csv`
- Images: `student_resource/data/{split}/*.jpg`

**Returns:**
- `pixel_values`: Preprocessed image tensor
- `input_ids`: Tokenized text
- `attention_mask`: Text attention mask
- `sample_id`: Sample identifier

## 📈 Training Progress Markers

### Checkpoint Saving

| Checkpoint Type | When Saved | Filename | Overwrites? |
|----------------|------------|----------|-------------|
| **Best Model** | When validation loss improves | `best_model.pt` | ✅ Yes |
| **Epoch Checkpoint** | After every epoch | `checkpoint_epoch_N.pt` | ❌ No |

### Progress Indicators

**Good Signs:**
- ✅ Train loss decreasing steadily
- ✅ Val loss decreasing or stable
- ✅ Val loss close to train loss

**Warning Signs:**
- ⚠️ Train loss decreasing but val loss increasing → Overfitting
- ⚠️ Both losses not decreasing → Learning rate issues
- ⚠️ Loss = NaN or exploding → Gradient issues

## ⏱️ Time Estimates

With ~63k images:
- Training: ~20-22 min per epoch
- Full training (if all 10 epochs): ~3.5-4 hours
- Likely (with early stopping): ~2-2.5 hours (5-7 epochs)

## 🐛 Troubleshooting

### Out of Memory
Reduce `BATCH_SIZE` in `config.py` (try 16 or 24)

### Images not found
Check that images exist in `student_resource/data/train/` and filenames match those in `image_link`

### CUDA out of memory on GPU 0
Script uses GPU 1 by default

### Training too slow
Increase `NUM_WORKERS` in `config.py`

### Want to use pretrained embeddings only
Just run `extract_embeddings.py` without training first (uses base CLIP model)

### Overfitting
- Reduce `EARLY_STOPPING_PATIENCE` to 2
- Increase `WEIGHT_DECAY` to 0.02
- Decrease `NUM_UNFROZEN_BLOCKS` to 2

### Underfitting
- Increase `NUM_UNFROZEN_BLOCKS` to 4
- Increase `LEARNING_RATE` to 1e-5

## 🎯 Next Steps

After generating embeddings:
1. Train a regression head (Ridge, MLP, etc.) on the embeddings
2. Predict prices for test set
3. Evaluate using SMAPE metric
4. Submit predictions to challenge portal

Good luck! 🚀
