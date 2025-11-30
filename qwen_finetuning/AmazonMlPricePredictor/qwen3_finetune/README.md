# Qwen3-0.6B Fine-tuning Pipeline

Complete end-to-end fine-tuning pipeline for product price prediction using Qwen3-Embedding-0.6B with QLoRA.

## Overview

This pipeline fine-tunes the entire Qwen3-0.6B model for optimal price prediction performance, rather than just using pre-computed embeddings.

### Key Features

- **End-to-end fine-tuning** of Qwen3-0.6B model with regression head
- **QLoRA** for memory-efficient training (~4-6GB GPU)
- **Log price prediction** for better numerical stability
- **Combined Huber + SMAPE loss** for robust optimization
- **Comprehensive evaluation** with multiple metrics
- **Ready for submission** with proper output formatting

## 📁 Project Structure

```
qwen3_finetune/
├── config/
│   └── config.py              # Configuration parameters
├── scripts/
│   ├── train_finetune.py      # Fine-tuning script
│   ├── evaluate_finetuned.py  # Model evaluation
│   ├── extract_embeddings.py  # Extract Qwen embeddings
│   └── predict_finetuned.py   # Prediction generation
├── models/                    # Fine-tuned models (created during training)
├── outputs/                   # Evaluation results (created during eval)
└── README.md                  # This file
```

## 🚀 Quick Start

### Complete Pipeline

```bash
cd qwen3_finetune

# Step 1: Fine-tune the model (2-4 hours)
python scripts/train_finetune.py

# Step 2: Evaluate (optional but recommended)
python scripts/evaluate_finetuned.py

# Step 3: Generate test predictions
python scripts/predict_finetuned.py \
    --test_csv /path/to/test.csv \
    --output_file test_out_finetuned.csv
```

## 📖 Detailed Usage

### Step 1: Fine-tune the Model

```bash
python scripts/train_finetune.py
```

**What it does:**
- Loads Qwen3-Embedding-0.6B pretrained model
- Adds regression head for price prediction
- Applies QLoRA for efficient fine-tuning
- Fine-tunes on combined text (image captions + catalog content)
- Predicts log prices for numerical stability
- Saves checkpoints and best model

**Training Features:**
- **Checkpointing**: Saves model every epoch
- **Resume training**: Can resume from any checkpoint
- **Comprehensive monitoring**: Prints loss, Huber, SMAPE, LR, grad norm
- **Mid-epoch validation**: Optional validation every N batches
- **Fast validation**: Optional validation sampling for speed

**Output files:**
- `models/qwen3_regression_finetuned.pt` - Final fine-tuned model
- `models/best_model.pt` - Best model checkpoint
- `models/checkpoint_epoch_N.pt` - Checkpoint for each epoch
- `models/model_metadata.json` - Model configuration and metrics
- `models/training_history.json` - Training curves and progress

**Expected time:** 2-4 hours (depends on GPU and convergence)

**Command options:**
```bash
# Custom training parameters
python scripts/train_finetune.py \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --num_epochs 15 \
    --patience 5

# Use CPU (slower but works)
python scripts/train_finetune.py --device cpu

# Custom QLoRA parameters
python scripts/train_finetune.py \
    --lora_r 32 \
    --lora_alpha 64

# Disable QLoRA (full fine-tuning, needs more memory)
python scripts/train_finetune.py --no-use_qlora

# Resume training from checkpoint
python scripts/train_finetune.py \
    --resume_from models/checkpoint_epoch_5.pt

# Faster validation (sample 1000 validation samples)
python scripts/train_finetune.py \
    --val_sample_size 1000

# More frequent monitoring
python scripts/train_finetune.py \
    --print_interval 25 \
    --val_interval 100
```

### Step 2: Evaluate the Model (Optional)

```bash
python scripts/evaluate_finetuned.py
```

**What it does:**
- Loads the fine-tuned model
- Evaluates on training data (or random sample)
- Calculates SMAPE, MAE, RMSE, R² metrics
- Shows sample predictions vs actual prices
- Analyzes error distribution and performance by price range

**Command options:**
```bash
# Evaluate on random sample (faster)
python scripts/evaluate_finetuned.py --num_samples 1000

# Use CPU
python scripts/evaluate_finetuned.py --device cpu

# Show more sample predictions
python scripts/evaluate_finetuned.py --show_samples 20
```

**Expected time:** Few minutes (depends on sample size)

### Step 3: Generate Test Predictions

```bash
python scripts/predict_finetuned.py \
    --test_csv /path/to/test.csv \
    --output_file test_out_finetuned.csv
```

**What it does:**
- Loads test.csv
- Uses catalog_content as input text
- Loads fine-tuned model
- Generates price predictions
- Post-processes (ensures positive values, rounds to 2 decimals)
- Saves to test_out_finetuned.csv

**Command options:**
```bash
# Evaluate on training sample first
python scripts/predict_finetuned.py \
    --test_csv /path/to/test.csv \
    --evaluate_train \
    --num_eval_samples 1000 \
    --output_file test_out_finetuned.csv

# Use CPU
python scripts/predict_finetuned.py \
    --test_csv /path/to/test.csv \
    --device cpu \
    --output_file test_out_finetuned.csv

# Custom batch size
python scripts/predict_finetuned.py \
    --test_csv /path/to/test.csv \
    --batch_size 4 \
    --output_file test_out_finetuned.csv
```

**Output:** `test_out_finetuned.csv` in submission format (sample_id, price)

**Expected time:** 10-30 minutes (depends on GPU and batch size)

## ⚙️ Configuration

Edit `config/config.py` to customize:

```python
# Model parameters
MODEL_NAME = "Alibaba-NLP/gte-Qwen2-0.6B-instruct"
MAX_LENGTH = 512
POOLING = "mean"

# QLoRA configuration
USE_QLORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# Training hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
PATIENCE = 3

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
```

## 📈 Model Architecture

### Components:

1. **Qwen3-Embedding-0.6B Backbone**
   - Pretrained text encoder
   - QLoRA adapters for efficient fine-tuning
   - Mean pooling for sentence embeddings

2. **Regression Head**
   - Input: 768-dimensional embeddings
   - Hidden layers: [512, 256, 128] with ReLU activation
   - Dropout: 0.1 for regularization
   - Output: Log price prediction

3. **Training Features**
   - QLoRA for memory-efficient fine-tuning
   - Log price prediction for numerical stability
   - Combined Huber + SMAPE loss function
   - Gradient accumulation and clipping

## 📊 Performance Expectations

Based on the fine-tuning approach:

- **Model Size:** ~0.6B parameters (base) + ~16M (QLoRA adapters)
- **Memory Usage:** ~4-6GB GPU memory (with QLoRA)
- **Training Time:** 2-4 hours for 10 epochs
- **Expected SMAPE:** 20-30% (better than embedding-only approach)
- **Inference Speed:** Fast (single forward pass)

## 🔍 Evaluation Metrics

The pipeline calculates:
- **SMAPE**: Symmetric Mean Absolute Percentage Error (primary metric)
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error

## 🔄 Comparison with 8B Model

| Aspect | Qwen3-0.6B | Qwen3-8B |
|--------|-------------|----------|
| Model Size | 0.6B params | 8B params |
| GPU Memory | ~2-3GB | ~8-10GB |
| Inference Speed | 3-5x faster | Baseline |
| Training Time | 1-2 hours | 3-4 hours |
| Expected SMAPE | 25-35% | 20-30% |
| Deployment | Easier | More complex |

## 🐛 Troubleshooting

### Out of GPU Memory
```bash
# Reduce batch size
python scripts/train_finetune.py --batch_size 2

# Increase gradient accumulation
python scripts/train_finetune.py --batch_size 2 --gradient_accumulation_steps 8
```

### No GPU Available
```bash
# Use CPU (much slower but works)
python scripts/train_finetune.py --device cpu
python scripts/predict_finetuned.py --test_csv /path/to/test.csv --device cpu --output_file test_out.csv
```

### Poor Performance
```bash
# Try different QLoRA settings
python scripts/train_finetune.py --lora_r 32 --lora_alpha 64

# Train for more epochs
python scripts/train_finetune.py --num_epochs 20 --patience 5

# Adjust learning rate
python scripts/train_finetune.py --learning_rate 1e-4
```

### Missing Dependencies
```bash
# Install all required packages
pip install torch transformers peft pandas numpy scikit-learn tqdm
```

## 💡 QLoRA Configuration

The pipeline uses QLoRA with the following settings:

- **Rank (r):** 16 (adjustable via `--lora_r`)
- **Alpha:** 32 (adjustable via `--lora_alpha`)
- **Dropout:** 0.1
- **Target Modules:** All linear layers in attention and MLP
- **Quantization:** 4-bit (automatic)

## 🎯 Advantages of Fine-tuning Approach

1. **End-to-end Learning:** Model learns task-specific representations
2. **Better Performance:** Typically outperforms embedding-only approaches
3. **Memory Efficient:** QLoRA allows fine-tuning with minimal memory
4. **Flexible:** Can easily adjust model architecture and training parameters
5. **Transferable:** Fine-tuned model can be used for similar tasks

## 📝 Notes

- **Virtual environment recommended**: Use venv for installation and running
- **Fine-tuning is the main step**: Takes most of the time
- **Evaluation is optional**: But recommended to check performance
- **Predictions are fast**: Single forward pass per sample
- **All steps are reproducible**: Fixed random seeds used

## 🎯 Next Steps

After running the pipeline:

1. **Check evaluation results** in `outputs/finetuned_evaluation_results.json`
2. **Review sample predictions** to ensure they look reasonable
3. **Submit predictions** using `test_out_finetuned.csv`
4. **Experiment with parameters** if performance needs improvement
5. **Consider ensemble methods** combining multiple fine-tuned models

Good luck! 🚀
