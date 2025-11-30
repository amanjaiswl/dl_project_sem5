"""Configuration parameters for Qwen3-0.6B Fine-tuning Pipeline"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent  # Go up to qwen3_embed root
PIPELINE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = PIPELINE_DIR / "models"
OUTPUTS_DIR = PIPELINE_DIR / "outputs"

# Data paths
TRAIN_CSV = Path("/home/subham/dev/student_resource/dataset/train.csv")
TEST_CSV = Path("/home/subham/dev/student_resource/dataset/test.csv")
CAPTIONS_CSV = Path("/home/subham/train_with_vllm_captions.csv")

# Processed data paths (already prepared)
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_PROCESSED = PROCESSED_DIR / "train_processed.csv"
TEST_PROCESSED = PROCESSED_DIR / "test_processed.csv"

# Model configuration
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MAX_LENGTH = 512  # Maximum sequence length
USE_FP16 = True  # Use mixed precision for faster training

# Text processing
TEXT_SEPARATOR = " [SEP] "
CAPTION_FALLBACK = True  # Use catalog_content only if caption is missing

# QLoRA Configuration
USE_QLORA = True
LORA_R = 16  # LoRA rank
LORA_ALPHA = 32  # LoRA alpha
LORA_DROPOUT = 0.1  # LoRA dropout
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training configuration
RANDOM_SEED = 42
VAL_SPLIT_RATIO = 0.1  # 10% validation split
LEARNING_RATE = 2e-4  # Lower learning rate for fine-tuning
NUM_EPOCHS = 10  # Fewer epochs for fine-tuning
BATCH_SIZE = 8  # Smaller batch size for fine-tuning
GRADIENT_ACCUMULATION_STEPS = 4  # Accumulate gradients
MAX_GRAD_NORM = 1.0  # Gradient clipping
WARMUP_STEPS = 100  # Warmup steps
PATIENCE = 3  # Early stopping patience

# Checkpointing and monitoring
VAL_SAMPLE_SIZE = 1000  # Number of validation samples for faster evaluation (None for full)
PRINT_INTERVAL = 50  # Print training loss every N batches
VAL_INTERVAL = 200  # Run validation every N batches during training
SAVE_EVERY_EPOCH = True  # Save checkpoint every epoch

# Regression head architecture
REGRESSION_HIDDEN_DIMS = [512, 256, 128]  # Hidden layer sizes
DROPOUT_RATE = 0.1  # Lower dropout for fine-tuning

# Price preprocessing
USE_LOG_PRICE = True  # Predict log prices for better numerical stability
PRICE_OFFSET = 1.0  # Add offset before taking log to avoid log(0)

# Hardware
DEVICE = "cuda"  # or "cpu"
NUM_WORKERS = 4

# Logging
LOG_DIR = OUTPUTS_DIR / "logs"
LOG_LEVEL = "INFO"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, OUTPUTS_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
