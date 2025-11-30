

import os, re, math, gc, joblib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import sys, time
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

print("=" * 60)

# ---------------- Enhanced Config ----------------
DATASET_PATH = Path('dataset/')
MODELS_DIR = Path('models_enhanced')
MODELS_DIR.mkdir(exist_ok=True)

# Improved parameters based on analysis
TOP_K_TOKENS = 1200        # Increased from 800 (more vocabulary coverage)
NFOLDS = 5
SEED = 42
MIN_TOKEN_FREQ = 3         # Filter very rare tokens
MAX_FEATURES_TFIDF = 500   # TF-IDF features to add
SVD_COMPONENTS = 50        # Dimensionality reduction

print(f"📊 CONFIGURATION:")
print(f"• Top tokens: {TOP_K_TOKENS}")
print(f"• TF-IDF features: {MAX_FEATURES_TFIDF}")
print(f"• SVD components: {SVD_COMPONENTS}")
print(f"• Cross-validation folds: {NFOLDS}")

# ---------------- Enhanced Helper Functions ----------------
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom < 1e-6] = 1e-6
    return np.mean(np.abs(y_pred - y_true) / denom) * 100.0

def enhanced_tokenizer(text):
    """Improved tokenizer with better preprocessing"""
    if not isinstance(text, str):
        return []
    
    # Convert to lowercase and handle special cases
    text = text.lower()
    
    # Replace common abbreviations and units
    replacements = {
        'oz': 'ounce', 'lb': 'pound', 'lbs': 'pounds',
        'fl oz': 'fluid_ounce', 'ct': 'count',
        'pcs': 'pieces', 'pkg': 'package'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Split on non-alphanumeric, but preserve numbers with units
    tokens = re.split(r'[^a-z0-9]+', text)
    tokens = [t for t in tokens if t and 2 <= len(t) <= 25]  # Better length filtering
    
    return tokens

def extract_enhanced_ipq(text):
    """Enhanced quantity extraction with more patterns"""
    if not isinstance(text, str):
        return 1
    
    text = text.lower()
    patterns = [
        r'pack of (\d{1,3})',
        r'(\d{1,3})\s*-?\s*pack',
        r'(\d{1,3})\s*pcs?',
        r'(\d{1,3})\s*pieces?',
        r'(\d{1,3})\s*ct\b',
        r'(\d{1,3})\s*count\b',
        r'x\s*(\d{1,3})\b',
        r'(\d{1,3})\s*pouches?',
        r'(\d{1,3})\s*bottles?',
        r'(\d{1,3})\s*cans?',
        r'(\d{1,3})\s*boxes?',
        r'set of (\d{1,3})',
        r'(\d{1,3})\s*units?'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                val = int(match.group(1))
                if 1 <= val <= 500:  # More reasonable upper bound
                    return val
            except:
                pass
    return 1

def extract_numerical_features(text):
    """Extract various numerical patterns from text"""
    if not isinstance(text, str):
        return {'numbers': [], 'weights': [], 'volumes': []}
    
    text = text.lower()
    
    # Extract all numbers
    numbers = [float(m) for m in re.findall(r'\d+\.?\d*', text)]
    
    # Extract weights (oz, lb, g, kg)
    weight_patterns = [
        r'(\d+\.?\d*)\s*(?:oz|ounce|ounces)',
        r'(\d+\.?\d*)\s*(?:lb|lbs|pound|pounds)',
        r'(\d+\.?\d*)\s*(?:g|gram|grams)',
        r'(\d+\.?\d*)\s*(?:kg|kilogram|kilograms)'
    ]
    weights = []
    for pattern in weight_patterns:
        weights.extend([float(m) for m in re.findall(pattern, text)])
    
    # Extract volumes (fl oz, ml, l)
    volume_patterns = [
        r'(\d+\.?\d*)\s*(?:fl\s*oz|fluid\s*ounce)',
        r'(\d+\.?\d*)\s*(?:ml|milliliter)',
        r'(\d+\.?\d*)\s*(?:l|liter|liters)'
    ]
    volumes = []
    for pattern in volume_patterns:
        volumes.extend([float(m) for m in re.findall(pattern, text)])
    
    return {
        'numbers': numbers,
        'weights': weights,
        'volumes': volumes
    }

# ---------------- Load Data ----------------
print("\n📂 LOADING DATASETS...")
train_df = pd.read_csv(DATASET_PATH / 'train.csv')
test_df = pd.read_csv(DATASET_PATH / 'test.csv')

print(f"• Training samples: {len(train_df):,}")
print(f"• Test samples: {len(test_df):,}")

# ---------------- Enhanced Vocabulary Building ----------------
print("\n🔤 BUILDING ENHANCED VOCABULARY...")

# Build token vocabulary with frequency filtering
counter = Counter()
for text in train_df['catalog_content'].fillna(''):
    tokens = enhanced_tokenizer(text)
    counter.update(tokens)

# Also include test data tokens (careful not to leak)
for text in test_df['catalog_content'].fillna(''):
    tokens = enhanced_tokenizer(text)
    counter.update(tokens)

# Filter tokens by frequency and select top K
filtered_tokens = [(tok, count) for tok, count in counter.items() if count >= MIN_TOKEN_FREQ]
most_common = [tok for tok, _ in sorted(filtered_tokens, key=lambda x: x[1], reverse=True)[:TOP_K_TOKENS]]
token_to_idx = {tok: i for i, tok in enumerate(most_common)}

print(f"• Total unique tokens: {len(counter):,}")
print(f"• Tokens with freq >= {MIN_TOKEN_FREQ}: {len(filtered_tokens):,}")
print(f"• Selected top tokens: {len(token_to_idx):,}")

# ---------------- Enhanced Feature Engineering ----------------
def build_enhanced_features(df, fit_tfidf=False, tfidf_vectorizer=None, svd_model=None):
    """Build comprehensive feature set"""
    print(f"🔧 Processing {len(df):,} samples...")
    start_time = time.time()
    
    n = len(df)
    catalog = df['catalog_content'].fillna('').astype(str)
    
    # === BASIC TEXT FEATURES ===
    print("  • Basic text features...")
    lengths = catalog.apply(len).values.reshape(-1, 1).astype(np.float32)
    word_counts = catalog.apply(lambda s: len(enhanced_tokenizer(s))).values.reshape(-1, 1).astype(np.float32)
    
    # Enhanced text statistics
    char_counts = catalog.apply(len).values.reshape(-1, 1).astype(np.float32)
    unique_word_counts = catalog.apply(lambda s: len(set(enhanced_tokenizer(s)))).values.reshape(-1, 1).astype(np.float32)
    avg_word_len = catalog.apply(lambda s: np.mean([len(t) for t in enhanced_tokenizer(s)]) if enhanced_tokenizer(s) else 0.0).values.reshape(-1, 1).astype(np.float32)
    digit_counts = catalog.apply(lambda s: sum(ch.isdigit() for ch in s)).values.reshape(-1, 1).astype(np.float32)
    upper_counts = catalog.apply(lambda s: sum(ch.isupper() for ch in s)).values.reshape(-1, 1).astype(np.float32)
    
    # === ENHANCED QUANTITY FEATURES ===
    print("  • Quantity and numerical features...")
    ipq_vals = catalog.apply(extract_enhanced_ipq).values.reshape(-1, 1).astype(np.float32)
    
    # Extract numerical features
    numerical_features = []
    for text in catalog:
        num_data = extract_numerical_features(text)
        features = [
            len(num_data['numbers']),                                    # count of numbers
            np.mean(num_data['numbers']) if num_data['numbers'] else 0,  # avg number
            np.max(num_data['numbers']) if num_data['numbers'] else 0,   # max number
            len(num_data['weights']),                                    # count of weights
            np.sum(num_data['weights']) if num_data['weights'] else 0,   # total weight
            len(num_data['volumes']),                                    # count of volumes
            np.sum(num_data['volumes']) if num_data['volumes'] else 0,   # total volume
        ]
        numerical_features.append(features)
    
    numerical_features = np.array(numerical_features, dtype=np.float32)
    
    # === TOKEN FREQUENCY FEATURES ===
    print("  • Token frequency features...")
    token_mat = np.zeros((n, len(token_to_idx)), dtype=np.float32)
    for i, text in enumerate(catalog):
        tokens = enhanced_tokenizer(text)
        if tokens:
            token_counts = Counter(tokens)
            for tok, cnt in token_counts.items():
                idx = token_to_idx.get(tok)
                if idx is not None:
                    token_mat[i, idx] = float(cnt)
    
    # Normalize token features (TF normalization)
    row_sums = token_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    token_mat = token_mat / row_sums
    
    # === TF-IDF FEATURES ===
    print("  • TF-IDF features...")
    tfidf_features = None
    if fit_tfidf:
        # Fit TF-IDF on training data
        tfidf_vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES_TFIDF,
            tokenizer=enhanced_tokenizer,
            token_pattern=None,
            lowercase=False,  # Already handled in tokenizer
            stop_words=None,
            ngram_range=(1, 2),  # Include bigrams
            min_df=3,
            max_df=0.95
        )
        tfidf_features = tfidf_vectorizer.fit_transform(catalog).toarray().astype(np.float32)
        
        # Apply SVD for dimensionality reduction
        svd_model = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED)
        tfidf_features = svd_model.fit_transform(tfidf_features).astype(np.float32)
        
    elif tfidf_vectorizer is not None and svd_model is not None:
        # Transform test data using fitted models
        tfidf_features = tfidf_vectorizer.transform(catalog).toarray().astype(np.float32)
        tfidf_features = svd_model.transform(tfidf_features).astype(np.float32)
    
    # === COMBINE ALL FEATURES ===
    feature_list = [
        lengths, word_counts, char_counts, unique_word_counts, avg_word_len,
        digit_counts, upper_counts, ipq_vals, numerical_features, token_mat
    ]
    
    if tfidf_features is not None:
        feature_list.append(tfidf_features)
    
    X = np.hstack(feature_list)
    
    processing_time = time.time() - start_time
    print(f"  ✅ Features built: {X.shape} in {processing_time:.1f}s")
    
    return X, tfidf_vectorizer, svd_model

# ---------------- Build Features ----------------
print("\n🏗️ BUILDING ENHANCED FEATURES...")

# Build training features (fit TF-IDF)
X_train, tfidf_vectorizer, svd_model = build_enhanced_features(
    train_df, fit_tfidf=True
)

# Build test features (transform only)
X_test, _, _ = build_enhanced_features(
    test_df, fit_tfidf=False, tfidf_vectorizer=tfidf_vectorizer, svd_model=svd_model
)

print(f"\n📊 FEATURE SUMMARY:")
print(f"• Training shape: {X_train.shape}")
print(f"• Test shape: {X_test.shape}")

# === PREPARE TARGET ===
y_raw = train_df['price'].fillna(0.0).astype(float).values
y_train = np.log1p(np.clip(y_raw, a_min=0.0, a_max=None))

print(f"• Target range (log): {y_train.min():.3f} to {y_train.max():.3f}")

# ---------------- Enhanced Model Training ----------------
print(f"\n🚀 ENHANCED LIGHTGBM TRAINING...")

# Improved LightGBM parameters
lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,        # Slower learning for better generalization
    'num_leaves': 255,            # More complex trees
    'min_data_in_leaf': 25,       # Prevent overfitting
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,             # L1 regularization
    'lambda_l2': 0.1,             # L2 regularization
    'verbosity': -1,
    'seed': SEED,
    'n_jobs': -1
}

# Manual K-fold implementation 
N = X_train.shape[0]
indices = np.arange(N)
rng = np.random.default_rng(SEED)
rng.shuffle(indices)

fold_sizes = [N // NFOLDS + (1 if i < (N % NFOLDS) else 0) for i in range(NFOLDS)]
folds = []
start = 0
for fs in fold_sizes:
    folds.append(indices[start:start+fs])
    start += fs

oof_preds = np.zeros(N, dtype=np.float32)
models = []
fold_scores = []

print(f"Starting {NFOLDS}-fold cross-validation...")

for k in range(NFOLDS):
    print(f"\n📊 FOLD {k+1}/{NFOLDS}:")
    
    val_idx = folds[k]
    train_idx = np.concatenate([folds[i] for i in range(NFOLDS) if i != k])
    
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    print(f"  • Training: {len(train_idx):,} samples")
    print(f"  • Validation: {len(val_idx):,} samples")
    
    # Create datasets
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    # Train model
    callbacks = [
        lgb.early_stopping(100),
        lgb.log_evaluation(0)  # Silent training
    ]
    
    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=3000,  # More rounds with early stopping
        valid_sets=[dtrain, dval],
        callbacks=callbacks
    )
    
    # Predict on validation fold
    pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    oof_preds[val_idx] = pred_val
    
    # Calculate fold SMAPE
    fold_pred_price = np.expm1(pred_val)
    fold_true_price = np.expm1(y_val)
    fold_smape = smape(fold_true_price, fold_pred_price)
    fold_scores.append(fold_smape)
    
    print(f"  ✅ Fold {k+1} SMAPE: {fold_smape:.3f}%")
    
    models.append(model)
    joblib.dump(model, MODELS_DIR / f'enhanced_lgb_fold_{k}.pkl')
    
    # Clean up memory
    del dtrain, dval
    gc.collect()

# === FINAL RESULTS ===
print(f"\n" + "="*60)
print(f"🎯 ENHANCED MODEL RESULTS")
print(f"="*60)

for i, score in enumerate(fold_scores):
    print(f"• Fold {i+1}: {score:.3f}% SMAPE")

mean_cv = np.mean(fold_scores)
std_cv = np.std(fold_scores)
print(f"\n📊 CROSS-VALIDATION SUMMARY:")
print(f"• Mean CV SMAPE: {mean_cv:.3f}% ± {std_cv:.3f}%")
print(f"• Best fold: {np.min(fold_scores):.3f}%")
print(f"• Worst fold: {np.max(fold_scores):.3f}%")

# Overall OOF score
oof_pred_price = np.expm1(oof_preds)
oof_true_price = np.expm1(y_train)
oof_smape = smape(oof_true_price, oof_pred_price)

print(f"\n🏆 OUT-OF-FOLD SMAPE: {oof_smape:.3f}%")

# Compare to baseline
baseline_smape = 52.89
improvement = baseline_smape - oof_smape
print(f"\n📈 IMPROVEMENT ANALYSIS:")
print(f"• Baseline (Mohil): {baseline_smape:.2f}% SMAPE")
print(f"• Enhanced: {oof_smape:.3f}% SMAPE")
print(f"• Improvement: {improvement:.3f} percentage points")

if improvement > 0:
    print(f"🎉 SUCCESS! {improvement:.3f} point improvement!")
elif improvement > -1:
    print(f"🤔 MARGINAL: {abs(improvement):.3f} point regression (within noise)")
else:
    print(f"❌ REGRESSION: {abs(improvement):.3f} point worse")

# ---------------- Generate Predictions ----------------
print(f"\n🔮 GENERATING FINAL PREDICTIONS...")

# Ensemble predictions from all folds
test_preds = np.zeros((len(models), X_test.shape[0]), dtype=np.float32)

for i, model in enumerate(models):
    test_preds[i] = model.predict(X_test, num_iteration=model.best_iteration)
    print(f"  ✅ Model {i+1}/{len(models)} predictions generated")

# Average predictions
pred_log = test_preds.mean(axis=0)
pred_price = np.expm1(pred_log)
pred_price = np.maximum(pred_price, 0.01)  # Ensure positive prices

# Create submission
submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': pred_price
})

submission.to_csv('test_out.csv', index=False)

print(f"\n🎯 SUBMISSION READY!")
print(f"• File: enhanced_test_out.csv")
print(f"• Samples: {len(submission):,}")
print(f"• Price range: ${pred_price.min():.2f} - ${pred_price.max():.2f}")
print(f"• Median price: ${np.median(pred_price):.2f}")

print(f"\n📋 Sample predictions:")
print(submission.head(10))

# Save models and artifacts
artifacts = {
    'token_vocab': most_common,
    'oof_smape': oof_smape,
    'cv_scores': fold_scores,
    'feature_count': X_train.shape[1]
}

with open('enhanced_artifacts.json', 'w') as f:
    json.dump(artifacts, f, indent=2)

joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(svd_model, 'svd_model.pkl')

print(f"\n✅ ENHANCED PIPELINE COMPLETE!")

