import numpy as np
import pandas as pd
from pathlib import Path
import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import r2_score
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from collections import Counter
import joblib
import warnings
warnings.filterwarnings('ignore')

ImageFile.LOAD_TRUNCATED_IMAGES = True

class AdvancedTextFeatureExtractor:
    """Enhanced text feature extraction based on the successful LightGBM approach"""
    
    def __init__(self, top_k_tokens=1200, min_token_freq=3, max_tfidf_features=500, svd_components=50):
        self.top_k_tokens = top_k_tokens
        self.min_token_freq = min_token_freq
        self.max_tfidf_features = max_tfidf_features
        self.svd_components = svd_components
        self.token_to_idx = {}
        self.tfidf_vectorizer = None
        self.svd_model = None
        
    def enhanced_tokenizer(self, text):
        """Improved tokenizer with better preprocessing"""
        if not isinstance(text, str):
            return []
        
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
        tokens = [t for t in tokens if t and 2 <= len(t) <= 25]
        
        return tokens
    
    def extract_enhanced_ipq(self, text):
        """Enhanced quantity extraction with more patterns"""
        if not isinstance(text, str):
            return 1
        
        text = text.lower()
        patterns = [
            r'pack of (\d{1,3})', r'(\d{1,3})\s*-?\s*pack', r'(\d{1,3})\s*pcs?',
            r'(\d{1,3})\s*pieces?', r'(\d{1,3})\s*ct\b', r'(\d{1,3})\s*count\b',
            r'x\s*(\d{1,3})\b', r'(\d{1,3})\s*pouches?', r'(\d{1,3})\s*bottles?',
            r'(\d{1,3})\s*cans?', r'(\d{1,3})\s*boxes?', r'set of (\d{1,3})',
            r'(\d{1,3})\s*units?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    val = int(match.group(1))
                    if 1 <= val <= 500:
                        return val
                except:
                    pass
        return 1
    
    def extract_numerical_features(self, text):
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
    
    def fit_vocabulary(self, texts):
        """Build vocabulary from training texts"""
        print("🔤 Building enhanced vocabulary...")
        
        # Build token vocabulary with frequency filtering
        counter = Counter()
        for text in texts:
            tokens = self.enhanced_tokenizer(text)
            counter.update(tokens)
        
        # Filter tokens by frequency and select top K
        filtered_tokens = [(tok, count) for tok, count in counter.items() if count >= self.min_token_freq]
        most_common = [tok for tok, _ in sorted(filtered_tokens, key=lambda x: x[1], reverse=True)[:self.top_k_tokens]]
        self.token_to_idx = {tok: i for i, tok in enumerate(most_common)}
        
        print(f"• Total unique tokens: {len(counter):,}")
        print(f"• Selected top tokens: {len(self.token_to_idx):,}")
        
        # Fit TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            tokenizer=self.enhanced_tokenizer,
            token_pattern=None,
            lowercase=False,
            stop_words=None,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.95
        )
        
        tfidf_features = self.tfidf_vectorizer.fit_transform(texts).toarray()
        
        # Fit SVD
        self.svd_model = TruncatedSVD(n_components=self.svd_components, random_state=42)
        self.svd_model.fit(tfidf_features)
        
        print(f"• TF-IDF features: {self.max_tfidf_features}")
        print(f"• SVD components: {self.svd_components}")
    
    def extract_features(self, texts):
        """Extract comprehensive text features"""
        n = len(texts)
        
        # Basic text features
        lengths = np.array([len(text) for text in texts]).reshape(-1, 1).astype(np.float32)
        word_counts = np.array([len(self.enhanced_tokenizer(text)) for text in texts]).reshape(-1, 1).astype(np.float32)
        char_counts = np.array([len(text) for text in texts]).reshape(-1, 1).astype(np.float32)
        unique_word_counts = np.array([len(set(self.enhanced_tokenizer(text))) for text in texts]).reshape(-1, 1).astype(np.float32)
        
        avg_word_len = []
        for text in texts:
            tokens = self.enhanced_tokenizer(text)
            avg_len = np.mean([len(t) for t in tokens]) if tokens else 0.0
            avg_word_len.append(avg_len)
        avg_word_len = np.array(avg_word_len).reshape(-1, 1).astype(np.float32)
        
        digit_counts = np.array([sum(ch.isdigit() for ch in text) for text in texts]).reshape(-1, 1).astype(np.float32)
        upper_counts = np.array([sum(ch.isupper() for ch in text) for text in texts]).reshape(-1, 1).astype(np.float32)
        
        # Quantity features
        ipq_vals = np.array([self.extract_enhanced_ipq(text) for text in texts]).reshape(-1, 1).astype(np.float32)
        
        # Numerical features
        numerical_features = []
        for text in texts:
            num_data = self.extract_numerical_features(text)
            features = [
                len(num_data['numbers']),
                np.mean(num_data['numbers']) if num_data['numbers'] else 0,
                np.max(num_data['numbers']) if num_data['numbers'] else 0,
                len(num_data['weights']),
                np.sum(num_data['weights']) if num_data['weights'] else 0,
                len(num_data['volumes']),
                np.sum(num_data['volumes']) if num_data['volumes'] else 0,
            ]
            numerical_features.append(features)
        numerical_features = np.array(numerical_features, dtype=np.float32)
        
        # Token frequency features
        token_mat = np.zeros((n, len(self.token_to_idx)), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = self.enhanced_tokenizer(text)
            if tokens:
                token_counts = Counter(tokens)
                for tok, cnt in token_counts.items():
                    idx = self.token_to_idx.get(tok)
                    if idx is not None:
                        token_mat[i, idx] = float(cnt)
        
        # Normalize token features
        row_sums = token_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        token_mat = token_mat / row_sums
        
        # TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform(texts).toarray().astype(np.float32)
        tfidf_features = self.svd_model.transform(tfidf_features).astype(np.float32)
        
        # Combine all features
        all_features = np.hstack([
            lengths, word_counts, char_counts, unique_word_counts, avg_word_len,
            digit_counts, upper_counts, ipq_vals, numerical_features, token_mat, tfidf_features
        ])
        
        return all_features

class HybridGroceryDataset(Dataset):
    """Dataset that combines advanced text features with images"""
    
    def __init__(self, csv_path, image_dir, text_extractor, tokenizer_name='distilbert-base-uncased', 
                 max_length=256, image_size=224, is_test=False, price_stats=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.is_test = is_test
        self.max_length = max_length
        self.price_stats = price_stats
        self.text_extractor = text_extractor
        
        # Tokenizer for deep learning
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Image transforms
        if not is_test:
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Extract advanced text features
        print("🔧 Extracting advanced text features...")
        catalog_texts = self.df['catalog_content'].fillna('').astype(str).tolist()
        self.advanced_text_features = self.text_extractor.extract_features(catalog_texts)
        print(f"• Advanced text features shape: {self.advanced_text_features.shape}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            sample_id = row['sample_id']
            catalog_text = str(row['catalog_content'])
            
            # Tokenize text for BERT
            text_encoding = self.tokenizer(
                catalog_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            
            input_ids = text_encoding['input_ids'].squeeze(0)
            attention_mask = text_encoding['attention_mask'].squeeze(0)
            
            # Load image
            image_path = os.path.join(self.image_dir, Path(row['image_link']).name)
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_transform(image)
            
            # Get advanced text features
            advanced_features = torch.tensor(
                self.advanced_text_features[idx], 
                dtype=torch.float32
            )
            
            sample = {
                'sample_id': sample_id,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'image': image_tensor,
                'advanced_features': advanced_features
            }
            
            if not self.is_test:
                price = row['price']
                if self.price_stats:
                    price = (np.log1p(price) - self.price_stats['log_mean']) / self.price_stats['log_std']
                sample['price'] = torch.tensor(price, dtype=torch.float32)
            
            return sample
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            next_idx = (idx + 1) % len(self.df)
            return self.__getitem__(next_idx)

def hybrid_collate_fn(batch):
    """Collate function for hybrid dataset"""
    batch_dict = {
        'sample_id': [item['sample_id'] for item in batch],
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'image': torch.stack([item['image'] for item in batch]),
        'advanced_features': torch.stack([item['advanced_features'] for item in batch]),
    }
    
    if 'price' in batch[0]:
        batch_dict['price'] = torch.stack([item['price'] for item in batch])
    
    return batch_dict

class HybridPriceModel(nn.Module):
    """Hybrid model combining deep learning with advanced text features"""
    
    def __init__(self, text_model_name='distilbert-base-uncased', advanced_feature_dim=1000,
                 hidden_dim=512, num_heads=8, dropout=0.2):
        super().__init__()
        
        # Text encoder (lightweight)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        # Freeze most layers
        for param in list(self.text_encoder.parameters())[:-20]:
            param.requires_grad = False
        
        # Image encoder (lightweight)
        resnet = models.resnet18(pretrained=True)  # Smaller than ResNet50
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze early layers
        for param in list(self.image_encoder.parameters())[:-20]:
            param.requires_grad = False
        
        # Dimensions
        text_dim = 768  # DistilBERT
        image_dim = 512  #18
        
        # Simple fusion (since advanced features do heavy lifting)
        self.text_proj = nn.Linear(text_dim, 256)
        self.image_proj = nn.Linear(image_dim, 256)
        
        # Advanced feature processor (most important component)
        self.advanced_processor = nn.Sequential(
            nn.Linear(advanced_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final fusion and prediction
        fusion_dim = 256 + 256 + 128  # text + image + advanced features
        
        self.final_predictor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
    def forward(self, input_ids, attention_mask, images, advanced_features):
        # Extract text features (lightweight)
        text_outputs = self.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        text_features = self.text_proj(text_outputs.last_hidden_state[:, 0, :])
        
        # Extract image features (lightweight)
        image_features = self.image_encoder(images)
        image_features = self.image_proj(image_features.view(image_features.size(0), -1))
        
        # Process advanced features (heavy lifting)
        advanced_processed = self.advanced_processor(advanced_features)
        
        # Final fusion
        all_features = torch.cat([text_features, image_features, advanced_processed], dim=1)
        
        # Price prediction
        price_pred = self.final_predictor(all_features).squeeze(1)
        
        return price_pred

def compute_price_stats(train_df):
    """Compute price statistics for normalization"""
    prices = train_df['price'].values
    log_prices = np.log1p(prices)
    
    return {
        'log_mean': np.mean(log_prices),
        'log_std': np.std(log_prices),
        'raw_mean': np.mean(prices),
        'raw_std': np.std(prices)
    }

def calculate_smape(predictions, targets):
    """Calculate SMAPE"""
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    numerator = np.abs(predictions - targets)
    denominator = (np.abs(targets) + np.abs(predictions)) / 2
    
    smape = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(smape) * 100

def train_hybrid_model(
    model, train_loader, val_loader, price_stats,
    num_epochs=10, learning_rate=1e-4, device='cuda',
    save_path='hybrid_grocery_price_model.pth'
):
    """Train the hybrid model"""
    
    model = model.to(device)
    
    # Use Huber loss
    criterion = nn.HuberLoss(delta=1.0)
    
    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=0.01
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=3, T_mult=2, eta_min=1e-6
    )
    
    best_smape = float('inf')
    patience = 3
    patience_counter = 0
    
    print(f"Starting hybrid training on {device}...")
    print("="*80)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            advanced_features = batch['advanced_features'].to(device)
            prices = batch['price'].to(device)
            
            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask, images, advanced_features)
            loss = criterion(predictions, prices)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Convert back to original scale for metrics
            with torch.no_grad():
                pred_original = torch.expm1(
                    predictions * price_stats['log_std'] + price_stats['log_mean']
                )
                target_original = torch.expm1(
                    prices * price_stats['log_std'] + price_stats['log_mean']
                )
                
                train_preds.extend(pred_original.cpu().numpy())
                train_targets.extend(target_original.cpu().numpy())
            
            train_losses.append(loss.item())
            
            batch_smape = calculate_smape(
                pred_original.cpu().numpy(), 
                target_original.cpu().numpy()
            )
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'smape': f'{batch_smape:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in val_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                images = batch['image'].to(device)
                advanced_features = batch['advanced_features'].to(device)
                prices = batch['price'].to(device)
                
                predictions = model(input_ids, attention_mask, images, advanced_features)
                loss = criterion(predictions, prices)
                
                pred_original = torch.expm1(
                    predictions * price_stats['log_std'] + price_stats['log_mean']
                )
                target_original = torch.expm1(
                    prices * price_stats['log_std'] + price_stats['log_mean']
                )
                
                val_preds.extend(pred_original.cpu().numpy())
                val_targets.extend(target_original.cpu().numpy())
                val_losses.append(loss.item())
        
        # Calculate metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_r2 = r2_score(train_targets, train_preds)
        val_r2 = r2_score(val_targets, val_preds)
        train_smape = calculate_smape(train_preds, train_targets)
        val_smape = calculate_smape(val_preds, val_targets)
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train R²: {train_r2:.4f} | Train SMAPE: {train_smape:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val R²:   {val_r2:.4f} | Val SMAPE:   {val_smape:.2f}%")
        
        # Save best model and early stopping
        if val_smape < best_smape:
            best_smape = val_smape
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_smape': val_smape,
                'price_stats': price_stats
            }, save_path)
            print(f"  ✓ Best model saved! (Val SMAPE = {val_smape:.2f}%)")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break
            
        print("="*80)
    
    return model

if __name__ == "__main__":
    print("🚀 Running Hybrid Price Prediction Model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    image_dir_train = "../resource/images/train/"
    train_csv = "../student_resource/dataset/train.csv"
    
    # Load data and compute price statistics
    train_df = pd.read_csv(train_csv)
    price_stats = compute_price_stats(train_df)
    
    # Initialize text feature extractor
    text_extractor = AdvancedTextFeatureExtractor()
    
    # Fit on training texts
    catalog_texts = train_df['catalog_content'].fillna('').astype(str).tolist()
    text_extractor.fit_vocabulary(catalog_texts)
    
    # Create dataset
    full_dataset = HybridGroceryDataset(
        csv_path=train_csv,
        image_dir=image_dir_train,
        text_extractor=text_extractor,
        tokenizer_name="distilbert-base-uncased",
        is_test=False,
        price_stats=price_stats
    )
    
    advanced_feature_dim = full_dataset.advanced_text_features.shape[1]
    print(f"Advanced feature dimension: {advanced_feature_dim}")
    
    # Split dataset
    from torch.utils.data import random_split
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=64,  # Larger batch size since model is lighter
        shuffle=True,
        num_workers=8,
        collate_fn=hybrid_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        collate_fn=hybrid_collate_fn,
        pin_memory=True
    )
    
    # Initialize hybrid model
    model = HybridPriceModel(
        advanced_feature_dim=advanced_feature_dim,
        hidden_dim=512,
        num_heads=8,
        dropout=0.2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train the model
    trained_model = train_hybrid_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        price_stats=price_stats,
        num_epochs=16,
        learning_rate=5e-5,
        device=device,
        save_path='hybrid_grocery_price_model.pth'
    )
    
    # Save text extractor for inference
    joblib.dump(text_extractor, 'text_extractor.pkl')
    print("✅ Text extractor saved for inference")
