"""
Two-branch fusion model for price prediction
Combines CLIP image embeddings with Qwen text embeddings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBranch(nn.Module):
    """MLP branch for processing a single modality"""
    
    def __init__(self, layer_dims, dropout=0.2, use_layer_norm=True):
        """
        Args:
            layer_dims: List of dimensions [input_dim, hidden1, ..., output_dim]
            dropout: Dropout probability
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            
            if use_layer_norm and i < len(layer_dims) - 2:
                layers.append(nn.LayerNorm(layer_dims[i+1]))
            
            if i < len(layer_dims) - 2:
                layers.append(nn.GELU())
                
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class CrossAttentionFusion(nn.Module):
    """Cross-attention between image and text modalities"""
    
    def __init__(self, fused_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = fused_dim // num_heads
        assert fused_dim % num_heads == 0, "fused_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(fused_dim, fused_dim)
        self.k_proj = nn.Linear(fused_dim, fused_dim)
        self.v_proj = nn.Linear(fused_dim, fused_dim)
        self.out_proj = nn.Linear(fused_dim, fused_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(fused_dim)
        
    def forward(self, fused):
        """
        Args:
            fused: (batch, fused_dim) - concatenated embeddings
        Returns:
            output: (batch, fused_dim) - attention-modulated features
        """
        batch_size = fused.size(0)
        
        Q = self.q_proj(fused)
        K = self.k_proj(fused)
        V = self.v_proj(fused)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(0, 1).contiguous().view(batch_size, -1)
        
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        output = self.layer_norm(fused + output)
        
        return output


class GatedMoERegressionHead(nn.Module):
    """Gated Mixture-of-Experts regression head"""
    
    def __init__(self, input_dim, num_experts=3, hidden_dims=[256, 128], dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], num_experts),
            nn.Softmax(dim=-1)
        )
        
        self.experts = nn.ModuleList([
            MLPBranch(
                layer_dims=[input_dim] + hidden_dims + [1],
                dropout=dropout,
                use_layer_norm=True
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim)
        Returns:
            output: (batch,) - weighted sum of expert predictions
            gate_weights: (batch, num_experts) - gating weights
        """
        gate_weights = self.gate(x)
        expert_outputs = torch.stack([expert(x).squeeze(-1) for expert in self.experts], dim=1)
        output = torch.sum(gate_weights * expert_outputs, dim=1)
        
        return output, gate_weights


class TwoBranchFusionModel(nn.Module):
    """
    Two-branch fusion model: CLIP image + Qwen text (with structural features)
    
    Architecture:
        - Image branch: Process CLIP image embeddings (768 → 512)
        - Text branch: Process Qwen text embeddings (1027 → 512)
        - Fusion: Concatenate outputs (1024 dims)
        - Regression head: Final prediction
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.predict_log = config.PREDICT_LOG
        
        # Learnable scaling parameters
        self.image_scale = nn.Parameter(torch.ones(1))
        self.text_scale = nn.Parameter(torch.ones(1))
        
        # Branch networks
        self.image_branch = MLPBranch(
            layer_dims=config.IMAGE_BRANCH_DIMS,
            dropout=config.DROPOUT_RATE,
            use_layer_norm=config.USE_LAYER_NORM
        )
        
        self.text_branch = MLPBranch(
            layer_dims=config.TEXT_BRANCH_DIMS,
            dropout=config.DROPOUT_RATE,
            use_layer_norm=config.USE_LAYER_NORM
        )
        
        # Regression head
        self.regression_head = MLPBranch(
            layer_dims=config.REGRESSION_HEAD_DIMS,
            dropout=config.DROPOUT_RATE / 2,
            use_layer_norm=config.USE_LAYER_NORM
        )
        
        # Optional residual connection
        if config.USE_RESIDUAL:
            self.residual_proj = nn.Linear(config.FUSION_DIM, 1)
    
    def forward(self, image_emb, text_emb):
        """
        Args:
            image_emb: (batch, 768) - CLIP image embeddings
            text_emb: (batch, 1027) - Qwen text embeddings
            
        Returns:
            predictions: (batch,) - predicted prices
        """
        # Normalize embeddings
        image_emb = F.normalize(image_emb, p=2, dim=-1)
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        
        # Apply learnable scaling
        image_emb = image_emb * self.image_scale
        text_emb = text_emb * self.text_scale
        
        # Process each modality
        image_features = self.image_branch(image_emb)
        text_features = self.text_branch(text_emb)
        
        # Fusion: Concatenate
        fused = torch.cat([image_features, text_features], dim=-1)
        
        # Regression head
        output = self.regression_head(fused).squeeze(-1)
        
        # Optional residual
        if hasattr(self, 'residual_proj'):
            residual = self.residual_proj(fused).squeeze(-1)
            output = output + residual
        
        return output
    
    def predict_price(self, image_emb, text_emb):
        """Make price predictions in original scale"""
        output = self.forward(image_emb, text_emb)
        
        if self.predict_log:
            prices = torch.expm1(output)
            prices = torch.clamp(prices, min=0.0)
        else:
            prices = torch.clamp(output, min=0.0)
        
        return prices


class AdvancedMoEFusionModel(nn.Module):
    """
    Advanced two-branch fusion with cross-attention and MoE
    
    Architecture:
        - Image/Text branches
        - Cross-attention fusion
        - Gated MoE regression head
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.predict_log = config.PREDICT_LOG
        
        # Learnable scaling
        self.image_scale = nn.Parameter(torch.ones(1))
        self.text_scale = nn.Parameter(torch.ones(1))
        
        # Branch networks
        self.image_branch = MLPBranch(
            layer_dims=config.IMAGE_BRANCH_DIMS,
            dropout=config.DROPOUT_RATE,
            use_layer_norm=config.USE_LAYER_NORM
        )
        
        self.text_branch = MLPBranch(
            layer_dims=config.TEXT_BRANCH_DIMS,
            dropout=config.DROPOUT_RATE,
            use_layer_norm=config.USE_LAYER_NORM
        )
        
        # Cross-attention fusion
        num_heads = getattr(config, 'NUM_ATTENTION_HEADS', 8)
        self.cross_attention = CrossAttentionFusion(
            fused_dim=config.FUSION_DIM,
            num_heads=num_heads,
            dropout=config.DROPOUT_RATE
        )
        
        # Gated MoE regression head
        num_experts = getattr(config, 'NUM_EXPERTS', 3)
        expert_hidden_dims = getattr(config, 'EXPERT_HIDDEN_DIMS', [256, 128])
        
        self.regression_head = GatedMoERegressionHead(
            input_dim=config.FUSION_DIM,
            num_experts=num_experts,
            hidden_dims=expert_hidden_dims,
            dropout=config.DROPOUT_RATE / 2
        )
    
    def forward(self, image_emb, text_emb):
        """
        Args:
            image_emb: (batch, 768)
            text_emb: (batch, 1027)
            
        Returns:
            predictions: (batch,)
            gate_weights: (batch, num_experts)
        """
        # Normalize embeddings
        image_emb = F.normalize(image_emb, p=2, dim=-1)
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        
        # Apply learnable scaling
        image_emb = image_emb * self.image_scale
        text_emb = text_emb * self.text_scale
        
        # Process through branches
        image_features = self.image_branch(image_emb)
        text_features = self.text_branch(text_emb)
        
        # Initial fusion
        fused = torch.cat([image_features, text_features], dim=-1)
        
        # Cross-attention
        modulated_fused = self.cross_attention(fused)
        
        # Gated MoE regression
        output, gate_weights = self.regression_head(modulated_fused)
        
        return output, gate_weights
    
    def predict_price(self, image_emb, text_emb):
        """Make price predictions in original scale"""
        output, gate_weights = self.forward(image_emb, text_emb)
        
        if self.predict_log:
            prices = torch.expm1(output)
            prices = torch.clamp(prices, min=0.0)
        else:
            prices = torch.clamp(output, min=0.0)
        
        return prices, gate_weights


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.append('/home/utkarsh/amazon_ml_2025/fusion_qwen')
    from config import Config
    
    config = Config()
    
    # Create model
    model = TwoBranchFusionModel(config)
    print(f"Base Model: {count_parameters(model):,} trainable parameters")
    
    # Test forward pass
    batch_size = 32
    image_emb = torch.randn(batch_size, 768)
    text_emb = torch.randn(batch_size, 1027)
    
    with torch.no_grad():
        output = model(image_emb, text_emb)
        print(f"Output shape: {output.shape}")
        
        prices = model.predict_price(image_emb, text_emb)
        print(f"Predicted prices shape: {prices.shape}")
        print(f"Price range: [{prices.min():.2f}, {prices.max():.2f}]")
    
    # Test advanced model
    print("\n" + "="*60)
    adv_model = AdvancedMoEFusionModel(config)
    print(f"Advanced Model: {count_parameters(adv_model):,} trainable parameters")
    
    with torch.no_grad():
        output, gate_weights = adv_model(image_emb, text_emb)
        print(f"Output shape: {output.shape}")
        print(f"Gate weights shape: {gate_weights.shape}")

