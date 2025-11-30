"""
Four-branch fusion model: CLIP image, SigLIP image, SigLIP text, Qwen text
Simple single-layer branches
"""
import torch
import torch.nn as nn


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLPBranch(nn.Module):
    """Simple MLP branch with single hidden layer"""
    
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2, use_layer_norm=True, use_residual=False):
        super().__init__()
        
        layers = []
        dims = [input_dim] + list(hidden_dims)
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(dims[i+1]))
            
            if i < len(dims) - 2:  # No activation on last layer
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout_rate))
        
        self.network = nn.Sequential(*layers)
        self.use_residual = use_residual and (input_dim == hidden_dims[-1])
        
        if self.use_residual and input_dim != hidden_dims[-1]:
            self.residual_proj = nn.Linear(input_dim, hidden_dims[-1])
        else:
            self.residual_proj = None
    
    def forward(self, x):
        out = self.network(x)
        
        if self.use_residual:
            if self.residual_proj is not None:
                x = self.residual_proj(x)
            out = out + x
        
        return out


class FourBranchFusionModel(nn.Module):
    """
    Four-branch fusion model with simple concatenation
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Four branches: CLIP image, SigLIP image, SigLIP text, Qwen text
        self.clip_image_branch = MLPBranch(
            config.CLIP_IMAGE_DIM,
            config.CLIP_IMAGE_BRANCH_DIMS[1:],
            config.DROPOUT_RATE,
            config.USE_LAYER_NORM,
            config.USE_RESIDUAL
        )
        
        self.siglip_image_branch = MLPBranch(
            config.SIGLIP_IMAGE_DIM,
            config.SIGLIP_IMAGE_BRANCH_DIMS[1:],
            config.DROPOUT_RATE,
            config.USE_LAYER_NORM,
            config.USE_RESIDUAL
        )
        
        self.siglip_text_branch = MLPBranch(
            config.SIGLIP_TEXT_DIM,
            config.SIGLIP_TEXT_BRANCH_DIMS[1:],
            config.DROPOUT_RATE,
            config.USE_LAYER_NORM,
            config.USE_RESIDUAL
        )
        
        self.qwen_text_branch = MLPBranch(
            config.QWEN_TEXT_DIM,
            config.QWEN_TEXT_BRANCH_DIMS[1:],
            config.DROPOUT_RATE,
            config.USE_LAYER_NORM,
            config.USE_RESIDUAL
        )
        
        # Regression head
        self.regression_head = MLPBranch(
            config.FUSION_DIM,
            config.REGRESSION_HEAD_DIMS[1:],
            config.DROPOUT_RATE,
            use_layer_norm=False,
            use_residual=False
        )
    
    def forward(self, clip_image, siglip_image, siglip_text, qwen_text):
        """
        Args:
            clip_image: [batch, 768]
            siglip_image: [batch, 1152]
            siglip_text: [batch, 768]
            qwen_text: [batch, 1027]
        
        Returns:
            predictions: [batch, 1]
        """
        # Process each branch
        clip_feat = self.clip_image_branch(clip_image)          # [batch, 512]
        siglip_img_feat = self.siglip_image_branch(siglip_image)  # [batch, 512]
        siglip_txt_feat = self.siglip_text_branch(siglip_text)    # [batch, 512]
        qwen_feat = self.qwen_text_branch(qwen_text)            # [batch, 512]
        
        # Concatenate all branches
        fused = torch.cat([clip_feat, siglip_img_feat, siglip_txt_feat, qwen_feat], dim=-1)  # [batch, 2048]
        
        # Regression
        output = self.regression_head(fused)  # [batch, 1]
        
        return output.squeeze(-1)  # [batch]


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion layer"""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key_value):
        """
        Args:
            query: [batch, dim]
            key_value: [batch, dim]
        
        Returns:
            attended: [batch, dim]
        """
        # Add sequence dimension
        q = query.unsqueeze(1)  # [batch, 1, dim]
        kv = key_value.unsqueeze(1)  # [batch, 1, dim]
        
        # Cross-attention
        attn_out, _ = self.multihead_attn(q, kv, kv)  # [batch, 1, dim]
        attn_out = attn_out.squeeze(1)  # [batch, dim]
        
        # Residual + norm
        out = self.norm(query + self.dropout(attn_out))
        
        return out


class GatedMoERegressionHead(nn.Module):
    """Mixture of Experts regression head with gating"""
    
    def __init__(self, input_dim, hidden_dims, num_experts=3, dropout=0.2):
        super().__init__()
        self.num_experts = num_experts
        
        # Create experts
        self.experts = nn.ModuleList([
            MLPBranch(input_dim, hidden_dims + [1], dropout, use_layer_norm=False)
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, input_dim]
        
        Returns:
            output: [batch, 1]
            gates: [batch, num_experts]
        """
        # Get gating weights
        gates = self.gate(x)  # [batch, num_experts]
        
        # Get expert predictions
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # [batch, 1, num_experts]
        
        # Weighted combination
        output = torch.sum(expert_outputs * gates.unsqueeze(1), dim=-1)  # [batch, 1]
        
        return output, gates


class AdvancedMoEFusionModel(nn.Module):
    """
    Four-branch fusion model with cross-attention and MoE
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Four branches
        self.clip_image_branch = MLPBranch(
            config.CLIP_IMAGE_DIM,
            config.CLIP_IMAGE_BRANCH_DIMS[1:],
            config.DROPOUT_RATE,
            config.USE_LAYER_NORM,
            config.USE_RESIDUAL
        )
        
        self.siglip_image_branch = MLPBranch(
            config.SIGLIP_IMAGE_DIM,
            config.SIGLIP_IMAGE_BRANCH_DIMS[1:],
            config.DROPOUT_RATE,
            config.USE_LAYER_NORM,
            config.USE_RESIDUAL
        )
        
        self.siglip_text_branch = MLPBranch(
            config.SIGLIP_TEXT_DIM,
            config.SIGLIP_TEXT_BRANCH_DIMS[1:],
            config.DROPOUT_RATE,
            config.USE_LAYER_NORM,
            config.USE_RESIDUAL
        )
        
        self.qwen_text_branch = MLPBranch(
            config.QWEN_TEXT_DIM,
            config.QWEN_TEXT_BRANCH_DIMS[1:],
            config.DROPOUT_RATE,
            config.USE_LAYER_NORM,
            config.USE_RESIDUAL
        )
        
        # Cross-attention fusion layers
        branch_dim = config.CLIP_IMAGE_BRANCH_DIMS[-1]  # All branches output 512
        
        # Image-to-image fusion
        self.image_fusion = CrossAttentionFusion(
            dim=branch_dim,
            num_heads=config.NUM_ATTENTION_HEADS,
            dropout=config.DROPOUT_RATE
        )
        
        # Text-to-text fusion
        self.text_fusion = CrossAttentionFusion(
            dim=branch_dim,
            num_heads=config.NUM_ATTENTION_HEADS,
            dropout=config.DROPOUT_RATE
        )
        
        # Image-text fusion
        self.image_text_fusion = CrossAttentionFusion(
            dim=branch_dim,
            num_heads=config.NUM_ATTENTION_HEADS,
            dropout=config.DROPOUT_RATE
        )
        
        # MoE Regression head
        self.regression_head = GatedMoERegressionHead(
            input_dim=config.FUSION_DIM,
            hidden_dims=config.EXPERT_HIDDEN_DIMS,
            num_experts=config.NUM_EXPERTS,
            dropout=config.DROPOUT_RATE
        )
    
    def forward(self, clip_image, siglip_image, siglip_text, qwen_text):
        """
        Args:
            clip_image: [batch, 768]
            siglip_image: [batch, 1152]
            siglip_text: [batch, 768]
            qwen_text: [batch, 1027]
        
        Returns:
            predictions: [batch]
            gates: [batch, num_experts]
        """
        # Process branches
        clip_feat = self.clip_image_branch(clip_image)          # [batch, 512]
        siglip_img_feat = self.siglip_image_branch(siglip_image)  # [batch, 512]
        siglip_txt_feat = self.siglip_text_branch(siglip_text)    # [batch, 512]
        qwen_feat = self.qwen_text_branch(qwen_text)            # [batch, 512]
        
        # Image-to-image cross-attention (CLIP ← SigLIP)
        fused_image = self.image_fusion(clip_feat, siglip_img_feat)
        
        # Text-to-text cross-attention (Qwen ← SigLIP)
        fused_text = self.text_fusion(qwen_feat, siglip_txt_feat)
        
        # Image-text cross-attention (Image ← Text)
        fused_multimodal = self.image_text_fusion(fused_image, fused_text)
        
        # Concatenate all features
        fused = torch.cat([
            fused_multimodal,  # Primary fused features
            fused_image,       # Image-only features
            fused_text,        # Text-only features
            clip_feat          # Original CLIP features
        ], dim=-1)  # [batch, 2048]
        
        # MoE regression
        output, gates = self.regression_head(fused)  # [batch, 1], [batch, num_experts]
        
        return output.squeeze(-1), gates  # [batch], [batch, num_experts]


# Test
if __name__ == "__main__":
    from config import Config
    
    config = Config()
    
    print("Testing FourBranchFusionModel...")
    model = FourBranchFusionModel(config)
    print(f"  Parameters: {count_parameters(model):,}")
    
    batch = 4
    clip_img = torch.randn(batch, 768)
    siglip_img = torch.randn(batch, 1152)
    siglip_txt = torch.randn(batch, 768)
    qwen_txt = torch.randn(batch, 1027)
    
    output = model(clip_img, siglip_img, siglip_txt, qwen_txt)
    print(f"  Output shape: {output.shape}")
    
    print("\nTesting AdvancedMoEFusionModel...")
    model_adv = AdvancedMoEFusionModel(config)
    print(f"  Parameters: {count_parameters(model_adv):,}")
    
    output, gates = model_adv(clip_img, siglip_img, siglip_txt, qwen_txt)
    print(f"  Output shape: {output.shape}")
    print(f"  Gates shape: {gates.shape}")
    
    print("\n✓ All tests passed!")
