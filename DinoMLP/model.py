import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

if not hasattr(F, 'scaled_dot_product_attention'):
    def _scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        scale = query.size(-1) ** -0.5
        attn = torch.matmul(query, key.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = torch.softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)
        return torch.matmul(attn, value)
    F.scaled_dot_product_attention = _scaled_dot_product_attention

class DinoMlp(nn.Module):
    def __init__(self, output_dim, hidden_dims = [64, 64]):
        super().__init__()

        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        for param in self.dino.parameters():
            param.requires_grad = False  # freeze — don't train the backbone
        
        dino_CLS_dim = 768  # ViT-S/14 outputs 384-dim, ViT-B/14 → 768
        dino_patch_dim = 768
        nums_of_cameras = 2
        robot_DoF = 7  # 3 for position, 4 for orientation (quaternion)
        input_dim = (dino_CLS_dim + dino_patch_dim) * nums_of_cameras + robot_DoF

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Dropout(0.1))
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        #layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
    
    def _extract(self, image):
        """Run DINO and return CLS + mean-pooled patch tokens concatenated."""
        out        = self.dino.forward_features(image)   # dict
        cls        = out['x_norm_clstoken']              # (B, 384)
        patches    = out['x_norm_patchtokens']           # (B, 256, 384)
        spatial    = patches.mean(dim=1)                 # (B, 384)
        return torch.cat([cls, spatial], dim=-1)         # (B, 768)

    def forward(self, x, image1, image2):
        # image: (batch, 3, 224, 224)
        with torch.no_grad():
            image1_features = self._extract(image1)  # → (batch, 384)
            image2_features = self._extract(image2)  # → (batch, 384)
        input_features = torch.cat([x, image1_features, image2_features], dim=1)  # → (batch, input_dim)
        action = self.network(input_features)      # → (batch, output_dim)
        return action