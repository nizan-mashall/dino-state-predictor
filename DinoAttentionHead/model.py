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


class DinoTransformerHead(nn.Module):
    """
    Single-action prediction with a full encoder-decoder transformer
    over DINO patch tokens — now with self-attention among image/state
    tokens (encoder) before cross-attention to the action query (decoder).
    """

    def __init__(
        self,
        action_dim=7,
        state_dim=7,
        hidden_dim=256,
        n_heads=8,
        n_enc_layers=2,    # ← self-attention layers, NEW
        n_dec_layers=2,    # ← cross-attention layers
    ):
        super().__init__()
        self.action_dim = action_dim

        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for p in self.dino.parameters():
            p.requires_grad = False
        dino_dim = 384

        self.patch_proj = nn.Linear(dino_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        self.num_patches = 256
        self.pos_embed_cam = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim) * 0.02)

        # ── NEW: transformer ENCODER — self-attention among memory tokens ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)

        # ── transformer DECODER — cross-attention from query to memory ─────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_dec_layers)

        self.action_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def _extract_tokens(self, image):
        with torch.no_grad():
            out = self.dino.forward_features(image)
            patches = out['x_norm_patchtokens']
        return self.patch_proj(patches) + self.pos_embed_cam

    def forward(self, state, image1, image2):
        tok1 = self._extract_tokens(image1)              # (B, 256, H)
        tok2 = self._extract_tokens(image2)               # (B, 256, H)
        state_tok = self.state_proj(state).unsqueeze(1)   # (B, 1, H)

        memory = torch.cat([tok1, tok2, state_tok], dim=1)  # (B, 513, H)

        # NEW: self-attention — tokens attend to each other first
        memory = self.encoder(memory)                       # (B, 513, H)

        B = memory.shape[0]
        query = self.action_query.expand(B, -1, -1)          # (B, 1, H)

        # cross-attention — query attends to the self-attended memory
        decoded = self.decoder(tgt=query, memory=memory)     # (B, 1, H)

        action = self.action_head(decoded.squeeze(1))
        return action


if __name__ == "__main__":
    model = DinoTransformerHead(action_dim=7, n_enc_layers=2, n_dec_layers=2)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,}")

    B = 4
    image1 = torch.randn(B, 3, 224, 224)
    image2 = torch.randn(B, 3, 224, 224)
    state = torch.randn(B, 7)

    action = model(state, image1, image2)
    print(action.shape)   # (4, 7)