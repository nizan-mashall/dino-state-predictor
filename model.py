import torch
import torch.nn as nn
import numpy as np

class DinoMlp(nn.Module):
    def __init__(self, output_dim, hidden_dims = [64, 64]):
        super().__init__()

        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for param in self.dino.parameters():
            param.requires_grad = False  # freeze — don't train the backbone
        
        dino_output_dim = 384  # ViT-S/14 outputs 384-dim, ViT-B/14 → 768
        robot_DoF = 7  # 3 for position, 4 for orientation (quaternion)
        input_dim = dino_output_dim + robot_DoF

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, x, image):
        # image: (batch, 3, 224, 224)
        with torch.no_grad():
            image_features = self.dino(image)  # → (batch, 384)
        input_features = torch.cat([x, image_features], dim=1)  # → (batch, input_dim)
        action = self.network(input_features)      # → (batch, output_dim)
        return action
    
if __name__ == "__main__":
    model = DinoMlp(output_dim=7, hidden_dims=[256, 256])

    image = torch.randn(4, 3, 224, 224)  # batch of 4
    state = torch.randn(4, 7)            # robot state

    action = model(state, image)
    print(action.shape)   # → torch.Size([4, 7])
    print(action.min(), action.max())  # → between -1 and 1