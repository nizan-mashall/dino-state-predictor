import os
import time
import h5py
import numpy as np
from PIL import Image
 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import DinoMlp
 
# Same quantile bounds used in the OpenVLA fine-tuning script
Q01 = np.array([-0.2737, -0.2854, -0.8444, -0.0562, -0.0814, -0.1755, -1.0], dtype=np.float32)
Q99 = np.array([ 0.7964,  0.3604,  1.0,     0.0558,  0.1694,  0.2551,  1.0], dtype=np.float32)
 
def normalize_action(action: np.ndarray) -> np.ndarray:
    """Map raw action to [-1, 1] using dataset quantiles."""
    a = 2.0 * (action - Q01) / (Q99 - Q01 + 1e-8) - 1.0
    return np.clip(a, -1.0, 1.0).astype(np.float32)
 
# DINOv2 expects ImageNet-normalised 224×224 tensors
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
 
 
class RoboSuiteDataset(Dataset):
    """
    Reads agentview RGB frames, end-effector state, and actions from an
    HDF5 file produced by RoboSuite / robomimic.
 
    Robot state = eef_pos (3) + eef_quat (4)  →  7-dim vector
    Target      = action (7)  normalised to [-1, 1]
 
    Set max_demos=None to use every demo in the file.
    """
 
    def __init__(self, hdf5_path: str, max_demos: int | None = 20):
        self.hdf5_path = hdf5_path
        self.f = h5py.File(hdf5_path, 'r')
 
        all_demos = list(self.f['data'].keys())
        self.demos = all_demos[:max_demos] if max_demos is not None else all_demos
 
        # Build a flat index: (demo_key, timestep)
        self.index = []
        for demo_key in self.demos:
            T = len(self.f['data'][demo_key]['actions'])
            for t in range(T):
                self.index.append((demo_key, t))
 
        print(f"[Dataset] {len(self.demos)} demos | {len(self.index)} timesteps")
 
    def __len__(self):
        return len(self.index)
 
    def __getitem__(self, idx):
        demo_key, t = self.index[idx]
        demo = self.f['data'][demo_key]
 
        # ── image ──────────────────────────────────────────────────────────
        raw_img = demo['obs']['agentview_image'][t]          # (H, W, 3) uint8
        image = IMAGE_TRANSFORM(Image.fromarray(raw_img.astype(np.uint8)))
 
        # ── robot state ────────────────────────────────────────────────────
        eef_pos  = demo['obs']['robot0_eef_pos'][t].astype(np.float32)   # (3,)
        eef_quat = demo['obs']['robot0_eef_quat'][t].astype(np.float32)  # (4,)
        state = torch.from_numpy(np.concatenate([eef_pos, eef_quat]))    # (7,)
 
        # ── action (normalised) ────────────────────────────────────────────
        raw_action = demo['actions'][t].astype(np.float32)
        action = torch.from_numpy(normalize_action(raw_action))           # (7,)
 
        return image, state, action
 
 
# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
 
def train(
    hdf5_path:    str  = '/users/ogal/nmashall/dino-state-predictor/demo_with_images.hdf5',
    checkpoint_dir: str = '/users/ogal/nmashall/dino-state-predictor/checkpoints_dino',
    max_demos:    int  = 20,
    epochs:       int  = 30,
    batch_size:   int  = 32,
    lr:           float = 1e-3,
    hidden_dims: list  = None,
    num_workers:  int  = 4,
):
    if hidden_dims is None:
        hidden_dims = [256, 256]
 
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device('cpu')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] device={device}")
 
    # ── dataset / dataloader ───────────────────────────────────────────────
    dataset = RoboSuiteDataset(hdf5_path, max_demos=max_demos)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
    )
 
    # ── model ──────────────────────────────────────────────────────────────
    model = DinoMlp(output_dim=7, hidden_dims=hidden_dims).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[Model] trainable params: {trainable:,} / {total:,}")
 
    # ── optimiser + scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = nn.L1Loss()
 
    # ── training loop ──────────────────────────────────────────────────────
    print("[Train] starting …")
    start = time.time()
 
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
 
        for batch_idx, (images, states, actions) in enumerate(dataloader):
            images  = images.to(device)                        # (B, 3, 224, 224)
            states  = states.to(device)                        # (B, 7)
            actions = actions.to(device)                       # (B, 7)
 
            preds = model(states, images)                      # (B, 7)
            loss  = criterion(preds, actions)
 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
 
            total_loss += loss.item()
 
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch:>3}] avg L1 loss: {avg_loss:.4f} "
              f"| lr: {scheduler.get_last_lr()[0]:.2e}")
 
        # save checkpoint every 5 epochs and at the end
        if epoch % 5 == 0 or epoch == epochs:
            ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pt')
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'avg_loss':    avg_loss,
                'hidden_dims': hidden_dims,
            }, ckpt_path)
            print(f"  → checkpoint saved: {ckpt_path}")
 
    total_time = time.time() - start
    print(f"[Train] done in {total_time/60:.1f} min")
 
 
# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
 
if __name__ == '__main__':
    train(
        hdf5_path='/users/ogal/nmashall/dino-state-predictor/demo_with_images.hdf5',
        checkpoint_dir='/users/ogal/nmashall/dino-state-predictor/checkpoints_dino',
        max_demos=200,
        epochs=30,
        batch_size=32,
        lr=1e-3,
        hidden_dims=[256, 256],
        num_workers=4,
    )