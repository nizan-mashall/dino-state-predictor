import os
import time
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from model import DinoTransformerHead

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class RoboSuiteDataset(Dataset):
    def __init__(self, hdf5_path: str, max_demos: int | None = 20):
        self.hdf5_path = hdf5_path
        self.f = h5py.File(hdf5_path, 'r')

        all_demos = list(self.f['data'].keys())
        self.demos = all_demos[:max_demos] if max_demos is not None else all_demos

        self.index = []
        for demo_key in self.demos:
            T = len(self.f['data'][demo_key]['actions'])
            for t in range(T):
                self.index.append((demo_key, t))

        print(f"[Dataset] {len(self.demos)} demos | {len(self.index)} timesteps", flush=True)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        demo_key, t = self.index[idx]
        demo = self.f['data'][demo_key]

        raw_img1 = demo['obs']['robot0_eye_in_hand_image'][t]
        image1 = IMAGE_TRANSFORM(Image.fromarray(raw_img1.astype(np.uint8)))

        raw_img2 = demo['obs']['agentview_image'][t]
        image2 = IMAGE_TRANSFORM(Image.fromarray(raw_img2.astype(np.uint8)))

        eef_pos  = demo['obs']['robot0_eef_pos'][t].astype(np.float32)
        eef_quat = demo['obs']['robot0_eef_quat'][t].astype(np.float32)
        state = torch.from_numpy(np.concatenate([eef_pos, eef_quat]))

        raw_action = demo['actions'][t].astype(np.float32)
        action = torch.from_numpy(raw_action)
        return image1, image2, state, action


def train(
    hdf5_path:      str   = '/content/drive/MyDrive/SAIL Lab/demo_with_images.hdf5',
    checkpoint_dir: str   = '/content/ckp_w_val',
    max_demos:      int   = 200,
    epochs:         int   = 30,
    batch_size:     int   = 32,
    lr:             float = 1e-3,
    hidden_dim:     int   = 256,     # ← renamed: transformer uses one hidden_dim, not a list
    n_enc_layers:   int   = 2,       # ← new: self-attention layers
    n_dec_layers:   int   = 2,       # ← new: cross-attention layers
    n_heads:        int   = 8,       # ← new
    num_workers:    int   = 4,
    val_split:      float = 0.2,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] device={device}", flush=True)

    full_dataset = RoboSuiteDataset(hdf5_path, max_demos=max_demos)
    total      = len(full_dataset)
    val_size   = int(total * val_split)
    train_size = total - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"[Split] {train_size} train | {val_size} val", flush=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
    )

    # ── model — DinoTransformerHead, not DinoMlp ────────────────────────────
    model = DinoTransformerHead(
        action_dim=7,
        state_dim=7,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_enc_layers=n_enc_layers,
        n_dec_layers=n_dec_layers,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"[Model] trainable params: {trainable:,} / {total_p:,}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.L1Loss()

    print("[Train] starting ...", flush=True)
    print(f"{'Epoch':>6} | {'Train L1':>10} | {'Val L1':>10} | {'Gap':>8} | {'LR':>10}", flush=True)
    print("-" * 58, flush=True)

    start = time.time()
    best_val_loss = float('inf')

    epochs_history     = []
    train_loss_history = []
    val_loss_history   = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images1, images2, states, actions in train_loader:
            images1 = images1.to(device)
            images2 = images2.to(device)
            states  = states.to(device)
            actions = actions.to(device)

            preds = model(states, images1, images2)   # same call signature as DinoMlp
            loss  = criterion(preds, actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images1, images2, states, actions in val_loader:
                images1 = images1.to(device)
                images2 = images2.to(device)
                states  = states.to(device)
                actions = actions.to(device)
                preds   = model(states, images1, images2)
                val_loss += criterion(preds, actions).item()

        avg_val_loss = val_loss / len(val_loader)
        gap = avg_val_loss - avg_train_loss

        scheduler.step()

        flag = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            flag = " <- best"
            torch.save({
                'epoch':         epoch,
                'model_state':   model.state_dict(),
                'optim_state':   optimizer.state_dict(),
                'train_loss':    avg_train_loss,
                'val_loss':      avg_val_loss,
                'hidden_dim':    hidden_dim,     # ← updated keys
                'n_enc_layers':  n_enc_layers,
                'n_dec_layers':  n_dec_layers,
                'n_heads':       n_heads,
            }, os.path.join(checkpoint_dir, 'best.pt'))

        elapsed = time.time() - start
        print(
            f"{epoch:>6} | {avg_train_loss:>10.4f} | {avg_val_loss:>10.4f} | "
            f"{gap:>+8.4f} | {scheduler.get_last_lr()[0]:>10.2e}{flag}  [{elapsed:.0f}s]",
            flush=True
        )

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        epochs_history.append(epoch)

        if epoch % 5 == 0 or epoch == epochs:
            ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pt')
            torch.save({
                'epoch':         epoch,
                'model_state':   model.state_dict(),
                'optim_state':   optimizer.state_dict(),
                'train_loss':    avg_train_loss,
                'val_loss':      avg_val_loss,
                'hidden_dim':    hidden_dim,
                'n_enc_layers':  n_enc_layers,
                'n_dec_layers':  n_dec_layers,
                'n_heads':       n_heads,
            }, ckpt_path)
            print(f"  -> checkpoint saved: {ckpt_path}", flush=True)

    total_time = time.time() - start
    print(f"[Train] done in {total_time/60:.1f} min", flush=True)
    print(f"[Train] best val loss: {best_val_loss:.4f}", flush=True)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_history, train_loss_history, label='Train Loss', marker='o')
    plt.plot(epochs_history, val_loss_history, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(checkpoint_dir, 'loss_plot.png'))


if __name__ == '__main__':
    train(
        hdf5_path='/users/ogal/nmashall/dino-state-predictor/demo_with_images.hdf5',
        checkpoint_dir='/users/ogal/nmashall/dino-state-predictor/DinoAttentionHead/ckp',
        max_demos=200,
        epochs=100,
        batch_size=32,        # ← lowered from 32, see note below
        lr=3e-4,
        hidden_dim=128,
        n_enc_layers=1,
        n_dec_layers=1,
        n_heads=4,
        num_workers=4,
        val_split=0.1,
    )