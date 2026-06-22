import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import DinoMlp
import robosuite as suite
import imageio
from robosuite.utils.placement_samplers import UniformRandomSampler

# These are the same bounds from tinybc_train.py
Q01 = np.array([-0.2737, -0.2854, -0.8444, -0.0562, -0.0814, -0.1755, -1.0], dtype=np.float32)
Q99 = np.array([ 0.7964,  0.3604,  1.0,     0.0558,  0.1694,  0.2551,  1.0], dtype=np.float32)

def unnormalize_action(action: np.ndarray) -> np.ndarray:
    """Map model output [-1, 1] back to original action space."""
    return 0.5 * (action + 1.0) * (Q99 - Q01 + 1e-8) + Q01

# ── same transform used during training ────────────────────────────────────
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── config ─────────────────────────────────────────────────────────────────
CHECKPOINT = '/users/ogal/nmashall/dino-state-predictor/checkpoints_colab/eye_in_hand_and_agentview.pt' # 200 demos eye in hand + agentview
#CHECKPOINT = '/users/ogal/nmashall/dino-state-predictor/checkpoints_colab/eye_in_hand.pt' #200 demos eye in hand
#CHECKPOINT = '/users/ogal/nmashall/dino-state-predictor/checkpoints_colab/epoch_30.pt' #200 demos agentview
#CHECKPOINT = '/users/ogal/nmashall/dino-state-predictor/checkpoints_dino/epoch_30.pt' #20 demos
VIDEO_PATH  = '/users/ogal/nmashall/dino-state-predictor/videos/eval_video.mp4'
NUM_STEPS   = 100
HIDDEN_DIMS = [256, 256]

# ── load model ─────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Eval] device={device}", flush=True)

model = DinoMlp(output_dim=7, hidden_dims=HIDDEN_DIMS).to(device)
ckpt  = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"[Eval] loaded checkpoint from epoch {ckpt['epoch']} "
      f"(val_loss={ckpt.get('val_loss', 'N/A')})", flush=True)

# ── task setup ────────────────────────────────────────────────────────────────
cube_sampler = UniformRandomSampler(
    name="ObjectSampler",
    x_range=[-0.1, 0.1],   # how far left/right the cube can spawn
    y_range=[-0.1, 0.1],   # how far forward/back
    rotation=None,          # None = random rotation, or set fixed angle
    ensure_object_boundary_in_range=False,
)

# ── environment ────────────────────────────────────────────────────────────
env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["agentview", "robot0_eye_in_hand"],
    camera_heights=256,
    camera_widths=256,
    #initialization_noise={"magnitude": 0.1, "type": "gaussian"}
    initialization_noise=None,           # ← robot starts at fixed position
    placement_initializer=cube_sampler,  # ← cube spawns randomly
)

score = 0 
trials = 10
frames = []

for trial in range(trials):

    obs = env.reset()
    was_success = False
    # ── evaluation loop ────────────────────────────────────────────────────────
    for step in range(NUM_STEPS):

        # save frame (flipped — robosuite images are upside down)
        frame = np.flipud(obs["agentview_image"])
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        frames.append(frame)

        # preprocess image for model (no flip — use original)
        
        pil_img1    = Image.fromarray(obs["robot0_eye_in_hand_image"].astype(np.uint8))
        img_tensor1 = IMAGE_TRANSFORM(pil_img1).unsqueeze(0).to(device)  # (1, 3, 224, 224)

        pil_img2    = Image.fromarray(obs["agentview_image"].astype(np.uint8))
        img_tensor2 = IMAGE_TRANSFORM(pil_img2).unsqueeze(0).to(device)  # (1, 3, 224, 224)

        # preprocess state
        eef_pos  = obs['robot0_eef_pos'].astype(np.float32)
        eef_quat = obs['robot0_eef_quat'].astype(np.float32)
        state    = torch.from_numpy(
                    np.concatenate([eef_pos, eef_quat])
                ).unsqueeze(0).to(device)                           # (1, 7)

        # model inference
        with torch.no_grad():
            action = model(state, img_tensor1, img_tensor2)                          # (1, 7)
        action_np = unnormalize_action(action.squeeze(0).cpu().numpy())                   # (7,)

        # step
        obs, reward, done, info = env.step(action_np)

        if reward == 1:
            was_success = True

        if done:
            print(f"  -> done at step {step} (task {'SUCCESS' if reward > 0 else 'FAILED'})", flush=True)
            break
    
    if was_success:
        score += 1
    # ── save video ─────────────────────────────────────────────────────────────
    env.close()


imageio.mimsave(VIDEO_PATH, [f.astype(np.uint8) for f in frames], fps=20)
print(f"[Eval] video saved to: {VIDEO_PATH}", flush=True)
print(f"[Eval] final score: {score}/{trials}", flush=True)