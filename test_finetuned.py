# given specific checkpoint, run a test episode, and save video for sanity check.



import torch
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image
import robosuite as suite
import imageio
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Load base model + LoRA weights
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
base_model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, '/code/checkpoints/epoch_4')
model = model.to('cuda')
model.eval()

# Run in environment
env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["frontview", "robot0_eye_in_hand", "agentview"],
    camera_heights=256,
    camera_widths=256,
)

obs = env.reset()
instruction = "pick up the red cube"
frames = []
total_reward = 0

for step in range(100):
    # Save front view

    agent_view = np.flipud(obs["agentview_image"])
    front_view = np.flipud(obs["frontview_image"])
    wrist_view = np.flipud(obs["robot0_eye_in_hand_image"])

    side_by_side_frame = np.hstack((agent_view, front_view, wrist_view))
    frames.append(side_by_side_frame)
    
    # Get wrist image for OpenVLA
    image = obs["agentview_image"]  # shape (256, 256, 3), uint8
    image_pil = Image.fromarray(image)
    
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, image_pil, return_tensors="pt")
    inputs = {k: v.to("cuda", dtype=torch.bfloat16) if v.dtype == torch.float32 else v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        # per dataset
        your_mean = [0.1724758948893022, 0.0058145044485826336, -0.16882960893854707, 0.003080295975832931, 0.005131051984319222, 0.011490695106420836, -0.40761431822884336]
        your_std  = [0.25917604661941557, 0.1298246124690503, 0.493942422490268, 0.02227463087148724, 0.06348723466224958, 0.08342674048727017, 0.913154186090666]
        your_min  = [-1.0, -0.5599999999999999, -1.0, -0.15065869688987732, -1.0, -0.5179753303527832, -1.0]
        your_max  = [1.0, 0.652, 1.0, 0.11863560229539871, 0.30509257316589355, 0.4782337248325348, 1.0]
        your_q01  = [-0.27369999999999994, -0.28535000000000005, -0.8443499999999999, -0.05623891334980726, -0.08138361163437366, -0.17549367770552635, -1.0]
        your_q99  = [0.7963500000000003, 0.36035000000000034, 1.0, 0.0558469627052547, 0.16937012597918513, 0.2550888940691948, 1.0]

        model.norm_stats["robosuite_lift"] = {
            "action": {
                "mean": your_mean,
                "std":  your_std,
                "min":  your_min,
                "max":  your_max,
                "q01":  your_q01,
                "q99":  your_q99,
            }
        }

        # Use your key
        action = model.predict_action(**inputs, unnorm_key="robosuite_lift", do_sample=False)
    
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()
    
    obs, reward, done, info = env.step(action)
    total_reward += reward
    
    if done:
        print(f"Episode done at step {step}!")
        break

print(f"Total reward: {total_reward:.3f}")
imageio.mimsave('/code/finetuned_test.mp4', frames, fps=20)
print("Video saved!")
env.close()