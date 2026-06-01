import torch
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import robosuite as suite
import imageio

print("Loading OpenVLA...")
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b",
    trust_remote_code=True
)
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda")
print("Model loaded!")

print("Loading RoboSuite environment...")
env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["frontview", "robot0_eye_in_hand"],
    camera_heights=256,
    camera_widths=256,
)

obs = env.reset()
instruction = "pick up the red cube"
print(f"Running inference for: '{instruction}'")

frames = []

for step in range(20):

    # Video recording
    wrist_image = obs["frontview_image"]
    
    # Stack side by side for video

    frames.append(np.flipud(wrist_image))
    # end of Video recording

    image = obs["robot0_eye_in_hand_image"]
    image_pil = Image.fromarray(image)

    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, image_pil, return_tensors="pt")
    inputs = {k: v.to("cuda", dtype=torch.bfloat16) if v.dtype == torch.float32 else v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        action = model.predict_action(
            **inputs,
            unnorm_key="bridge_orig",
            do_sample=False
        )

    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()

    print(f"Step {step+1} | Action: {action}")
    obs, reward, done, info = env.step(action)

    if done:
        print("Episode done!")
        break

print("Saving video...")
imageio.mimsave('/code/openvla_demo.mp4', frames, fps=20)
print("Video saved to /code/openvla_demo.mp4")

env.close()
print("Test complete!")
