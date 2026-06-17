# given specific checkpoint, run a test episode, and save video for sanity check.



import torch
import numpy as np
from PIL import Image
import robosuite as suite
import imageio
import os

#model =
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
frames = []
total_reward = 0

for step in range(100):
    # Save front view

    agent_view = np.flipud(obs["agentview_image"])
    front_view = np.flipud(obs["frontview_image"])
    wrist_view = np.flipud(obs["robot0_eye_in_hand_image"])

    #side_by_side_frame = np.hstack((agent_view, front_view, wrist_view))
    #frames.append(side_by_side_frame)
    frames.append(wrist_view)
    
    # Get wrist image for OpenVLA
    image = obs["agentview_image"]  # shape (256, 256, 3), uint8
    image_pil = Image.fromarray(image)
        

    
    obs, reward, done, info = env.step(action)
    total_reward += reward
    
    if done:
        print(f"Episode done at step {step}!")
        break

print(f"Total reward: {total_reward:.3f}")
imageio.mimsave('/code/finetuned_test.mp4', frames, fps=20)
print("Video saved!")
env.close()