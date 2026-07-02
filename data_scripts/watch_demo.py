import h5py
import numpy as np
import imageio

f = h5py.File('/code/data/demo_with_images.hdf5', 'r')

# Get first demo
demo = f['data']['demo_0']
frames = demo['obs']['agentview_image'][:]

print(f"Episode length: {len(frames)} steps")
print(f"Image shape: {frames[0].shape}")

# Save video
imageio.mimsave('/code/demo_0.mp4', frames, fps=20)
print("Video saved to /code/demo_0.mp4")

f.close()
