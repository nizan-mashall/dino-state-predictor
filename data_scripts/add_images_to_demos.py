import h5py
import numpy as np
import robosuite as suite
import json

input_path = '/users/ogal/nmashall/dino-state-predictor/data/lift/mh/low_dim_v141.hdf5'
output_path = '/users/ogal/nmashall/dino-state-predictor/data/lift/mh/lift_mix.hdf5'

f = h5py.File(input_path, 'r')
env_args = json.loads(f['data'].attrs['env_args'])

# Create environment
env = suite.make(
    env_args['env_name'],
    robots=env_args['env_kwargs']['robots'],
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["agentview", "robot0_eye_in_hand"],
    camera_heights=256,
    camera_widths=256,
)

out_f = h5py.File(output_path, 'w')
out_data = out_f.create_group('data')

demos = list(f['data'].keys())#[:200]
print(f"Processing {len(demos)} demos...")

for i, demo_key in enumerate(demos):
    demo = f['data'][demo_key]
    states = demo['states'][:]
    actions = demo['actions'][:]

    # Reset env to first state
    env.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()

    agentview_images = []
    wrist_images = []
    eef_pos = []
    eef_quat = []
    gripper_qpos = []

    for t in range(len(actions)):
        # Set state and render
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()
        
        # Get observations after setting state
        obs = env._get_observations(force_update=True)

        agentview_images.append(obs['agentview_image'].copy())
        wrist_images.append(obs['robot0_eye_in_hand_image'].copy())
        eef_pos.append(obs['robot0_eef_pos'].copy())
        eef_quat.append(obs['robot0_eef_quat'].copy())
        gripper_qpos.append(obs['robot0_gripper_qpos'].copy())

    # Save to output file
    out_demo = out_data.create_group(demo_key)
    out_demo.create_dataset('actions', data=actions)
    obs_group = out_demo.create_group('obs')
    obs_group.create_dataset('agentview_image', data=np.array(agentview_images))
    obs_group.create_dataset('robot0_eye_in_hand_image', data=np.array(wrist_images))
    obs_group.create_dataset('robot0_eef_pos', data=np.array(eef_pos))
    obs_group.create_dataset('robot0_eef_quat', data=np.array(eef_quat))
    obs_group.create_dataset('robot0_gripper_qpos', data=np.array(gripper_qpos))

    if (i+1) % 10 == 0:
        print(f"Processed {i+1}/{len(demos)} demos")

f.close()
out_f.close()
env.close()
print("Done! Saved to", output_path)
