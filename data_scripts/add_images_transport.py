import h5py
import numpy as np
import robosuite as suite
import json

input_path  = '/users/ogal/nmashall/dino-state-predictor/data/transport/ph/low_dim_v141.hdf5'
output_path = '/users/ogal/nmashall/dino-state-predictor/data/transport/ph/transport_images.hdf5'

f = h5py.File(input_path, 'r')
env_args = json.loads(f['data'].attrs['env_args'])

# Create environment — Transport uses TWO robots
env = suite.make(
    env_args['env_name'],
    robots=env_args['env_kwargs']['robots'],          # will be a list of 2 robots
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["agentview", "robot0_eye_in_hand", "robot1_eye_in_hand"],  # ← add robot1
    camera_heights=256,
    camera_widths=256,
)

out_f = h5py.File(output_path, 'w')
out_data = out_f.create_group('data')

demos = list(f['data'].keys())
print(f"Processing {len(demos)} demos...")

for i, demo_key in enumerate(demos):
    demo = f['data'][demo_key]
    states = demo['states'][:]
    actions = demo['actions'][:]

    env.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()

    agentview_images = []
    wrist0_images = []
    wrist1_images = []          # ← second robot's wrist camera
    eef0_pos, eef0_quat, gripper0_qpos = [], [], []
    eef1_pos, eef1_quat, gripper1_qpos = [], [], []   # ← second robot's state

    for t in range(len(actions)):
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()

        obs = env._get_observations(force_update=True)

        agentview_images.append(obs['agentview_image'].copy())
        wrist0_images.append(obs['robot0_eye_in_hand_image'].copy())
        wrist1_images.append(obs['robot1_eye_in_hand_image'].copy())

        eef0_pos.append(obs['robot0_eef_pos'].copy())
        eef0_quat.append(obs['robot0_eef_quat'].copy())
        gripper0_qpos.append(obs['robot0_gripper_qpos'].copy())

        eef1_pos.append(obs['robot1_eef_pos'].copy())
        eef1_quat.append(obs['robot1_eef_quat'].copy())
        gripper1_qpos.append(obs['robot1_gripper_qpos'].copy())

    out_demo = out_data.create_group(demo_key)
    out_demo.create_dataset('actions', data=actions)   # (T, 14) — both arms' actions

    obs_group = out_demo.create_group('obs')
    obs_group.create_dataset('agentview_image',          data=np.array(agentview_images))
    obs_group.create_dataset('robot0_eye_in_hand_image', data=np.array(wrist0_images))
    obs_group.create_dataset('robot1_eye_in_hand_image', data=np.array(wrist1_images))

    obs_group.create_dataset('robot0_eef_pos',      data=np.array(eef0_pos))
    obs_group.create_dataset('robot0_eef_quat',     data=np.array(eef0_quat))
    obs_group.create_dataset('robot0_gripper_qpos', data=np.array(gripper0_qpos))

    obs_group.create_dataset('robot1_eef_pos',      data=np.array(eef1_pos))
    obs_group.create_dataset('robot1_eef_quat',     data=np.array(eef1_quat))
    obs_group.create_dataset('robot1_gripper_qpos', data=np.array(gripper1_qpos))

    if (i+1) % 10 == 0:
        print(f"Processed {i+1}/{len(demos)} demos")

f.close()
out_f.close()
env.close()
print("Done! Saved to", output_path)