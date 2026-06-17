import robosuite as suite
import numpy as np
import cv2

env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=[ "robot0_eye_in_hand"],  # two cameras!
    camera_heights=256,
    camera_widths=256,
)

obs = env.reset()

for step in range(50):

    # Wrist camera (what the gripper sees)
    wrist = obs["robot0_eye_in_hand_image"]
    wrist = cv2.flip(wrist, 0)
    wrist = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)
    cv2.imshow("Wrist Camera", wrist)

    cv2.waitKey(1)
    
    action_low, action_high = env.action_spec
    print(f"Action space: low={action_low}, high={action_high}")
    action = np.random.uniform(action_low, action_high)
    
    obs, reward, done, info = env.step(action)
    env.render()
    
    if done:
        obs = env.reset()

env.close()
cv2.destroyAllWindows()