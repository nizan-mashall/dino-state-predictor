# Evaluation script to test model checkpoints on the Lift task in robosuite, 
# allowing us to track performance across epochs and identify the best checkpoint, and if it working.

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image
import robosuite as suite
import json

# Your dataset statistics
NORM_STATS = {
    "robosuite_lift": {
        "action": {
            "mean": [0.1724758948893022, 0.0058145044485826336, -0.16882960893854707, 0.003080295975832931, 0.005131051984319222, 0.011490695106420836, -0.40761431822884336],
            "std":  [0.25917604661941557, 0.1298246124690503, 0.493942422490268, 0.02227463087148724, 0.06348723466224958, 0.08342674048727017, 0.913154186090666],
            "min":  [-1.0, -0.5599999999999999, -1.0, -0.15065869688987732, -1.0, -0.5179753303527832, -1.0],
            "max":  [1.0, 0.652, 1.0, 0.11863560229539871, 0.30509257316589355, 0.4782337248325348, 1.0],
            "q01":  [-0.27369999999999994, -0.28535000000000005, -0.8443499999999999, -0.05623891334980726, -0.08138361163437366, -0.17549367770552635, -1.0],
            "q99":  [0.7963500000000003, 0.36035000000000034, 1.0, 0.0558469627052547, 0.16937012597918513, 0.2550888940691948, 1.0],
        }
    }
}

NUM_EPISODES = 2  # episodes per epoch
MAX_STEPS = 200    # steps per episode
CHECKPOINTS_DIR = '/code/checkpoints'

def evaluate_epoch(epoch, processor, base_model):
    checkpoint_path = f'{CHECKPOINTS_DIR}/epoch_{epoch}'
    
    if not os.path.exists(checkpoint_path):
        print(f"Epoch {epoch}: checkpoint not found, skipping")
        return None
    
    print(f"\n{'='*50}")
    print(f"Evaluating Epoch {epoch}")
    print(f"{'='*50}")
    
    # Load model
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model = model.to('cuda')
    model.eval()
    model.norm_stats.update(NORM_STATS)
    
    # Create environment
    env = suite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
    )
    
    successes = 0
    total_rewards = []
    
    for ep in range(NUM_EPISODES):
        obs = env.reset()
        total_reward = 0
        success = False
        
        for step in range(MAX_STEPS):
            image = obs["robot0_eye_in_hand_image"]
            image_pil = Image.fromarray(image)
            prompt = "In: What action should the robot take to pick up the red cube?\nOut:"
            
            inputs = processor(prompt, image_pil, return_tensors="pt")
            inputs = {k: v.to("cuda", dtype=torch.bfloat16) if v.dtype == torch.float32 else v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                action = model.predict_action(
                    **inputs,
                    unnorm_key="robosuite_lift",
                    do_sample=False
                )
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                success = True
                successes += 1
                break
        
        total_rewards.append(total_reward)
        status = "✅ SUCCESS" if success else "❌ FAIL"
        print(f"  Episode {ep+1}/{NUM_EPISODES}: reward={total_reward:.3f} {status}")
    
    env.close()
    
    success_rate = successes / NUM_EPISODES
    avg_reward = np.mean(total_rewards)
    
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Success rate: {successes}/{NUM_EPISODES} ({success_rate*100:.1f}%)")
    print(f"  Avg reward:   {avg_reward:.3f}")
    
    return {
        "epoch": epoch,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "successes": successes,
    }

def main():
    print("Loading base model...")
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True
    )
    base_model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # Find all available checkpoints
    epochs = sorted([
        int(d.replace('epoch_', ''))
        for d in os.listdir(CHECKPOINTS_DIR)
        if d.startswith('epoch_')
    ])
    
    print(f"Found checkpoints for epochs: {epochs}")
    
    results = []
    for epoch in epochs:
        result = evaluate_epoch(epoch, processor, base_model)
        if result:
            results.append(result)
    
    # Print summary table
    print(f"\n{'='*50}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"{'Epoch':<10} {'Success Rate':<15} {'Avg Reward':<15}")
    print(f"{'-'*40}")
    for r in results:
        print(f"{r['epoch']:<10} {r['success_rate']*100:.1f}%{'':<10} {r['avg_reward']:.3f}")
    
    # Best epoch
    if results:
        best = max(results, key=lambda x: x['success_rate'])
        print(f"\nBest epoch: {best['epoch']} with {best['success_rate']*100:.1f}% success rate")
    
    # Save results
    with open('/code/eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /code/eval_results.json")

if __name__ == "__main__":
    main()
