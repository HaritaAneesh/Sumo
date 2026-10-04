import os
import sys
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Tell Python exactly where to find sumo_env.py
sys.path.append('/home/pavilion/ros2_ws/src/sumo_traffic_env/sumo_traffic_env')
from sumo_env import SumoTrafficEnv
# Strict Journal Quality Settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 12,
    'font.size': 10,
    'legend.fontsize': 10,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight'
})

def evaluate_checkpoint(model_path, env, num_episodes=10):
    """Calculates accuracy (success rate) for a single checkpoint."""
    model = PPO.load(model_path)
    success_count = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_steps = 0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_steps += 1
            if ep_steps > 300: # Timeout safety
                break
                
        # Register successful navigation
        if info.get('is_success', False) or reward > 0: # Adjust condition based on your exact env logic
            success_count += 1
            
    return (success_count / num_episodes) * 100

def generate_accuracy_curve():
    log_dir = "/home/pavilion/ros2_ws/src/rl_agent/logs/"
    checkpoint_files = glob.glob(os.path.join(log_dir, "*.zip"))
    
    if not checkpoint_files:
        print("No checkpoint .zip files found in logs directory.")
        return

    # Extract step numbers and sort files
    checkpoints = []
    for file in checkpoint_files:
        match = re.search(r'ckpt_(\d+)_steps', file)
        if match:
            steps = int(match.group(1))
            checkpoints.append((steps, file))
    
    checkpoints.sort(key=lambda x: x[0])
    
    print(f"Found {len(checkpoints)} checkpoints. Starting evaluation...")
    env = SumoTrafficEnv()
    
    steps_list = []
    accuracy_list = []

    for steps, file in checkpoints:
        print(f"Evaluating checkpoint at {steps} steps...")
        accuracy = evaluate_checkpoint(file, env, num_episodes=15)
        steps_list.append(steps)
        accuracy_list.append(accuracy)
        print(f"--> Accuracy: {accuracy:.1f}%")

    # Plot the true accuracy progression
    os.makedirs("journal_figures", exist_ok=True)
    plt.figure(figsize=(6, 4))
    
    plt.plot(steps_list, accuracy_list, marker='o', linestyle='-', color='#004488', linewidth=1.5)
    
    plt.title("Model Accuracy (Success Rate) vs. Training Steps", pad=10)
    plt.xlabel("Training Steps")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    out_path = "journal_figures/fig2_accuracy_progression.pdf"
    plt.savefig(out_path)
    plt.close()
    print(f"\nSaved precise accuracy progression graph to {out_path}")

if __name__ == "__main__":
    generate_accuracy_curve()
