import sys
sys.path.insert(0, '/home/pavilion/ros2_ws/src/sumo_traffic_env')
from sumo_traffic_env.sumo_env import SumoTrafficEnv
from stable_baselines3 import SAC
import time

# Initialize environment with GUI enabled to watch the agent
env = SumoTrafficEnv(gui=True)

# Load the latest saved SAC checkpoint
model_path = './logs/sac_ckpt_150000_steps.zip'
print(f"Loading SAC model from {model_path}...")
model = SAC.load(model_path, env=env)

print("Starting SAC evaluation visualization...")
obs, info = env.reset()
done = False
episode_reward = 0
step_count = 0

while not done:
    # Predict action using the trained SAC model (deterministic=True for evaluation)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    episode_reward += reward
    step_count += 1
    
    # Print real-time info to terminal just like the PPO script
    print(f"Step {step_count}: Reward = {reward:.2f} | Info: {info}")

print(f"Episode finished! Total Reward: {episode_reward:.2f} in {step_count} steps.")
env.close()
