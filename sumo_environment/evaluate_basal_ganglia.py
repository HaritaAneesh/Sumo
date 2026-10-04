import gymnasium as gym
from stable_baselines3 import SAC
from sumo_env import SumoTrafficEnv
import time

# Ensure this points to your saved 20,000 step model
model_path = "./models/basal_ganglia_gating_20000_steps"

# Initialize the environment with the GUI enabled
env = SumoTrafficEnv(gui=True, step_length=0.1)

model = SAC.load(model_path, env=env)

episodes = 5
print("Launching SUMO GUI. Quick: Adjust the 'Delay (ms)' slider at the top of the window!")

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    
    print(f"--- Starting Episode {ep + 1} ---")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated
        
    print(f"Episode {ep + 1} finished with reward: {episode_reward:.2f}")

env.close()
