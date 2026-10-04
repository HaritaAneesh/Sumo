import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
# Import your custom environment
from sumo_env import SumoTrafficEnv

def make_env():
    # gui=False makes training 100x faster
    return SumoTrafficEnv(gui=False)

# 1. Wrap the environment for Stable Baselines
env = DummyVecEnv([make_env])

# 2. Initialize the PPO Agent
# MlpPolicy is a standard neural network that will map your 19-value observation to actions
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.0003, 
    n_steps=2048,
    batch_size=64,
    gamma=0.99
)

# Optional: Save the model every 10,000 steps so you don't lose progress
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='ppo_hbg')

print("======================================================")
print(" Starting H-BG Brain Training (PPO)")
print("======================================================")

# 3. Train! 
# 100,000 timesteps is a good starting point for a simple intersection
model.learn(total_timesteps=100000, callback=checkpoint_callback)

# 4. Save final model
model.save("ppo_sumo_brain_final")
print("✓ Training complete. Brain saved as 'ppo_sumo_brain_final.zip'")
