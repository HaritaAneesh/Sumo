import sys
sys.path.insert(0, '/home/pavilion/ros2_ws/src/sumo_traffic_env')
from sumo_traffic_env.sumo_env import SumoTrafficEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# Dirs
log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)

# Env
env = SumoTrafficEnv(gui=False)

# PPO (Basal Ganglia reward already in SumoTrafficEnv)
model = PPO('MlpPolicy', env, 
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            device='cuda' if __import__('torch').cuda.is_available() else 'cpu')

# Checkpoint every 50K steps
checkpoint_cb = CheckpointCallback(save_freq=50000, save_path=log_dir, name_prefix='ppo_ckpt')

print("Training PPO on SUMO traffic...")
model.learn(total_timesteps=500_000, callback=checkpoint_cb, log_interval=10)

# Save final model
model.save(f'{log_dir}/ppo_agent_final')
print(f"✓ Model saved to {log_dir}")

env.close()
