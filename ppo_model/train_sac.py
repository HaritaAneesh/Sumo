import sys
sys.path.insert(0, '/home/pavilion/ros2_ws/src/sumo_traffic_env')
from sumo_traffic_env.sumo_env import SumoTrafficEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# Dirs
log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)

# Env - EXACT same environment and rules as your PPO model
env = SumoTrafficEnv(gui=False)

# SAC (Replacing PPO, keeping the exact same environment setup)
model = SAC('MlpPolicy', env, 
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=64,
            gamma=0.99,
            tau=0.005,
            verbose=1,
            device='cuda' if __import__('torch').cuda.is_available() else 'cpu')

# Checkpoint every 50K steps (same as your PPO script)
checkpoint_cb = CheckpointCallback(save_freq=50000, save_path=log_dir, name_prefix='sac_ckpt')

print("Training SAC on SUMO traffic...")
model.learn(total_timesteps=500_000, callback=checkpoint_cb, log_interval=10)

# Save final model
model.save(f'{log_dir}/sac_agent_final')
print(f"✓ Model saved to {log_dir}")
env.close()
