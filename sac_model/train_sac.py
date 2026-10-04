import os
import sys
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# 1. Point to your new SUMO environment folder on the Desktop
sys.path.append('/home/pavilion/Desktop/Bio_Inspired_RL_Project/sumo_environment')
from sumo_env import SumoTrafficEnv

if __name__ == "__main__":
    # Create the logs directory if it doesn't exist
    os.makedirs('./logs', exist_ok=True)

    # Initialize the Environment
    env = SumoTrafficEnv()
    
    # 2. Setup Checkpoint Callback (Saves the model every 50,000 steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./logs/',
        name_prefix='sac_ckpt'
    )
    
    # 3. Initialize the SAC model (Replaces PPO)
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4, 
        buffer_size=100000,     # SAC's Experience Replay Buffer
        batch_size=256, 
        ent_coef='auto',        # Automatically maximizes entropy to handle sensor noise
        tensorboard_log="./logs/sac_tensorboard/"
    )
    
    # 4. Train the model
    print("Starting SAC training...")
    model.learn(
        total_timesteps=500000,
        callback=checkpoint_callback
    )
    
    # 5. Save the final model
    model.save("./logs/sac_agent_final")
    print("Training complete. Final model saved to ./logs/sac_agent_final")
