import os
import sys
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

# Route path to the environment directory
sys.path.append(os.path.abspath('../sumo_environment'))
from sumo_env_shielded import SumoTrafficEnv

def train():
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./logs/td3_tensorboard/", exist_ok=True)

    print("Initializing Shielded SUMO Environment for TD3 Training...")
    env = SumoTrafficEnv(sumo_cfg=os.path.abspath("../sumo_environment/network/urban.sumocfg"), gui=False)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.15 * np.ones(n_actions)
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./logs/",
        name_prefix="td3_ckpt"
    )

    print("Configuring TD3 Architecture...")
    model = TD3(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        action_noise=action_noise,
        tensorboard_log="./logs/td3_tensorboard/",
        verbose=1
    )

    print("Commencing TD3 Training across 150,000 steps...")
    model.learn(total_timesteps=150000, callback=checkpoint_callback)

    final_model_path = "./logs/td3_model_final"
    model.save(final_model_path)
    print(f"Training complete. Model saved to: {final_model_path}")
    env.close()

if __name__ == "__main__":
    train()
