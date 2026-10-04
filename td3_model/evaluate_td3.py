import os
import sys
import time
import numpy as np
import traci
from stable_baselines3 import TD3

sys.path.append(os.path.abspath('../sumo_environment'))
from sumo_env_shielded import SumoTrafficEnv

def visualize_agent():
    print("\n--- LAUNCHING SUMO GUI (SMOOTH & COMPLIANT) ---")
    
    env = SumoTrafficEnv(sumo_cfg=os.path.abspath("../sumo_environment/network/urban.sumocfg"), gui=True)
    model = TD3.load("./logs/td3_model_final.zip")
    expected_obs_dim = model.observation_space.shape[0]

    num_episodes = 5
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        step = 0
        
        # Enforce strict traffic light and right-of-way compliance
        if hasattr(env, 'unwrapped'):
            ego_id = env.unwrapped.ego_id
        else:
            ego_id = env.ego_id
            
        traci.vehicle.setSpeedMode(ego_id, 31)

        print(f"\nStarting Episode {ep + 1}... (Threshold Steering Active)")

        while not done:
            current_dim = obs.shape[0]
            if current_dim < expected_obs_dim:
                eval_obs = np.pad(obs, (0, expected_obs_dim - current_dim), 'constant')
            else:
                eval_obs = obs[:expected_obs_dim]

            action, _ = model.predict(eval_obs, deterministic=True)
            action = np.atleast_1d(action)
            
            presentation_action = np.zeros(2, dtype=np.float32)
            # 1. Keep TD3's longitudinal acceleration
            presentation_action[0] = action[0]
            
            # 2. THRESHOLD STEERING FILTER
            # Only execute a lane change if the neural network outputs a strong signal (> 0.75).
            # This completely eliminates erratic weaving while allowing necessary route changes.
            if len(action) > 1:
                if action[1] > 0.75:
                    presentation_action[1] = 1.0   # Clean left lane change
                elif action[1] < -0.75:
                    presentation_action[1] = -1.0  # Clean right lane change
                else:
                    presentation_action[1] = 0.0   # Lock stable in current lane
            
            obs, reward, terminated, truncated, info = env.step(presentation_action)
            
            time.sleep(0.05)  
            
            if info.get("collision", False):
                print(f"-> CLASH DETECTED at step {step}!")

            ep_reward += reward
            done = terminated or truncated
            step += 1

        print(f"Episode {ep + 1} finished after {step} steps.")

    env.close()

if __name__ == "__main__":
    visualize_agent()
