import sys
import numpy as np
import traci
from stable_baselines3 import PPO

# Ensure path to your environment
sys.path.insert(0, '/home/pavilion/ros2_ws/src/sumo_traffic_env/sumo_traffic_env')
from sumo_env import SumoTrafficEnv

def track_full_episode(model_path):
    print(f"--- Loading Model: {model_path} ---")
    model = PPO.load(model_path)
    
    env = SumoTrafficEnv(gui=True)
    obs, _ = env.reset()
    
    done = False
    step_count = 0
    
    print("--- Tracking Full Episode ---")
    try:
        while not done:
            # Use the trained brain to choose actions
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Extract Hippocampal Part (0-4)
            peak_cell = np.argmax(obs[0:5])
            velocity = obs[5] * 14.0
            pos = traci.vehicle.getPosition("ego_0")
            
            # Print status every 50 steps to keep the terminal readable
            if step_count % 50 == 0:
                print(f"Step: {step_count} | Pos: ({pos[0]:.1f}, {pos[1]:.1f}) | "
                      f"Vel: {velocity:.1f}m/s | Peak Place Cell: {peak_cell}")
            
            step_count += 1
            
    except Exception as e:
        print(f"Tracking error: {e}")
    finally:
        env.close()
        print(f"Episode finished at step {step_count}.")

if __name__ == "__main__":
    model_path = "/home/pavilion/ros2_ws/src/rl_agent/logs/ppo_ckpt_150000_steps.zip"
    track_full_episode(model_path)
