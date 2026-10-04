import sys
import time
from stable_baselines3 import PPO

# Ensure the script can find your sumo_env.py
sys.path.insert(0, '/home/pavilion/ros2_ws/src/sumo_traffic_env/sumo_traffic_env')
from sumo_env import SumoTrafficEnv

def visualize():
    print("--- Launching H-BG Agent GUI Visualization ---")
    
    # 1. Initialize environment with GUI enabled
    # This will open the SUMO window automatically
    env = SumoTrafficEnv(gui=True)
    
    # 2. Load your final trained model
    model_path = "/home/pavilion/ros2_ws/src/rl_agent/logs/ppo_agent_final"
    model = PPO.load(model_path)
    
    print(f"✓ Model '{model_path}' loaded. Starting visualization...")
    
    obs, _ = env.reset()
    
    # Initialize baseline for velocity tracking
    previous_velocity = 0.0
    
    try:
        # Run for a set number of steps or until manual interruption
        for step in range(10000000):
            # The Brain (PPO) selects the best action for the current state
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"RAW INFO DICT: {info}")
 
            # --- Live Velocity Tracking ---
            current_velocity = info.get('velocity', 0.0) 
            velocity_change = current_velocity - previous_velocity
            
            # Print it live to the terminal
            print(f"Step {step}: Vel = {current_velocity:.2f} m/s | Δv = {velocity_change:.2f} m/s")
            
            # Overwrite the previous velocity for the next loop iteration
            previous_velocity = current_velocity
            # ------------------------------
            
            # Add a small delay so your eyes can follow the car
            time.sleep(0.05)
            
            if terminated or truncated:
                print("✓ Episode complete. Resetting...")
                obs, _ = env.reset()
                
                # Reset the baseline for the brand new episode
                previous_velocity = 0.0 
                
    except KeyboardInterrupt:
        print("\n--- Visualization stopped by user ---")
    finally:
        env.close()

if __name__ == "__main__":
    visualize()
