import sys
import time
from stable_baselines3 import PPO

# Import your custom environment
sys.path.insert(0, '/home/pavilion/ros2_ws/src/sumo_traffic_env')
from sumo_env import SumoTrafficEnv

print("======================================================")
print(" Loading Trained H-BG Brain (80k Steps)")
print("======================================================")

# 1. Initialize Environment WITH GUI
env = SumoTrafficEnv(gui=True)

# 2. Load the checkpoint model
model_path = "./models/ppo_hbg_80000_steps"
try:
    model = PPO.load(model_path)
    print(f"✓ Neural Network '{model_path}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Could not find {model_path}. Check your ./models/ folder!")
    sys.exit(1)

obs, _ = env.reset()

# 3. Watch the Agent Drive
for i in range(1500): # Run for a long time so you can watch
    # The Brain decides the action based on the Hippocampal observation
    # deterministic=True tells the AI to use its best learned action, not a random exploration one
    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, term, trunc, info = env.step(action)
    
    # Slow it down so you can actually watch it in the SUMO GUI
    time.sleep(0.05)
    
    if term or trunc:
        print("✓ Episode complete. Resetting...")
        obs, _ = env.reset()

env.close()
