import sys
import time
import numpy as np
import traci
# Ensure your path is set to the directory containing sumo_traffic_env
sys.path.insert(0, '/home/pavilion/ros2_ws/src/sumo_traffic_env')
from sumo_env import SumoTrafficEnv

print("--- Launching H-BG Visualizer ---")
# gui=True allows you to see the actual SUMO intersection
env = SumoTrafficEnv(gui=True)

try:
    obs, _ = env.reset()
    print("-> Simulation successfully reset.")
    
    # Run for 500 steps (approx 50 seconds of simulation time)
    for i in range(500):
        # Action: [Throttle, Steer]
        # In a real training run, this action would come from model.predict(obs)
        obs, rew, term, trunc, _ = env.step([0.5, 0.0])
        
        if i % 10 == 0:
            # Hippocampal Context: Which 'Place Cell' is active?
            # obs[0:5] contains our encoded spatial map
            active_cell = np.argmax(obs[:5])
            speed_kph = obs[5] * 14.0
            print(f"Step: {i} | Place Cell: {active_cell} | Speed: {speed_kph:.2f} m/s")
            
        if term or trunc:
            print("✓ Episode finished.")
            break
        time.sleep(0.05)
finally:
    env.close()
