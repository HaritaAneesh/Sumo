import sys
import numpy as np
import traci  # Import traci directly

# Point to your environment
sys.path.insert(0, '/home/pavilion/ros2_ws/src/sumo_traffic_env/sumo_traffic_env')
from sumo_env import SumoTrafficEnv

def show_spatial_map():
    print("--- Hippocampal Mapping Output: Real-time Activation ---")
    
    # Initialize env
    env = SumoTrafficEnv(gui=False)
    obs, _ = env.reset()
    
    # 1. Get position using traci (the industry standard for SUMO)
    # Replace 'ego_vehicle' with the actual vehicle ID from your scenario
    # If you aren't sure, check your .sumocfg or XML file
    vehicle_id = "ego_0" 
    try:
        x, y = traci.vehicle.getPosition(vehicle_id)
        print(f"Current Vehicle Position: (x={x:.2f}, y={y:.2f})")
    except Exception as e:
        print("Could not retrieve vehicle position via traci. Trying alternate approach...")
        # If 'ego_vehicle' is wrong, this will print what's available
        print(f"Available vehicles: {traci.vehicle.getIDList()}")
        return

    # 2. Get the hippocampal mapping
    mapping = env._get_obs() 
    
    print("\n--- Hippocampal Activation Vector (Input to PPO) ---")
    print(f"Activation Signature (first 10 units): {mapping[:10]}")
    
    peak_cell = np.argmax(mapping)
    print(f"\nPeak Spatial Activation at Place Cell index: {peak_cell}")
    print("This index represents the 'where' in the agent's cognitive map.")

    env.close()

if __name__ == "__main__":
    show_spatial_map()
