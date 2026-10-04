import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3

sys.path.append(os.path.abspath('../sumo_environment'))
from sumo_env_shielded import SumoTrafficEnv
import traci

def run_stress_test():
    print("\n--- INITIATING DETERMINISTIC SHIELD STRESS TEST ---")
    print("Accelerating to cruising speed, then injecting a phantom obstacle to force mathematical shield activation...")
    
    env = SumoTrafficEnv(sumo_cfg=os.path.abspath("../sumo_environment/network/urban.sumocfg"), gui=False)
    model = TD3.load("./logs/td3_model_final.zip")
    expected_obs_dim = model.observation_space.shape[0]
    
    obs, _ = env.reset()
    velocities = []
    interventions = []
    
    done = False
    step = 0
    obstacle_triggered = False
    mock_dist = 40.0
    
    # --- MONKEY PATCH TRACI FOR GUARANTEED TRAP ---
    original_getLeader = traci.vehicle.getLeader
    original_getSpeed = traci.vehicle.getSpeed
    
    def mock_getLeader(vehID, dist=80.0):
        if vehID == env.ego_id and obstacle_triggered:
            return ("phantom_obstacle", mock_dist)
        return original_getLeader(vehID, dist)
        
    def mock_getSpeed(vehID):
        if vehID == "phantom_obstacle":
            return 0.0
        return original_getSpeed(vehID)
        
    traci.vehicle.getLeader = mock_getLeader
    traci.vehicle.getSpeed = mock_getSpeed
    # ----------------------------------------------
    
    while not done and step < 300:
        current_dim = obs.shape[0]
        if current_dim < expected_obs_dim:
            eval_obs = np.pad(obs, (0, expected_obs_dim - current_dim), 'constant')
        else:
            eval_obs = obs[:expected_obs_dim]

        action, _ = model.predict(eval_obs, deterministic=True)
        action = np.atleast_1d(action)
        
        safe_action = np.zeros(2, dtype=np.float32)
        safe_action[0] = 1.0  # Force acceleration to get up to speed quickly
        if action.shape[0] >= 2:
            safe_action[1] = action[1]
            
        current_speed = original_getSpeed(env.ego_id)
        
        # Trigger the trap once we are moving fast
        if not obstacle_triggered and current_speed > 12.0:
            print(f"\n-> [Step {step}] TRAP SPRUNG! Injecting phantom stationary obstacle exactly 40m ahead.")
            obstacle_triggered = True
            
        if obstacle_triggered:
            safe_action[0] = 1.0  # Continue flooring the gas to fight the shield
            mock_dist -= current_speed * 0.1  # Update distance (assuming 10Hz simulation step)
            if mock_dist <= 0:
                print("-> IMPACT! Shield failed to stop in time.")
                break
                
        obs, reward, terminated, truncated, info = env.step(safe_action)
        
        velocities.append(info.get("velocity", 0.0))
        is_intervening = info.get("shield_intervention", False)
        interventions.append(1 if is_intervening else 0)
        
        done = terminated or truncated or info.get("collision", False)
        
        # Stop early if the car successfully braked to a halt to avoid the phantom
        if obstacle_triggered and info.get("velocity", 0.0) < 0.2 and is_intervening:
            print(f"-> [Step {step}] SUCCESS! Shield forced vehicle to a complete safe stop.")
            print(f"-> Distance remaining to obstacle: {mock_dist:.2f}m")
            break
            
        step += 1

    # Restore original functions and close
    traci.vehicle.getLeader = original_getLeader
    traci.vehicle.getSpeed = original_getSpeed
    env.close()

    print("\n--- STRESS TEST RESULTS ---")
    print(f"Total Steps: {step}")
    print(f"Total Shield Interventions Triggered: {sum(interventions)}")
    
    # Generate the Proof Graph
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(velocities, label='Ego Vehicle Speed (m/s)', color='blue', linewidth=2)
    
    intervention_indices = [i for i, x in enumerate(interventions) if x == 1]
    if intervention_indices:
        plt.scatter(intervention_indices, [velocities[i] for i in intervention_indices], 
                    color='red', label='Shield Override (Braking Applied)', zorder=5)

    plt.title("Bio-Inspired TTC Safety Shield: Deterministic Override Response", fontweight='bold')
    plt.xlabel("Simulation Timesteps")
    plt.ylabel("Velocity (m/s)")
    plt.axhline(y=0, color='black', linewidth=1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    os.makedirs("./journal_figures/", exist_ok=True)
    plt.savefig("./journal_figures/fig_shield_override.pdf")
    print("\nProof graph saved to ./journal_figures/fig_shield_override.pdf")

if __name__ == "__main__":
    run_stress_test()
