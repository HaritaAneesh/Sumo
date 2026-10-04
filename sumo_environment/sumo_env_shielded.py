import os
import sys
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Ensure SUMO_HOME is set in system environment
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

class SumoTrafficEnv(gym.Env):
    """
    Bio-Inspired Reinforcement Learning Environment for High-Density Traffic Navigation.
    Features:
      - Hippocampal-inspired multi-directional sensory grid observation space.
      - Continuous Looming-based Time-To-Collision (TTC) Bio-Safety Shield.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, sumo_cfg="network/osm.sumocfg", gui=False, max_steps=1000):
        super(SumoTrafficEnv, self).__init__()

        self.sumo_cfg = sumo_cfg
        self.gui = gui
        self.max_steps = max_steps
        self.current_step = 0
        self.ego_id = "ego_vehicle"

        # Bio-Shield Configuration Parameters
        self.tau_crit = 2.0         # Critical Time-To-Collision threshold (seconds)
        self.d_safe = 4.0           # Minimum safety buffer distance (meters)
        self.max_decel = -4.5       # Maximum emergency deceleration (m/s^2)
        self.max_accel = 2.5        # Maximum acceleration (m/s^2)
        self.target_velocity = 15.0 # Desired cruise speed (m/s)

        # Metrics for Paper & Evaluation
        self.shield_interventions = 0
        self.total_collisions = 0
        self.min_observed_ttc = float('inf')

        # Action Space: Continuous [Acceleration (-4.5 to 2.5 m/s^2), Lane Change / Lateral (-1.0 to 1.0)]
        self.action_space = spaces.Box(
            low=np.array([-4.5, -1.0], dtype=np.float32),
            high=np.array([2.5, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation Space (16 Dimensions)
        self.obs_dim = 16
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.shield_interventions = 0
        self.total_collisions = 0
        self.min_observed_ttc = float('inf')

        try:
            traci.close()
        except Exception:
            pass

        sumo_binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_cfg,
            "--start",
            "--quit-on-end",
            "--collision.action", "warn",
            "--no-step-log", "true"
        ]

        traci.start(sumo_cmd)

        # Advance simulation until ego vehicle is inserted
        ego_loaded = False
        init_steps = 0
        while not ego_loaded and init_steps < 100:
            traci.simulationStep()
            init_steps += 1
            if self.ego_id in traci.vehicle.getIDList():
                ego_loaded = True

        if not ego_loaded:
            veh_list = traci.vehicle.getIDList()
            if len(veh_list) > 0:
                self.ego_id = veh_list[0]
            else:
                self.ego_id = "veh0"

        # Disable automatic collision avoidance to let RL & Shield control dynamics
        traci.vehicle.setSpeedMode(self.ego_id, 0)
        traci.vehicle.setLaneChangeMode(self.ego_id, 0)

        obs = self._get_hippocampal_observation()
        info = {}
        return obs, info

    def step(self, action):
            self.current_step += 1

            # Safety check: if ego vehicle somehow left the simulation, force termination
            if self.ego_id not in traci.vehicle.getIDList():
                obs = np.zeros(self.obs_dim, dtype=np.float32)
                return obs, -100.0, True, False, {"collision": True, "velocity": 0.0, "total_interventions": self.shield_interventions}

            raw_accel = float(action[0])
            raw_steer = float(action[1])

            # Apply Bio-Inspired Time-To-Collision Safety Shield
            safe_accel, intervened = self._apply_bio_shield(raw_accel)
            if intervened:
                self.shield_interventions += 1

            # Execute Kinematics in SUMO
            current_speed = traci.vehicle.getSpeed(self.ego_id)
            next_speed = max(0.0, current_speed + safe_accel * 1.0)
            traci.vehicle.setSpeed(self.ego_id, next_speed)

            # Execute Lateral / Lane Change Decision
            if raw_steer > 0.3:
                current_lane = traci.vehicle.getLaneIndex(self.ego_id)
                if current_lane < traci.edge.getLaneNumber(traci.vehicle.getRoadID(self.ego_id)) - 1:
                    traci.vehicle.changeLane(self.ego_id, current_lane + 1, 2.0)
            elif raw_steer < -0.3:
                current_lane = traci.vehicle.getLaneIndex(self.ego_id)
                if current_lane > 0:
                    traci.vehicle.changeLane(self.ego_id, current_lane - 1, 2.0)

            traci.simulationStep()

            # Check Environment State
            collision = self.ego_id in traci.simulation.getCollidingVehiclesIDList()
            if collision:
                self.total_collisions += 1

            terminated = collision or (self.current_step >= self.max_steps)
            truncated = False

            reward = self._compute_reward(current_speed, safe_accel, raw_accel, collision, intervened)
            obs = self._get_hippocampal_observation()
            info = {
                "shield_intervention": intervened,
                "total_interventions": self.shield_interventions,
                "collision": collision,
                "velocity": current_speed,
                "min_ttc": self.min_observed_ttc
            }

            return obs, reward, terminated, truncated, info

    
    def _apply_bio_shield(self, raw_accel):
        ego_speed = traci.vehicle.getSpeed(self.ego_id)
        leader = traci.vehicle.getLeader(self.ego_id, 80.0)

        safe_accel = raw_accel
        intervened = False

        if leader is not None:
            lead_id, distance = leader
            lead_speed = traci.vehicle.getSpeed(lead_id)
            rel_speed = ego_speed - lead_speed

            if rel_speed > 0.05:
                tau = distance / rel_speed
                if tau < self.min_observed_ttc:
                    self.min_observed_ttc = tau

                if tau < self.tau_crit:
                    effective_dist = max(distance - self.d_safe, 0.5)
                    required_decel = - (rel_speed ** 2) / (2.0 * effective_dist)
                    bounded_decel = float(np.clip(required_decel, self.max_decel, 0.0))

                    if raw_accel > bounded_decel:
                        safe_accel = bounded_decel
                        intervened = True

        return safe_accel, intervened

    def _get_hippocampal_observation(self):
        # Safety check: if vehicle vanished, return zero array immediately
        if self.ego_id not in traci.vehicle.getIDList():
            return np.zeros(self.obs_dim, dtype=np.float32)

        ego_speed = traci.vehicle.getSpeed(self.ego_id)
        current_lane = traci.vehicle.getLaneIndex(self.ego_id)
        lane_pos = traci.vehicle.getLanePosition(self.ego_id)

        sensory_rays = np.full(8, 60.0, dtype=np.float32)
        rel_speeds = np.zeros(4, dtype=np.float32)
        ttc_value = 10.0

        leader = traci.vehicle.getLeader(self.ego_id, 60.0)
        if leader is not None:
            lead_id, dist = leader
            sensory_rays[0] = dist
            l_speed = traci.vehicle.getSpeed(lead_id)
            rel_speeds[0] = ego_speed - l_speed
            if rel_speeds[0] > 0.05:
                ttc_value = min(10.0, dist / rel_speeds[0])

        follower = traci.vehicle.getFollower(self.ego_id, 60.0)
        if follower is not None:
            fol_id, dist = follower
            sensory_rays[4] = dist
            if fol_id != "":
                f_speed = traci.vehicle.getSpeed(fol_id)
                rel_speeds[1] = f_speed - ego_speed

        obs = np.array([
            ego_speed / self.target_velocity,
            float(current_lane),
            lane_pos / 500.0,
            sensory_rays[0] / 60.0,
            sensory_rays[1] / 60.0,
            sensory_rays[2] / 60.0,
            sensory_rays[3] / 60.0,
            sensory_rays[4] / 60.0,
            sensory_rays[5] / 60.0,
            sensory_rays[6] / 60.0,
            sensory_rays[7] / 60.0,
            rel_speeds[0] / 10.0,
            rel_speeds[1] / 10.0,
            rel_speeds[2] / 10.0,
            rel_speeds[3] / 10.0,
            ttc_value / 10.0
        ], dtype=np.float32)

        return np.nan_to_num(obs, nan=0.0)

    def _compute_reward(self, current_speed, safe_accel, raw_accel, collision, intervened):
        if collision:
            return -100.0

        r_speed = 1.0 - abs(self.target_velocity - current_speed) / self.target_velocity
        r_smooth = - 0.1 * abs(safe_accel)
        r_shield = - 3.0 if intervened else 0.0
        r_discrepancy = - 0.5 * abs(raw_accel - safe_accel)

        return r_speed + r_smooth + r_shield + r_discrepancy

    def close(self):
        try:
            traci.close()
        except Exception:
            pass
