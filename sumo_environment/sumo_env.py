import os
import sys
import numpy as np
import gymnasium as gym
import random
from gymnasium import spaces

if "SUMO_HOME" in os.environ:
    sys.path += [os.path.join(os.environ["SUMO_HOME"], "tools")]
import traci

SUMO_CFG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "network", "urban.sumocfg"
)
EGO_ID = "ego_0"

TL_JUNCTIONS = {
    "n_north": {"offset": 0,  "cycle": 70},
    "n_east":  {"offset": 18, "cycle": 70},
    "n_south": {"offset": 36, "cycle": 70},
    "n_west":  {"offset": 54, "cycle": 70},
    "c_ne":    {"offset": 9,  "cycle": 70},
    "c_nw":    {"offset": 27, "cycle": 70},
    "c_se":    {"offset": 45, "cycle": 70},
    "c_sw":    {"offset": 63, "cycle": 70},
    "r1":      {"offset": 5,  "cycle": 40},
    "r2":      {"offset": 15, "cycle": 40},
    "r3":      {"offset": 25, "cycle": 40},
    "r4":      {"offset": 35, "cycle": 40},
}

TL_LINKS = {
    'n_north': 9, 'n_east': 9, 'n_south': 9, 'n_west': 9,
    'c_ne': 15, 'c_nw': 14, 'c_se': 16, 'c_sw': 16
}

def _get_tl_phase(tl_id, sim_time):
    cfg = TL_JUNCTIONS[tl_id]
    t = (sim_time + cfg["offset"]) % cfg["cycle"]
    cycle = cfg["cycle"]
    green_end   = int(cycle * 0.43)
    yellow_end  = int(cycle * 0.50)
    red_end     = int(cycle * 0.93)
    if t < green_end:
        return "green"
    elif t < yellow_end:
        return "yellow"
    elif t < red_end:
        return "red"
    else:
        return "yellow2"

def _apply_tl_state(tl_id, phase_str, n_links):
    h = n_links // 2
    r = n_links - h
    if phase_str == "green":
        state = "G" * h + "r" * r
    elif phase_str == "yellow":
        state = "y" * h + "r" * r
    elif phase_str == "red":
        state = "r" * r + "G" * h
    else:
        state = "r" * r + "y" * h
    traci.trafficlight.setRedYellowGreenState(tl_id, state)

def step_all_tl(sim_time):
    active = traci.trafficlight.getIDList()
    for tl_id, n in TL_LINKS.items():
        if tl_id in active:
            phase = _get_tl_phase(tl_id, sim_time)
            _apply_tl_state(tl_id, phase, n)

class SumoTrafficEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, gui=False, step_length=0.1):
        super().__init__()
        self.gui = gui
        self.step_length = step_length
        self._sumo_running = False
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )
        self._step = 0
        self._max_steps = 3600

    def _start_sumo(self):
        binary = "sumo-gui" if self.gui else "sumo"
        traci.start([
            binary, "-c", SUMO_CFG,
            "--step-length", str(self.step_length),
            "--no-warnings", "true",
            "--collision.action", "warn",
            "--ignore-route-errors", "true",
            "--time-to-teleport", "120"
        ])
        self._sumo_running = True

    def _close_sumo(self):
        if self._sumo_running:
            try:
                traci.close()
            except Exception:
                pass
            self._sumo_running = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._close_sumo()
        self._start_sumo()
        self._step = 0
        for _ in range(30):
            traci.simulationStep()
            if EGO_ID in traci.vehicle.getIDList():
                break
        return self._get_obs(), {}

    def handle_continuous_routing(self, veh_id):
        try:
            current_index = traci.vehicle.getRouteIndex(veh_id)
            current_route = traci.vehicle.getRoute(veh_id)
            
            if current_index >= len(current_route) - 2:
                all_edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
                if all_edges:
                    next_target = random.choice(all_edges)
                    traci.vehicle.changeTarget(veh_id, next_target)
        except traci.TraCIException:
            pass

    def inject_dynamic_background_traffic(self, step_counter, base_prob=0.35):
        traffic_density = base_prob + 0.15 * np.sin(step_counter / 500.0)

        if random.random() < traffic_density:
            veh_id = f"bg_veh_{step_counter}"
            all_edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
            if len(all_edges) < 2: return
            spawn_edge = random.choice(all_edges)
            exit_edge = random.choice(all_edges)
            v_type = random.choice(["aggressive_car", "heavy_truck", "erratic_driver", "sedan", "bicycle", "bus", "motorcycle"])
            try:
                route = traci.simulation.findRoute(spawn_edge, exit_edge)
                if route.edges:
                    route_id = f"route_{veh_id}"
                    traci.route.add(routeID=route_id, edges=route.edges)
                    traci.vehicle.add(
                        vehID=veh_id,
                        routeID=route_id,
                        typeID=v_type,
                        depart="now"
                    )
            except traci.TraCIException:
                pass 

    def clear_traffic_jams(self, max_wait_time=12, max_signal_time=7):
        for veh_id in traci.vehicle.getIDList():
            if veh_id != "ego_0": 
                try:
                    if traci.vehicle.getWaitingTime(veh_id) > max_wait_time:
                        traci.vehicle.remove(veh_id)
                except traci.TraCIException:
                    pass
        for tls_id in traci.trafficlight.getIDList():
            try:
                time_left = traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
                if time_left > max_signal_time:
                    traci.trafficlight.setPhaseDuration(tls_id, max_signal_time)
            except traci.TraCIException:
                pass

    def get_noisy_observation(self, raw_obs, noise_std=0.02):
        noise = np.random.normal(0, noise_std, size=raw_obs.shape)
        noisy_obs = raw_obs.copy()
        noisy_obs[5:] += noise[5:]
        return noisy_obs

    def step(self, action):
        self._apply_action(action)
        traci.simulationStep()
        self._step += 1
        obs   = self._get_obs()
        rew   = self._compute_reward()
        
        # Determine termination if ego crashed or left
        collided_vehicles = traci.simulation.getCollidingVehiclesIDList()
        term = "ego_0" in collided_vehicles or EGO_ID not in traci.vehicle.getIDList()
        trunc = False

        if obs is None: return obs, rew, term, trunc, {}
        
        self.handle_continuous_routing("ego_0")
        self.inject_dynamic_background_traffic(self._step)
        self.clear_traffic_jams()
        
        return obs, rew, term, trunc, {}

    def close(self):
        self._close_sumo()

    def _apply_action(self, action):
        if not traci.simulation.getMinExpectedNumber() > 0: return
        if EGO_ID not in traci.vehicle.getIDList(): return

        # Disable SUMO's internal Krauss car-following safety limits
        traci.vehicle.setSpeedMode(EGO_ID, 0)

        throttle = float(action[0])
        spd = traci.vehicle.getSpeed(EGO_ID)
        mspd = traci.vehicle.getMaxSpeed(EGO_ID)
        
        # Calculate new speed safely bounded between 0 and Max Speed
        new_speed = min(mspd, max(0.0, spd + throttle * 3.0 * self.step_length))
        traci.vehicle.setSpeed(EGO_ID, new_speed)

    def _get_obs(self):
        obs = np.zeros(19, dtype=np.float32)
        if EGO_ID not in traci.vehicle.getIDList(): return obs
        
        lane_id = traci.vehicle.getLaneID(EGO_ID)
        place_id = 4
        if "n_in" in lane_id:
            place_id = 0
        elif "ring_12" in lane_id or ":r1" in lane_id:
            place_id = 1
        elif "ring_23" in lane_id or ":r2" in lane_id:
            place_id = 2
        elif "ring_34" in lane_id or ":r3" in lane_id:
            place_id = 3
        obs[place_id] = 1.0 
        
        obs[5] = traci.vehicle.getSpeed(EGO_ID) / 14.0
        obs[6] = traci.vehicle.getLanePosition(EGO_ID) / 400.0
        
        others = [v for v in traci.vehicle.getIDList() if v != EGO_ID]
        x, y = traci.vehicle.getPosition(EGO_ID)
        near = sorted(others, key=lambda v: (traci.vehicle.getPosition(v)[0]-x)**2 + (traci.vehicle.getPosition(v)[1]-y)**2)[:3]
        for i, v in enumerate(near):
            vx, vy = traci.vehicle.getPosition(v)
            obs[7 + i*3] = (vx - x) / 100.0
            obs[8 + i*3] = (vy - y) / 100.0
            obs[9 + i*3] = traci.vehicle.getSpeed(v) / 14.0
        return self.get_noisy_observation(obs)

    def _compute_reward(self):
        if EGO_ID not in traci.vehicle.getIDList():
            return -50.0

        speed = traci.vehicle.getSpeed(EGO_ID)
        mspd = traci.vehicle.getMaxSpeed(EGO_ID)
        x, y = traci.vehicle.getPosition(EGO_ID)

        # 1. Forward Progress
        target_speed = mspd * 0.8  
        r_progress = speed / mspd
        r_velocity = -abs(speed - target_speed) / mspd

        # 2. Safety Potential Field
        r_safety = 0.0
        sigma_safety = 10.0
        others = [v for v in traci.vehicle.getIDList() if v != EGO_ID]
        for v in others:
            vx, vy = traci.vehicle.getPosition(v)
            dist_sq = (vx - x)**2 + (vy - y)**2
            if dist_sq < 400:  
                r_safety -= np.exp(-dist_sq / (2 * sigma_safety**2))

        # 3. Signal Awareness
        r_signal = 0.0
        try:
            links = traci.vehicle.getNextLinks(EGO_ID)
            if links and len(links[0]) >= 4 and isinstance(links[0][2], str):
                tl_id, tl_idx = links[0][2], links[0][3]
                if tl_id in traci.trafficlight.getIDList():
                    state = traci.trafficlight.getRedYellowGreenState(tl_id)
                    if tl_idx < len(state):
                        if state[tl_idx] in ['r', 'y'] and speed > 1.0:
                            r_signal -= 2.0  
                        elif state[tl_idx] in ['G', 'g'] and speed < 1.0:
                            r_signal -= 1.0  
        except Exception:
            pass

        # 4. Terminal Collision Penalty
        if EGO_ID in traci.simulation.getCollidingVehiclesIDList():
            return -50.0  

        # Reweighted properly
        r_total = (1.0 * r_progress) + (0.5 * r_velocity) + (0.5 * r_safety) + (1.0 * r_signal)
        return float(r_total)
