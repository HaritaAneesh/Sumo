"""
TraCI-based traffic light controller.
4-phase cycle per junction, staggered offsets.
"""
import traci

TL_JUNCTIONS = {
    'n_north': {'offset': 0,  'cycle': 70},
    'n_east':  {'offset': 18, 'cycle': 70},
    'n_south': {'offset': 36, 'cycle': 70},
    'n_west':  {'offset': 54, 'cycle': 70},
    'c_ne':    {'offset': 9,  'cycle': 70},
    'c_nw':    {'offset': 27, 'cycle': 70},
    'c_se':    {'offset': 45, 'cycle': 70},
    'c_sw':    {'offset': 63, 'cycle': 70},
}

PHASE_GREEN  = 0   # 30s green
PHASE_YELLOW = 1   # 5s yellow
PHASE_RED    = 2   # 30s red
PHASE_YELLOW2= 3   # 5s yellow

def get_phase(tl_id, sim_time):
    cfg   = TL_JUNCTIONS[tl_id]
    t     = (sim_time + cfg['offset']) % cfg['cycle']
    if   t < 30: return 'green'
    elif t < 35: return 'yellow'
    elif t < 65: return 'red'
    else:        return 'yellow'

def apply_tl(tl_id, phase_str, n_links):
    if phase_str == 'green':
        h = n_links // 2
        state = 'G' * h + 'r' * (n_links - h)
    elif phase_str == 'yellow':
        h = n_links // 2
        state = 'y' * h + 'r' * (n_links - h)
    elif phase_str == 'red':
        h = n_links // 2
        state = 'r' * (n_links - h) + 'G' * h
    else:
        h = n_links // 2
        state = 'r' * (n_links - h) + 'y' * h
    traci.trafficlight.setRedYellowGreenState(tl_id, state)

# link counts from earlier check
TL_LINKS = {
    'n_north': 9,  'n_east': 9,
    'n_south': 9,  'n_west': 9,
    'c_ne': 15, 'c_nw': 14,
    'c_se': 16, 'c_sw': 16,
}

def step_tl(sim_time):
    """Call every simulation step."""
    for tl_id in TL_JUNCTIONS:
        if tl_id in traci.trafficlight.getIDList():
            phase = get_phase(tl_id, sim_time)
            apply_tl(tl_id, phase, TL_LINKS[tl_id])
