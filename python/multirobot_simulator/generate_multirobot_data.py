# We are simulating drones
#
# What we need: 
#   - Controller:
#   - 

import os

try:
    from multirobot_simulator.controller import *
    from multirobot_simulator.trajectory_generator import *
    from multirobot_simulator.simulator import *
except ModuleNotFoundError:
    # Allow direct execution: python .../generate_multirobot_data.py
    from controller import *
    from trajectory_generator import *
    from simulator import *

config_path = "python/multirobot_simulator/config/sim_config.json"
traj_path = "python/multirobot_simulator/config/trajectories.json"
log_path = "test_data/multidrone"

def get_ids(conf_path):
    path = Path(conf_path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    bots = cfg['bots']
    return list(bots.keys())


def get_n_trajectory_midpoints(conf_path):
    path = Path(conf_path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return int(cfg.get("n_trajectory_midpoints", 0))


bot_ids = get_ids(config_path)
n_mid = get_n_trajectory_midpoints(config_path)
trajectory_generation(bot_ids, traj_path, visualize=False, n_trajectory_midpoints=n_mid)
word_sim, _, _ = WorldSim.create(trajectory_path=traj_path, log_path=log_path)

# Write bot_ids.txt for C++ AgentManager (one robot ID per line)
os.makedirs(log_path, exist_ok=True)
bot_ids_path = os.path.join(log_path, "bot_ids.txt")
with open(bot_ids_path, "w", encoding="utf-8") as f:
    for bid in bot_ids:
        f.write(str(bid) + "\n")
print(f"Wrote {bot_ids_path}")

for k in range(10000):
    word_sim.step()
    if word_sim.reached_dest_all():
        break