# We are simulating drones
#
# What we need: 
#   - Controller:
#   - 

from controller import *
from trajectory_generator import *
from simulator import *

traj_path = "config/trajectories.json"
log_path = "log"

trajectory_generation(['0','1','2','3','4'], traj_path, visualize=False)
word_sim, _, _ = WorldSim.create(trajectory_path=traj_path, log_path = log_path)

for k in range(10000):
    word_sim.step()
    if word_sim.reached_dest_all():
        break