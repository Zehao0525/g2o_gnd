from controller import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - activates 3D projection

import os, json, copy
from pathlib import Path
from typing import Optional, List


DT = 0.5


# ---- Object that writes messages to txt files -----
class LineWriter:
    def __init__(self, path: str, line_buffered: bool = True, batch_size: int = 0):
        """
        path: file to append to
        line_buffered=True -> rely on OS/stdio line buffering
        batch_size > 0 -> accumulate that many lines before writing (extra efficient)
        """
        # buffering=1 -> line buffered text mode; buffering=-1 (or omit) -> system default (~8KB)
        self._fh = open(path, "w", buffering=1 if line_buffered else -1, encoding="utf-8")
        self._batch_size = int(batch_size)
        self._buf: List[str] = []

    def write_line(self, line: str):
        if self._batch_size > 0:
            self._buf.append(line)
            if len(self._buf) >= self._batch_size:
                self._fh.writelines(self._buf)
                self._buf.clear()
        else:
            self._fh.write(line)

    def flush(self):
        if self._buf:
            self._fh.writelines(self._buf)
            self._buf.clear()
        self._fh.flush()

    def close(self):
        self.flush()
        self._fh.close()

    # optional: use with `with LineWriter(...) as w:`
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        self.close()


# ---- Keeps track of what's in the world -----
class WorldSim:
    def __init__(self, drone_sims:list):
        self.drone_sims = sorted(drone_sims, key=lambda drone_sims: drone_sims.drone_id)
        self.drone_positions = None
        self.synced = False
        self.positions()

    @staticmethod
    def create(
        config_path: str = "config/sim_config.json",
        trajectory_path: str = None,
        log_path: str = None,
        N_DRONES: int = None,
        yaw0 = 0
    ):
        # TODO get rid of yaw0
        """
        Factory to build a WorldSim and its DroneSim instances.

        Returns:
            world_sim, ctrls, sims
        """
        # 1) Start with an empty world
        world_sim = WorldSim([])

        ctrls = []
        sims = []

        cur_path = Path(__file__).resolve().parent
        config_path = os.path.join(cur_path, config_path)
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        dt = cfg['dt']

        bots = cfg['bots']

        dids = list(bots.keys())

        if N_DRONES is None:
            print("Creating simulation for ", len(dids), " Drones")
            N_DRONES = len(dids)
        elif N_DRONES > len(dids):
            print("N_DRONES too large, setting to len(dids) = ", len(dids))
            N_DRONES = len(dids)

        for i in range(N_DRONES):
            # JSON keys are assumed "0", "1", ..., or adjust if your config uses "1"-based
            drone_id = dids[i]

            #trajectory_path = trajectory_path,

            try:
                ctrl = VelocityXZWaypointController.from_json(
                    config_path,
                    drone_id,
                    dt,
                    trajectory_path
                )
            except NameError:
                raise RuntimeError(
                    "VelocityXZWaypointController (and LimitsVel, if needed) "
                    "must be defined/imported before calling WorldSim.create()."
                )

            # Join log path
            msg_log_path=f"msg_log_{drone_id}.txt"
            gt_log_path=f"gt_log_{drone_id}.txt"
            if not log_path is None:
                log_path = os.path.join(cur_path, log_path)
                os.makedirs(log_path, exist_ok=True)
                msg_log_path = os.path.join(log_path,msg_log_path)
                gt_log_path = os.path.join(log_path,gt_log_path)

            sim = DroneSim(
                drone_id,
                ctrl,
                world_sim=world_sim,
                msg_log_path=msg_log_path,
                gt_log_path=gt_log_path,
                config_path=config_path,
                dt=dt,
                yaw0=yaw0,
            )

            ctrls.append(ctrl)
            sims.append(sim)

        # 3) Attach sims back to world
        world_sim.set_drone_sims(sims)

        return world_sim, ctrls, sims

    def set_drone_sims(self, drone_sims:list):
        self.drone_sims = sorted(drone_sims, key=lambda drone_sims: drone_sims.drone_id)
        self.synced = False

    def positions(self):
        if not self.synced:
            drone_positions = {}
            for d in self.drone_sims:
                drone_positions.update({d.drone_id : d.s.copy()})
            self.drone_positions = drone_positions
            self.synced = True
        return self.drone_positions
    
    def step(self):
        for d in self.drone_sims:
            d.step()
        self.synced = False
    
    def reached_dest_all(self):
        for d in self.drone_sims:
            if not d.reached_dest(): return False
        return True


# ---------- Utility ----------
def yaw_to_quat_xyzw(yaw: float) -> np.ndarray:
    """
    Yaw (rad) -> quaternion [x, y, z, w],
    representing a pure rotation about +z by `yaw`.
    """
    half = 0.5 * yaw
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=float)


# ------ Sensor sims ----------
# Holds all the sensor simulators
# Calls them in sequence
class SensorSimManager:
    def __init__(self, world_sim, drone_id:str, config:dict, writer:LineWriter):
        self.drone_id = drone_id
        self.config = config
        self.world_sim = world_sim
        self.gps_sensor = None
        self.rel_pose_sensor = None
        self.writer = writer

        sensors_cfg = config["sensors"]

        # --- GPS ---
        gps_cfg = sensors_cfg.get("gps")
        if gps_cfg and gps_cfg.get("active", False):
            # gps.error_std is [σx, σz, ?] in config. We only use first 2.
            gps_err_std = np.array(gps_cfg["error_std"][:2], dtype=float)
            self.gps_sensor = GPSSensor(
                drone_id=drone_id,
                frequency=float(gps_cfg["frequency"]),
                error_std=gps_err_std
            )

        # --- Relative pose / bot observer ---
        bot_obs_cfg = sensors_cfg.get("bot_observer")
        if bot_obs_cfg and bot_obs_cfg.get("active", False):
            max_range = bot_obs_cfg["range"][0][1]  # [0, R] -> take R
            err_std_arr = np.array(bot_obs_cfg["error_std"], dtype=float)

            # TODO for later
            # Pad to 4 elems [σx,σy,σz,σyaw] (config gives you 3)
            # if err_std_arr.shape[0] == 3:
            #     err_std_arr = np.concatenate([err_std_arr, [0.0]])

            self.rel_pose_sensor = RelPosSensor(
                drone_id=drone_id,
                frequency=float(bot_obs_cfg["frequency"]),
                error_std=err_std_arr,
                sensor_range=float(max_range),
                range_dependent_error=False  # could come from cfg if you add it
            )

    def close(self):
        self.writer = None
        self.world_sim = None
    
    # TODO Add config

    def step(self, dt):
        #print("self.world_sim",self.world_sim.drone_sims)
        positions = self.world_sim.positions()

        # GPS
        if self.gps_sensor:
            gps_msg = self.gps_sensor.step(dt, positions)
            if gps_msg:
                self.writer.write_line(gps_msg)

        # Relative pose
        if self.rel_pose_sensor:
            rel_msg = self.rel_pose_sensor.step(dt, positions)
            if rel_msg:
                #print("write write")
                self.writer.write_line(rel_msg)


# When activated, queries world sim for location of host
class GPSSensor:
    def __init__(self, drone_id:str, frequency, error_std:np.ndarray):
        self.period = 1/frequency
        self.error_std = error_std
        self.timer = 0
        self.last_io = 0
        self.drone_id = drone_id

    # TODO Add config

    def step(self, dt, positions):
        io = None
        self.timer += dt
        if self.timer > self.period + self.last_io:
            real_pos = np.array([positions[self.drone_id][0], positions[self.drone_id][1]])
            noisy_pos = real_pos + self.error_std * np.random.standard_normal(2)
            io = noisy_pos
            self.last_io += self.period
        return self.generate_io(io)
    
    def generate_io(self, io_vals):
        if io_vals is None: return None
        return f"{self.timer} gps {io_vals[0]} {io_vals[1]} {self.error_std[0]} 0.0 0.0 {self.error_std[1]}\n"



# RangeBearing sensor sim
# Rel Pose sensor sim:
#   Gives the relative pose of the neighbour drones within a certain range
# When activated, queries world sim for location of host
class RelPosSensor:
    def __init__(self, drone_id:str, frequency, error_std:np.ndarray, sensor_range:float, range_dependent_error = False):
        self.period = 1/frequency
        self.drone_id = drone_id
        # This is 4 elements
        self.error_std = error_std
        self.range_dependent_error = range_dependent_error
        self.sensor_range = sensor_range
        self.sensor_range_sqr = sensor_range**2
        self.timer = 0
        self.last_io = 0

    def step(self, dt, positions):
        io = None
        self.timer += dt
        if self.timer > self.period + self.last_io:
            io = ''
            sx,sy,sz,syaw,_,_,_,_ = positions[self.drone_id]
            for did, dpos in positions.items():
                if did == self.drone_id: continue
                #print('detect', did, ' ', end = '    ')
                wc_rel_pose = np.array([dpos[0], dpos[1], dpos[2]]) - np.array([sx, sy, sz])
                dist_sqr = np.sum(wc_rel_pose**2)
                #print('range', dist_sqr, ' ', self.sensor_range_sqr,  end = '')
                if dist_sqr > self.sensor_range_sqr: continue
                dist = np.sqrt(dist_sqr)
                yaw_diff = dpos[3] - syaw
                rel_pose = np.array([wc_rel_pose[0] * np.cos(-syaw) - wc_rel_pose[1] * np.sin(-syaw),
                                     wc_rel_pose[0] * np.sin(-syaw) + wc_rel_pose[1] * np.cos(-syaw),
                                     wc_rel_pose[2],
                                     yaw_diff])
                terr = self.error_std
                if self.range_dependent_error:
                    terr *= 2 * dist / self.sensor_range
                noisy_reading = rel_pose + terr * np.random.standard_normal(4)
                io = io + self.generate_io(did, noisy_reading, terr)
                #print('ioed', end = '')
            #print()
            self.last_io += self.period
        return io
    
    def generate_io(self, did, io_vals, e):
        if io_vals is None: return None
        quat_val =yaw_to_quat_xyzw(io_vals[3])
        # TODO So far we are just uing 1.0 for error etd for ro and pitch. Drone never turn in these directions so these numbers should be just fine
        return f"""{self.timer} relpos {did} {io_vals[0]} {io_vals[1]} {io_vals[2]} {quat_val[0]} {quat_val[1]} {quat_val[2]} {quat_val[3]} {e[0]} 0.0 0.0 0.0 0.0 0.0 {e[1]} 0.0 0.0 0.0 0.0 {e[2]} 0.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 {e[3]}\n"""






# ---------- Minimal kinematic sim (velocity interface) ----------
class DroneSim:
    """
    State (world frame): [x, y, z, yaw, vx, vy, vz, yaw_rate]
    Inputs (commanded velocities):
      - v_fwd: body-x forward speed (projects into world x–z via yaw about +y)
      - vz:    world-z vertical speed
      - omega_yaw: yaw rate (about +y)
    Notes:
      - No motion in world-y (kept zero).
      - Perfect actuation: we integrate commanded velocities directly.
    """
    def __init__(self, drone_id:str, 
                 controller : VelocityXZWaypointController, 
                 world_sim :WorldSim = None, 
                 msg_log_path : str = 'msg_log.txt',
                 gt_log_path : str = 'gt_log.txt',
                 config_path : str = 'config/sim_config',
                 dt = DT, 
                 yaw0 = None):
        self.drone_id = drone_id

        self.ctrl = controller
        self.message_writer = LineWriter(msg_log_path, line_buffered=True, batch_size=20)
        self.gt_writer = LineWriter(gt_log_path, line_buffered=True, batch_size=20)
        self.world_sim = world_sim

         # ---------- Load config ----------
        with open(config_path, 'r') as f:
            config = json.load(f)

        bot_cfg = config["bots"][drone_id]
        self.bot_cfg = bot_cfg

        # Align dt
        dt = config["dt"]
        self.dt = dt
        self.ctrl.dt = dt
        self.timer = 0

        # TODO Add Sensor manager
        if yaw0 is None:
            yaw_0 = 0
        else:
            yaw_0 = yaw0

        start_pose = self.ctrl.wp[0]
        x0 = np.array([start_pose[0], start_pose[1], start_pose[2], yaw_0, 0, 0, 0, 0])
        self.s = x0.astype(float)

        if not world_sim is None:
            self.sensor_manager = SensorSimManager(world_sim, self.drone_id, bot_cfg, self.message_writer)
        else:
            self.sensor_manager = None


    def close(self):
        # Remove all circular pointers
        self.world_sim = None

    def step(self):
        x, y, z, yaw, vx, vy, vz, r = self.s

        # controller inputs
        pos  = np.array([x, y, z])
        vel  = np.array([vx, vy, vz])
        quat = yaw_to_quat_xyzw(yaw)

        # --- velocity controller interface ---
        u = self.ctrl.step(pos, quat, vel, r)
        v_fwd  = u["v_fwd"]
        vz_cmd     = u["vz"]
        omega_cmd  = u["omega_yaw"]

        self.record_odom(v_fwd, vz_cmd, omega_cmd)
        

        # Body-x forward -> world x,z (yaw about +y)
        vx = v_fwd * np.cos(yaw)
        vy = v_fwd * np.sin(yaw)
        vz = vz_cmd  # no y motion

        # Integrate pose
        x  += self.dt * vx
        y  += self.dt * vy
        z  += self.dt * vz

        # Yaw kinematics
        r   = omega_cmd
        yaw += self.dt * r
        yaw = (yaw + np.pi) % (2*np.pi) - np.pi

        # log
        self.timer  += self.dt

        self.s[:] = [x, y, z, yaw, vx, vy, vz, r]
        line = f"{self.timer} pose {x} {y} {z} {yaw}, {vx}, {vy}, {vz}, {r}\n"
        self.gt_writer.write_line(line)

        if not self.sensor_manager is None:
            self.sensor_manager.step(self.dt)
    
    def record_odom(self, vx, vy, r):
        noise_vx = vx
        noise_vy = vy
        noise_r = r
        line = f"{self.timer} odom {noise_vx} {noise_vy} {noise_r}\n"
        self.message_writer.write_line(line)

    def reached_dest(self):
        return self.ctrl.terminal

        