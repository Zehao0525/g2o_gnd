try:
    # Preferred when running as module:
    #   python -m multirobot_simulator.simulator
    from multirobot_simulator.controller import *
except ModuleNotFoundError:
    # Fallback when running as a file:
    #   python python/multirobot_simulator/simulator.py
    from controller import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - activates 3D projection

import os, json, copy
from pathlib import Path
from typing import Optional, List, Dict, Mapping, Any


DT = 0.5


def normalize_landmarks_mapping(m: Mapping[Any, Any]) -> Dict[int, np.ndarray]:
    """
    Build landmark_id -> xyz with integer ids (log format: t lmobs_rp <int> ...).
    Json/config keys may be strings; values are length-3 positions.
    """
    out: Dict[int, np.ndarray] = {}
    for lm_id, xyz in m.items():
        try:
            ik = int(lm_id)
        except (TypeError, ValueError):
            continue
        try:
            arr = np.asarray(xyz, dtype=float).reshape(3)
        except Exception:
            continue
        out[ik] = arr
    return out



def bounded_rv(mu=0.0, sigma=1.0, n_samples=1, seed=None, rng: Optional[np.random.Generator] = None):
    """
    Sample a 1D bounded random variable by:
      - sampling uniformly in a 2D disk
      - projecting onto the x-axis

    Result:
      mean = mu
      variance = sigma^2
      support = [mu - 2*sigma, mu + 2*sigma]
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    R = 2.0 * sigma

    # Uniform angle
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)

    # Correct radial distribution for uniform-in-area disk
    rho = R * np.sqrt(rng.uniform(0.0, 1.0, size=n_samples))

    # x-projection
    x = rho * np.cos(theta)

    # shift to desired mean
    return mu + x


def sample_additive_noise_zero_mean(
    std: np.ndarray,
    noise_on: bool,
    bounded_noise: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Independent zero-mean additive noise with per-component std `std`.

    - noise_on False: no perturbation (zeros).
    - noise_on True, bounded_noise False: Gaussian N(0, sigma^2) per axis.
    - noise_on True, bounded_noise True: bounded disk-projection noise per axis
      (support ±2*sigma, variance sigma^2).
    """
    std_arr = np.asarray(std, dtype=float)
    out_shape = std_arr.shape
    flat = std_arr.reshape(-1)
    n = flat.size
    if not noise_on:
        return np.zeros(n, dtype=float).reshape(out_shape)
    if not bounded_noise:
        return (flat * rng.standard_normal(n)).reshape(out_shape)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        if flat[i] <= 0.0:
            continue
        R = 2.0 * flat[i]
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        rho = R * float(np.sqrt(rng.uniform(0.0, 1.0)))
        out[i] = rho * np.cos(theta)
    return out.reshape(out_shape)


def _make_rng(drone_id: str, salt: str = "") -> np.random.Generator:
    """Stable seed from drone id + salt (for reproducible noise streams per component)."""
    try:
        d = int(str(drone_id))
    except ValueError:
        d = sum(ord(c) for c in str(drone_id))
    h = sum(ord(c) for c in salt)
    seed = (d * 1_000_003 + h) % (2**32)
    return np.random.default_rng(seed)


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
        config_path: str = "python/multirobot_simulator/config/sim_config.json",
        trajectory_path: str = None,
        log_path: str = None,
        N_DRONES: int = None,
        yaw0 = 0,
        verbose = False,
        landmark_path: Optional[str] = None,
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

        # cur_path = Path(__file__).resolve().parent
        # config_path = os.path.join(cur_path, config_path)
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        dt = cfg['dt']

        bots = cfg['bots']

        dids = list(bots.keys())

        if N_DRONES is None:
            if verbose: print("Creating simulation for ", len(dids), " Drones")
            N_DRONES = len(dids)
        elif N_DRONES > len(dids):
            if verbose:print("N_DRONES too large, setting to len(dids) = ", len(dids))
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
                # log_path = os.path.join(cur_path, log_path)
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
                landmark_path=landmark_path,
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
        # Invalidate the pose cache so each drone's sensors see an up-to-date `self` pose.
        #
        # Bug we avoid: previously only the first drone (e.g. "0") rebuilt the cache after
        # integrating; later drones hit synced=True and reused a snapshot where their own
        # entry was still the *pre-step* state. That makes lmobs_rp / relpos inconsistent
        # with the GT trajectory (and looks like "only drone 0 is correct").
        #
        # Note: within one world step, robots with a higher sort order than the observer
        # still have not integrated yet, so inter-robot relative pose can be off by one dt
        # toward those neighbors; fixing that fully would require a two-phase step
        # (integrate all, then run all sensors once).
        self.synced = False
        for d in self.drone_sims:
            d.step()
            self.synced = False
    
    def reached_dest_all(self):
        for d in self.drone_sims:
            if not d.reached_dest(): return False
        return True


def _upper_triangle_6x6(m: np.ndarray) -> list:
    """Return the 21 upper-triangle entries (row-major) of a 6x6 matrix."""
    vals = []
    for r in range(6):
        for c in range(r, 6):
            vals.append(float(m[r, c]))
    return vals


def _cov_from_error_std(std: list) -> np.ndarray:
    """
    Build a 6x6 covariance matrix from an error_std list.
    Supported lengths:
      - 3: [sx, sy, sz]            -> diag(x,y,z)
      - 4: [sx, sy, sz, syaw]      -> diag(x,y,z,yaw)
      - 6: [sx, sy, sz, srx, sry, srz] -> diag(x,y,z,roll,pitch,yaw)
    """
    s = np.asarray(std, dtype=float).flatten()
    info = np.zeros((6, 6), dtype=float)
    if s.size >= 3:
        info[0, 0] = 1/s[0] ** 2
        info[1, 1] = 1/s[1] ** 2
        info[2, 2] = 1/s[2] ** 2
    if s.size == 4:
        info[5, 5] = 1/s[3] ** 2
    elif s.size >= 6:
        info[3, 3] = 1/s[3] ** 2
        info[4, 4] = 1/s[4] ** 2
        info[5, 5] = 1/s[5] ** 2
    return info


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
    def __init__(
        self,
        world_sim,
        drone_id: str,
        config: dict,
        writer: LineWriter,
        landmarks: Dict[int, np.ndarray] | None = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.drone_id = drone_id
        self.config = config
        self.world_sim = world_sim
        self.gps_sensor = None
        self.rel_pose_sensor = None
        self.lm_rel_pos_sensor = None
        self.writer = writer
        self.landmarks = landmarks if landmarks is not None else {}
        self.rng = rng if rng is not None else _make_rng(drone_id, "sensors")

        sensors_cfg = config["sensors"]

        # --- GPS ---
        gps_cfg = sensors_cfg.get("gps")
        if gps_cfg and gps_cfg.get("active", False):
            # gps.error_std is [σx, σz, ?] in config. We only use first 2.
            gps_err_std = np.array(gps_cfg["error_std"][:2], dtype=float)
            gps_noise_on = bool(gps_cfg.get("noise_on", True))
            gps_bounded = bool(gps_cfg.get("bounded_noise", False))
            self.gps_sensor = GPSSensor(
                drone_id=drone_id,
                frequency=float(gps_cfg["frequency"]),
                error_std=gps_err_std,
                noise_on=gps_noise_on,
                bounded_noise=gps_bounded,
                rng=self.rng,
            )

        # --- Relative pose / bot observer ---
        bot_obs_cfg = sensors_cfg.get("bot_observer")
        if bot_obs_cfg and bot_obs_cfg.get("active", False):
            max_range = bot_obs_cfg["range"][0][1]  # [0, R] -> take R
            err_std_arr = np.array(bot_obs_cfg["error_std"], dtype=float)
            bot_noise_on = bool(bot_obs_cfg.get("noise_on", True))
            bot_bounded = bool(bot_obs_cfg.get("bounded_noise", False))

            # TODO for later
            # Pad to 4 elems [σx,σy,σz,σyaw] (config gives you 3)
            # if err_std_arr.shape[0] == 3:
            #     err_std_arr = np.concatenate([err_std_arr, [0.0]])

            self.rel_pose_sensor = RelPosSensor(
                drone_id=drone_id,
                frequency=float(bot_obs_cfg["frequency"]),
                error_std=err_std_arr,
                sensor_range=float(max_range),
                range_dependent_error=False,  # could come from cfg if you add it
                noise_on=bot_noise_on,
                bounded_noise=bot_bounded,
                rng=self.rng,
            )

        # --- Landmark relative-position observer ---
        lm_obs_cfg = sensors_cfg.get("lm_observer")
        if lm_obs_cfg and lm_obs_cfg.get("active", False):
            max_range = lm_obs_cfg["range"][0][1]  # [0, R] -> take R
            # For landmark relative-position sensor we use [sx, sy, sz].
            lm_err_std = np.array(lm_obs_cfg.get("error_std", [0.0, 0.0, 0.0])[:3], dtype=float)
            lm_noise_on = bool(lm_obs_cfg.get("noise_on", True))
            lm_bounded = bool(lm_obs_cfg.get("bounded_noise", False))
            self.lm_rel_pos_sensor = LandmarkRelPosSensor(
                drone_id=drone_id,
                frequency=float(lm_obs_cfg["frequency"]),
                error_std=lm_err_std,
                sensor_range=float(max_range),
                landmarks=self.landmarks,
                range_dependent_error=False,
                noise_on=lm_noise_on,
                bounded_noise=lm_bounded,
                rng=self.rng,
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

        # Landmark relative position
        if self.lm_rel_pos_sensor:
            lm_msg = self.lm_rel_pos_sensor.step(dt, positions)
            if lm_msg:
                self.writer.write_line(lm_msg)


# When activated, queries world sim for location of host
class GPSSensor:
    def __init__(
        self,
        drone_id: str,
        frequency: float,
        error_std: np.ndarray,
        noise_on: bool = True,
        bounded_noise: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        self.period = 1 / frequency
        self.error_std = error_std
        self.noise_on = noise_on
        self.bounded_noise = bounded_noise
        self.rng = rng if rng is not None else np.random.default_rng()
        self.timer = 0
        self.last_io = 0
        self.drone_id = drone_id

    def step(self, dt, positions):
        io = None
        self.timer += dt
        if self.timer > self.period + self.last_io:
            real_pos = np.array([positions[self.drone_id][0], positions[self.drone_id][1]])
            noise = sample_additive_noise_zero_mean(
                self.error_std, self.noise_on, self.bounded_noise, self.rng
            ).reshape(2)
            noisy_pos = real_pos + noise
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
    def __init__(
        self,
        drone_id: str,
        frequency: float,
        error_std: np.ndarray,
        sensor_range: float,
        range_dependent_error=False,
        noise_on: bool = True,
        bounded_noise: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        self.period = 1 / frequency
        self.drone_id = drone_id
        # This is 4 elements
        self.error_std = error_std
        self.range_dependent_error = range_dependent_error
        self.noise_on = noise_on
        self.bounded_noise = bounded_noise
        self.rng = rng if rng is not None else np.random.default_rng()
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
                es = np.asarray(self.error_std, dtype=float).ravel()
                if es.size < 4:
                    es = np.pad(es, (0, 4 - int(es.size)))
                terr = es[:4].copy()
                if self.range_dependent_error:
                    terr = terr * (2 * dist / self.sensor_range)
                noise = sample_additive_noise_zero_mean(
                    terr, self.noise_on, self.bounded_noise, self.rng
                ).reshape(4)
                noisy_reading = rel_pose + noise
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


class LandmarkRelPosSensor:
    """
    Landmark relative-position observer.
    Emits message type:
      t lmobs_rp landmark_id rel_x rel_y rel_z std_x std_y std_z
    where relative position is in observer body frame (yaw-only rotation).
    """

    def __init__(
        self,
        drone_id: str,
        frequency: float,
        error_std: np.ndarray,
        sensor_range: float,
        landmarks: Dict[int, np.ndarray],
        range_dependent_error: bool = False,
        noise_on: bool = True,
        bounded_noise: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        self.period = 1 / frequency
        self.drone_id = drone_id
        self.error_std = np.asarray(error_std, dtype=float).reshape(3)
        self.range_dependent_error = range_dependent_error
        self.noise_on = noise_on
        self.bounded_noise = bounded_noise
        self.rng = rng if rng is not None else np.random.default_rng()
        self.sensor_range = sensor_range
        self.sensor_range_sqr = sensor_range ** 2
        self.landmarks = landmarks
        self.timer = 0.0
        self.last_io = 0.0

    def step(self, dt, positions):
        io = None
        self.timer += dt
        if self.timer > self.period + self.last_io:
            io = ""
            sx, sy, sz, syaw, _, _, _, _ = positions[self.drone_id]
            for lm_id, lm_pos in self.landmarks.items():
                wc_rel_pos = np.asarray(lm_pos, dtype=float) - np.array([sx, sy, sz], dtype=float)
                dist_sqr = float(np.sum(wc_rel_pos ** 2))
                if dist_sqr > self.sensor_range_sqr:
                    continue
                dist = np.sqrt(dist_sqr)
                rel_pos = np.array(
                    [
                        wc_rel_pos[0] * np.cos(-syaw) - wc_rel_pos[1] * np.sin(-syaw),
                        wc_rel_pos[0] * np.sin(-syaw) + wc_rel_pos[1] * np.cos(-syaw),
                        wc_rel_pos[2],
                    ],
                    dtype=float,
                )
                terr = self.error_std.copy()
                if self.range_dependent_error:
                    terr *= 2 * dist / self.sensor_range
                noise = sample_additive_noise_zero_mean(
                    terr, self.noise_on, self.bounded_noise, self.rng
                ).reshape(3)
                noisy_reading = rel_pos + noise
                io = io + self.generate_io(lm_id, noisy_reading, terr)
            self.last_io += self.period
        return io

    def generate_io(self, lm_id: int, io_vals: np.ndarray, e: np.ndarray):
        if io_vals is None:
            return None
        return (
            f"{self.timer} lmobs_rp {int(lm_id)} "
            f"{io_vals[0]} {io_vals[1]} {io_vals[2]} "
            f"{e[0]} {e[1]} {e[2]}\n"
        )






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
                 yaw0 = None,
                 landmark_path: Optional[str] = None):
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

        # Shared RNG for odometry noise and sensors (order: record_odom, then sensor_manager.step)
        self._rng = _make_rng(drone_id, "drone")

        # Odometry noise std: [v_fwd, vz, omega] (3-element ndarray)
        ctrl_cfg = bot_cfg["controller"]
        raw = ctrl_cfg.get("odom_error_std", [0.0, 0.0, 0.0])
        self.odom_error_std = np.array(raw, dtype=float).reshape(3)
        self.odom_noise_on = bool(ctrl_cfg.get("odom_noise_on", True))
        self.odom_bounded_noise = bool(ctrl_cfg.get("bounded_noise", False))

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

        # ---------- Initialization message (first log line) ----------
        # Can be provided either globally (config["initialization"]) or per-bot (bot_cfg["initialization"]).
        init_cfg = {}
        if isinstance(config.get("initialization"), dict):
            init_cfg.update(config["initialization"])
        if isinstance(bot_cfg.get("initialization"), dict):
            init_cfg.update(bot_cfg["initialization"])

        # Default behavior (no initialization config at all):
        #   fixed=true, pose=(0,0,0,yaw=0), covariance=I
        default_init = bool(init_cfg.get("default_init", init_cfg == {}))
        fixed_init = bool(init_cfg.get("fixed_init", False))
        init_error_std = init_cfg.get("error_std", init_cfg.get("init_error_std", [1.0, 1.0, 1.0, 1.0]))

        if default_init:
            # zeros initial pose, fixed=true, identity covariance.
            self.s[:] = np.zeros(8, dtype=float)
            init_x, init_y, init_z, init_yaw = 0.0, 0.0, 0.0, 0.0
            fixed = True
            cov = np.eye(6, dtype=float)
        elif fixed_init:
            # Use initial pose, fixed=true, identity covariance.
            init_x, init_y, init_z, init_yaw = float(self.s[0]), float(self.s[1]), float(self.s[2]), float(self.s[3])
            fixed = True
            cov = np.eye(6, dtype=float)
        else:
            # Use initial pose, fixed=false, covariance from error_std.
            init_x, init_y, init_z, init_yaw = float(self.s[0]), float(self.s[1]), float(self.s[2]), float(self.s[3])
            fixed = False
            cov = _cov_from_error_std(init_error_std)

        quat_xyzw = yaw_to_quat_xyzw(init_yaw)  # [x,y,z,w]
        cov_ut = _upper_triangle_6x6(cov)
        fixed_str = "true" if fixed else "false"
        init_line = (
            f"init {self.drone_id} {init_x} {init_y} {init_z} "
            f"{quat_xyzw[0]} {quat_xyzw[1]} {quat_xyzw[2]} {quat_xyzw[3]} "
            f"{fixed_str} " + " ".join(map(str, cov_ut)) + "\n"
        )
        self.message_writer.write_line(init_line)

        # Log initial GT pose at t=0.0 so GT and SLAM trajectories start aligned.
        # GT line format matches what the SLAM side expects in DataBasedSimulation::readNextGT():
        #   t pose x y z yaw, vx, vy, vz, r
        x0, y0, z0, yaw0_log = float(self.s[0]), float(self.s[1]), float(self.s[2]), float(self.s[3])
        vx0, vy0, vz0, r0 = float(self.s[4]), float(self.s[5]), float(self.s[6]), float(self.s[7])
        gt_line0 = (
            f"{self.timer} pose {x0} {y0} {z0} {yaw0_log}, "
            f"{vx0}, {vy0}, {vz0}, {r0}\n"
        )
        self.gt_writer.write_line(gt_line0)

        # Landmarks: optional override path (e.g. batch run dir landmarks.json), then
        # inline config["landmarks"], then config["landmark_path"].
        # IDs are always int in memory and in msg_log as decimal integers.
        landmarks: Dict[int, np.ndarray] = {}
        lm_override = Path(landmark_path) if landmark_path else None
        if lm_override is not None and lm_override.exists():
            with lm_override.open("r", encoding="utf-8") as lf:
                lm_cfg = json.load(lf)
            if isinstance(lm_cfg, dict):
                landmarks = normalize_landmarks_mapping(lm_cfg)
        elif isinstance(config.get("landmarks"), dict):
            landmarks = normalize_landmarks_mapping(config["landmarks"])
        elif "landmark_path" in config:
            lm_path = Path(str(config["landmark_path"]))
            if lm_path.exists():
                with lm_path.open("r", encoding="utf-8") as lf:
                    lm_cfg = json.load(lf)
                if isinstance(lm_cfg, dict):
                    landmarks = normalize_landmarks_mapping(lm_cfg)

        if not world_sim is None:
            self.sensor_manager = SensorSimManager(
                world_sim,
                self.drone_id,
                bot_cfg,
                self.message_writer,
                landmarks=landmarks,
                rng=self._rng,
            )
        else:
            self.sensor_manager = None


    def close(self):
        # Remove all circular pointers
        self.world_sim = None

    def step(self):
        x, y, z, yaw, vx, vy, vz, r = self.s

        self.timer  += self.dt

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
        

        # Body forward in horizontal plane (X–Y); world +Z up; yaw about +Z
        vx = v_fwd * np.cos(yaw)
        vy = v_fwd * np.sin(yaw)
        vz = vz_cmd

        # Integrate pose
        x  += self.dt * vx
        y  += self.dt * vy
        z  += self.dt * vz

        # Yaw kinematics
        r   = omega_cmd
        yaw += self.dt * r
        yaw = (yaw + np.pi) % (2*np.pi) - np.pi

        # log

        self.s[:] = [x, y, z, yaw, vx, vy, vz, r]
        line = f"{self.timer} pose {x} {y} {z} {yaw}, {vx}, {vy}, {vz}, {r}\n"
        self.gt_writer.write_line(line)

        # need to watch out, floating point problems. 
        if not self.sensor_manager is None:
            self.sensor_manager.step(self.dt)
    
    def record_odom(self, v_fwd, v_up, omega_yaw):
        """Log body forward speed, world-up (+Z) speed, and yaw rate about +Z (matches C++ Z-up frame)."""
        # odom_noise_on: whether to perturb logged odom; bounded_noise: Gaussian vs bounded when on.
        nvec = sample_additive_noise_zero_mean(
            self.odom_error_std,
            self.odom_noise_on,
            self.odom_bounded_noise,
            self._rng,
        ).reshape(3)
        noise_fwd = float(v_fwd + nvec[0])
        noise_up = float(v_up + nvec[1])
        noise_omega = float(omega_yaw + nvec[2])

        std_fwd, std_up, std_omega = self.odom_error_std
        # Log odometry uncertainty as variances (NOT information).
        # C++ DataBasedSimulation will decode these variances into a 6x6 information matrix.
        var_fwd = float(std_fwd * std_fwd)
        var_up = float(std_up * std_up)
        var_yaw = float(std_omega * std_omega)

        # Format: t odom v_fwd v_up omega var_fwd var_up var_yaw
        line = (
            f"{self.timer} odom {noise_fwd} {noise_up} {noise_omega} "
            f"{var_fwd} {var_up} {var_yaw}\n"
        )
        self.message_writer.write_line(line)

    def reached_dest(self):
        return self.ctrl.terminal

        