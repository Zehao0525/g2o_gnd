from typing import List, Tuple, Optional
import csv
import math

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R

import numpy as np

import evo
from evo.core import trajectory,metrics,sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core.metrics import APE
from evo.tools import log, plot
from evo.core.transformations import quaternion_from_euler

def read_se2_vertices(filename):
    se2_poses = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("TUTORIAL_VERTEX_SE2"):
                parts = line.strip().split()
                vertex_id = int(parts[1])
                x, y, theta = map(float, parts[2:5])
                se2_poses.append((vertex_id, x, y, theta))
    return sorted(se2_poses, key=lambda v: v[0])  # sort by vertex id

def read_se3_vertices_as_se2(filename, idbound = None, realign = False):
    pose_dict = {}  # vertex_id -> (x, y, theta)
    if idbound:
        idlow,idup = idbound
    else:
        idlow,idup = (-1, 100000000000000)

    initialized = False
    with open(filename, 'r') as file:
        for vid,line in enumerate(file):
            if line.startswith("VERTEX_SE3:QUAT"):
                tokens = line.strip().split()
                vertex_id = int(tokens[1])
                x, y = float(tokens[2]), float(tokens[3])
                qx, qy, qz, qw = map(float, tokens[5:9])

                # Convert quaternion to rotation matrix
                r = R.from_quat([qx, qy, qz, qw])
                rot_matrix = r.as_matrix()

                # Extract yaw angle
                theta = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])

                if not initialized:
                    initialized = True
                    print(x,y,theta)
                    if realign:
                        initial_x = x
                        initial_y = y
                        initial_theta = theta
                    else:
                        initial_x = 0
                        initial_y = 0
                        initial_theta = 0

                # Store / overwrite to ensure unique IDs
                if vertex_id >=idlow and vertex_id < idup:
                    x_diff = x-initial_x
                    y_diff = y-initial_y
                    theta_diff = theta-initial_theta
                    # Store / overwrite to ensure unique IDs
                    pose_dict[vertex_id] = (x_diff * np.cos(-initial_theta) - y_diff * np.sin(-initial_theta), 
                                            x_diff * np.sin(-initial_theta) + y_diff * np.cos(-initial_theta), 
                                            np.arctan2(np.sin(theta_diff), np.cos(theta_diff)))

    # Convert dict to sorted list of tuples (id, x, y, theta)
    return [(vid, *pose_dict[vid]) for vid in sorted(pose_dict)]


def read_tum_vertices_as_se2(filename):
    pose_dict = {}  # vertex_id -> (x, y, theta)

    with open(filename, 'r') as file:
        for vertex_id,line in enumerate(file):
            tokens = line.strip().split()
            x, y = float(tokens[1]), float(tokens[2])
            qx, qy, qz, qw = map(float, tokens[4:8])

            # Convert quaternion to rotation matrix
            r = R.from_quat([qx, qy, qz, qw])
            rot_matrix = r.as_matrix()

            # Extract yaw angle
            theta = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])

            if vertex_id == 0:
                print(x,y,theta)
                initial_x = x
                initial_y = y
                initial_theta = theta

            x_diff = x-initial_x
            y_diff = y-initial_y
            theta_diff = theta-initial_theta
            # Store / overwrite to ensure unique IDs
            pose_dict[tokens[0]] = (x_diff * np.cos(-initial_theta) - y_diff * np.sin(-initial_theta), 
                                    x_diff * np.sin(-initial_theta) + y_diff * np.cos(-initial_theta), 
                                    np.arctan2(np.sin(theta_diff), np.cos(theta_diff)))

    # Convert dict to sorted list of tuples (id, x, y, theta)
    return [(ts, *pose_dict[ts]) for ts in sorted(pose_dict)]

def plot_trajectory(se2_poses, traj_color = 'gray', plot_direction = True, label = 'Trajectory', alpha = 1.0):
    xs = [x for (_, x, _, _) in se2_poses]
    ys = [y for (_, _, y, _) in se2_poses]
    thetas = [theta for (_, _, _, theta) in se2_poses]

    # Compute dynamic arrow length
    arrow_length = 0.1

    # Normalize index for colormap
    num = len(se2_poses)
    colors = cm.viridis(np.linspace(0, 1, num))  # Use any colormap: viridis, plasma, jet, etc.
    plt.plot(xs, ys, linestyle='-', color=traj_color, label=label, alpha=alpha)
    if plot_direction:
        for i in range(num):
            dx = arrow_length * np.cos(thetas[i])
            dy = arrow_length * np.sin(thetas[i])
            plt.quiver(xs[i], ys[i], dx, dy,
                    angles='xy', scale_units='xy', scale=1, color=colors[i])
            
def plot_landmarks(se2_poses, color='gray', label='lm', show_id = True):
    xs = [x for (_, x, _, _) in se2_poses]
    ys = [y for (_, _, y, _) in se2_poses]
    ids = [i for (i, _, _, _) in se2_poses]

    plt.scatter(xs, ys, color=color, label=label, marker='x')

    # Annotate each point with its ID
    if show_id:
        for x, y, i in zip(xs, ys, ids):
            plt.text(x, y, str(i), fontsize=8, ha='right', va='bottom')

def compute_ape(traj1, traj2):
    np_traj1 = np.array(traj1)
    np_traj2 = np.array(traj2)
    coord_diffs = np_traj1[:,1:3] - np_traj2[:,1:3]
    coord_diffs_sqr = coord_diffs**2
    sqr_dists = coord_diffs_sqr.sum(axis = 1)
    return np.mean(sqr_dists)

def list_to_pose_path(xy_list, dt = 0.1):
    """
    Converts (x, y) list to PosePath3D with zero z and identity rotation.
    """
    positions_xyz = np.array([[x, y, 0.0] for _,x, y,_ in xy_list])
    orientations_quat_wxyz = np.tile([1.0, 0.0, 0.0, 0.0], (len(xy_list), 1))  # identity quaternions
    timestamps = np.arange(len(xy_list)) * dt  # dummy timestamps spaced by dt
    return PoseTrajectory3D(positions_xyz=positions_xyz,
                            orientations_quat_wxyz=orientations_quat_wxyz,
                            timestamps=timestamps)

# -----------------------------
# Core helpers
# -----------------------------





def _make_pose_traj_from_array_with_timestamps(arr):
    """
    arr: shape (N, M), first column is timestamp (seconds).
         Columns:
           - if M >= 8: [t, x, y, z, qx, qy, qz, qw]
           - if M >= 4: [t, x, y, z] (orientation -> identity)
    """
    a = np.asarray(arr)
    if a.ndim != 2 or a.shape[1] < 4:
        raise ValueError("traj_ref must be (N, >=4) with timestamps in col 0.")
    ts = a[:, 0].astype(float)

    positions = []
    quats = []

    for _, x, y, theta in arr:
        positions.append([x, y, 0.0])
        quats.append(quaternion_from_euler(0.0, 0.0, theta))  # [x, y, z, w]
    return trajectory.Trajectory(np.asarray(positions), np.asarray(quats), ts)


def _make_pose_traj_from_array_uniform_time(arr, freq_hz, t0):
    """
    arr: shape (N, M), NO timestamps.
         Columns:
           - if M >= 7: [x, y, z, qx, qy, qz, qw]
           - if M >= 3: [x, y, z]
    freq_hz: sampling rate to synthesize timestamps
    t0: start time (seconds)
    """
    a = np.asarray(arr)
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError("traj_est must be (N, >=3) without timestamps.")
    n = a.shape[0]
    if freq_hz is None or freq_hz <= 0:
        raise ValueError("freq_est must be > 0 for timestamp-less traj_est.")
    ts = t0 + np.arange(n, dtype=float) / float(freq_hz)

    positions = []
    quats = []

    for _, x, y, theta in arr:
        positions.append([x, y, 0.0])
        quats.append(quaternion_from_euler(0.0, 0.0, theta))  # [x, y, z, w]
    return trajectory.Trajectory(np.asarray(positions), np.asarray(quats), ts)

def compute_frequency_ratio(traj_ref, traj_est):
    """
    'Frequency difference': ratio of lengths.
    >1 means ref has more samples, <1 means est has more.
    """
    n_ref = len(traj_ref)
    n_est = len(traj_est)
    ratio_ref_over_est = float(n_ref) / float(n_est) if n_est > 0 else float("inf")
    ratio_est_over_ref = float(n_est) / float(n_ref) if n_ref > 0 else float("inf")
    return {
        "n_ref": n_ref,
        "n_est": n_est,
        "ref_over_est": ratio_ref_over_est,
        "est_over_ref": ratio_est_over_ref,
    }


# ---------- Main: build evo trajectories, sync, and compute APE ----------

def ref_list_to_evo(traj_ref):
    return _make_pose_traj_from_array_with_timestamps(traj_ref)

def est_to_evo_with_ref(
    t_ref, traj_est, freq_ref=None, freq_est=None):
    """
    Adjusted version:
      - traj_ref HAS timestamps in column 0 (seconds).
      - traj_est has NO timestamps; we synthesize them from freq_est.

    Inputs
    ------
    traj_ref : array-like (N, >=4)   [t, x, y, z, (qx, qy, qz, qw optional)]
    traj_est : array-like (M, >=3)   [x, y, z, (qx, qy, qz, qw optional)]
    freq_ref : float or None         (used only for defaults; we infer from ref if None)
    freq_est : float or None         (if None, inferred from ref * length ratio)
    t_max_diff : float or None       association tolerance (seconds)
    interpolate : bool               (flag kept for API; association is nearest-neighbor here)

    Returns
    -------
    dict with 'frequency_ratio', 't_max_diff', 'interpolated', APE stats, and objects.
    """
    # --- frequency ratio purely by lengths (as before)
    n_ref = t_ref.num_poses
    n_est = len(traj_est)
    ratio_est_over_ref = float(n_est) / float(n_ref) if n_ref > 0 else float("inf")


    # Derive ref rate from timestamps if not given
    if freq_ref is None:
        if len(t_ref.timestamps) >= 2:
            dt_ref = np.median(np.diff(t_ref.timestamps))
            freq_ref = 1.0 / dt_ref if dt_ref > 0 else 10.0
        else:
            freq_ref = 10.0

    # Infer est rate if not provided: scale ref rate by length ratio
    if freq_est is None:
        freq_est = max(1e-6, freq_ref * ratio_est_over_ref)

    # Synthesize timestamps for estimate starting at ref's first timestamp
    t0 = float(t_ref.timestamps[0]) if len(t_ref.timestamps) else 0.0
    t_est = _make_pose_traj_from_array_uniform_time(traj_est, freq_hz=freq_est, t0=t0)

    return t_ref, t_est


def compute_ape_with_evo_traj(
    t_ref, t_est, freq_est=None, t_max_diff=None, interpolate=False
):
    """
    Adjusted version:
      - traj_ref HAS timestamps in column 0 (seconds).
      - traj_est has NO timestamps; we synthesize them from freq_est.

    Inputs
    ------
    traj_ref : array-like (N, >=4)   [t, x, y, z, (qx, qy, qz, qw optional)]
    traj_est : array-like (M, >=3)   [x, y, z, (qx, qy, qz, qw optional)]
    freq_ref : float or None         (used only for defaults; we infer from ref if None)
    freq_est : float or None         (if None, inferred from ref * length ratio)
    t_max_diff : float or None       association tolerance (seconds)
    interpolate : bool               (flag kept for API; association is nearest-neighbor here)

    Returns
    -------
    dict with 'frequency_ratio', 't_max_diff', 'interpolated', APE stats, and objects.
    """

    # Derive ref rate from timestamps if not given
    if freq_ref is None:
        if len(t_ref.timestamps) >= 2:
            dt_ref = np.median(np.diff(t_ref.timestamps))
            freq_ref = 1.0 / dt_ref if dt_ref > 0 else 10.0
        else:
            freq_ref = 10.0

    # Choose a sensible default association tolerance if not provided:
    if t_max_diff is None:
        dt_ref_med = np.median(np.diff(t_ref.timestamps)) if len(t_ref.timestamps) >= 2 else (1.0 / freq_ref)
        dt_est_med = np.median(np.diff(t_est.timestamps)) if len(t_est.timestamps) >= 2 else (1.0 / freq_est)
        t_max_diff = 0.5 * max(dt_ref_med, dt_est_med)

    # Time association (nearest neighbor). If you truly need interpolation,
    # you could resample one trajectory onto the other's timeline.
    t_ref_sync, t_est_sync = sync.associate_trajectories(
        t_ref, t_est, max_diff=t_max_diff
    )

    # Compute APE (translation component)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    # Align and/or correct scale can be requested here if desired:
    # ape.process_data((t_ref_sync, t_est_sync), align=True, correct_scale=True)
    ape.process_data((t_ref_sync, t_est_sync))

    return {
        "t_max_diff": t_max_diff,
        "interpolated": bool(interpolate),
        "rmse": ape.get_statistic(metrics.StatisticsType.rmse),
        "mean": ape.get_statistic(metrics.StatisticsType.mean),
        "median": ape.get_statistic(metrics.StatisticsType.median),
        "max": ape.get_statistic(metrics.StatisticsType.max),
        "std": ape.get_statistic(metrics.StatisticsType.std),
        "data": ape,
        "traj_ref": t_ref,
        "traj_est": t_est,
    }



def compute_ape_with_evo(
    traj_ref, traj_est, freq_ref=None, freq_est=None, t_max_diff=None, interpolate=False, align = False
):
    """
    Adjusted version:
      - traj_ref HAS timestamps in column 0 (seconds).
      - traj_est has NO timestamps; we synthesize them from freq_est.

    Inputs
    ------
    traj_ref : array-like (N, >=4)   [t, x, y, z, (qx, qy, qz, qw optional)]
    traj_est : array-like (M, >=3)   [x, y, z, (qx, qy, qz, qw optional)]
    freq_ref : float or None         (used only for defaults; we infer from ref if None)
    freq_est : float or None         (if None, inferred from ref * length ratio)
    t_max_diff : float or None       association tolerance (seconds)
    interpolate : bool               (flag kept for API; association is nearest-neighbor here)

    Returns
    -------
    dict with 'frequency_ratio', 't_max_diff', 'interpolated', APE stats, and objects.
    """
    # --- frequency ratio purely by lengths (as before)
    freq_ratio = compute_frequency_ratio(traj_ref, traj_est)

    # Build evo trajectory for reference FROM ITS TIMESTAMPS
    t_ref = _make_pose_traj_from_array_with_timestamps(traj_ref)

    # Derive ref rate from timestamps if not given
    if freq_ref is None:
        if len(t_ref.timestamps) >= 2:
            dt_ref = np.median(np.diff(t_ref.timestamps))
            freq_ref = 1.0 / dt_ref if dt_ref > 0 else 10.0
        else:
            freq_ref = 10.0

    # Infer est rate if not provided: scale ref rate by length ratio
    if freq_est is None:
        freq_est = max(1e-6, freq_ref * freq_ratio["est_over_ref"])

    # Synthesize timestamps for estimate starting at ref's first timestamp
    t0 = float(t_ref.timestamps[0]) if len(t_ref.timestamps) else 0.0
    t_est = _make_pose_traj_from_array_uniform_time(traj_est, freq_hz=freq_est, t0=t0)

    # Choose a sensible default association tolerance if not provided:
    if t_max_diff is None:
        dt_ref_med = np.median(np.diff(t_ref.timestamps)) if len(t_ref.timestamps) >= 2 else (1.0 / freq_ref)
        dt_est_med = np.median(np.diff(t_est.timestamps)) if len(t_est.timestamps) >= 2 else (1.0 / freq_est)
        t_max_diff = 0.5 * max(dt_ref_med, dt_est_med)

    # Time association (nearest neighbor). If you truly need interpolation,
    # you could resample one trajectory onto the other's timeline.
    t_ref_sync, t_est_sync = sync.associate_trajectories(
        t_ref, t_est, max_diff=t_max_diff
    )

    if align:
        t_est_sync.align(t_ref_sync)

    # Compute APE (translation component)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    # Align and/or correct scale can be requested here if desired:
    # ape.process_data((t_ref_sync, t_est_sync), align=True, correct_scale=True)
    ape.process_data((t_ref_sync, t_est_sync))

    return {
        "frequency_ratio": freq_ratio,
        "t_max_diff": t_max_diff,
        "interpolated": bool(interpolate),
        "rmse": ape.get_statistic(metrics.StatisticsType.rmse),
        "mean": ape.get_statistic(metrics.StatisticsType.mean),
        "median": ape.get_statistic(metrics.StatisticsType.median),
        "max": ape.get_statistic(metrics.StatisticsType.max),
        "std": ape.get_statistic(metrics.StatisticsType.std),
        "data": ape,
        "traj_ref": t_ref,
        "traj_est": t_est,
    }







def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """
    Convert quaternion (x,y,z,w) to yaw (rotation about Z) using a rotation matrix
    to avoid convention mistakes. Returns angle in radians.
    """
    # normalize (defensive)
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n == 0.0:
        return 0.0
    x, y, z, w = qx/n, qy/n, qz/n, qw/n

    # Rotation matrix (from unit quaternion)
    R00 = 1 - 2*(y*y + z*z)
    R10 = 2*(x*y + z*w)
    # yaw = atan2(R[1,0], R[0,0])
    return math.atan2(R10, R00)

def csv_to_xytheta_list(csv_path: str, realign = False) -> List[Tuple[int, float, float, float]]:
    """
    Parse a DPGO trajectory CSV (pose_index,qx,qy,qz,qw,tx,ty,tz) and return
    a list of (id, x, y, theta) in the x–y plane (theta in radians).
    """
    out: List[Tuple[int, float, float, float]] = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        if realign:
            initialized = False
            initial_x, initial_y, initial_theta = 0,0,0
            for row in r:
                i = int(row["pose_index"])
                qx, qy, qz, qw = (float(row["qx"]), float(row["qy"]),
                                float(row["qz"]), float(row["qw"]))
                x, y = float(row["tx"]), float(row["ty"])
                theta = _quat_to_yaw(qx, qy, qz, qw)
                if not initialized:
                    initial_x, initial_y, initial_theta = x, y, theta
                    initialized = True
                x_diff = x-initial_x
                y_diff = y-initial_y
                theta_diff = theta-initial_theta
                # Store / overwrite to ensure unique IDs
                out.append((i,      x_diff * np.cos(-initial_theta) - y_diff * np.sin(-initial_theta), 
                                    x_diff * np.sin(-initial_theta) + y_diff * np.cos(-initial_theta), 
                                    np.arctan2(np.sin(theta_diff), np.cos(theta_diff))))
        else:
            for row in r:
                i = int(row["pose_index"])
                qx, qy, qz, qw = (float(row["qx"]), float(row["qy"]),
                                float(row["qz"]), float(row["qw"]))
                x, y = float(row["tx"]), float(row["ty"])
                theta = _quat_to_yaw(qx, qy, qz, qw)
                out.append((i, x, y, theta))
    # sort by id just in case
    out.sort(key=lambda t: t[0])
    return out










def plot_ape_colormap(traj_est, traj_ref, ape_metric, *, plot_mode="xy", show=True, fig=None, ax=None):
    """
    Plot reference trajectory (grey dashed) and estimated trajectory colored by APE.

    Parameters
    ----------
    traj_est : evo.core.trajectory.PoseTrajectory3D (or similar)
        Estimated trajectory (aligned or not—whatever you pass is what is drawn).
    traj_ref : evo.core.trajectory.PoseTrajectory3D
        Reference trajectory.
    ape_metric : evo.core.metrics.APE
        An APE metric instance after .process_data(...). Uses ape_metric.error.
    plot_mode : {"xy","xz","yz","xyz"}, optional
        Projection mode for plotting. Default "xy".
    show : bool, optional
        If True, call plt.show() at the end. Default True.
    fig, ax : optional
        Existing matplotlib figure/axis to draw into.

    Returns
    -------
    (fig, ax)
        The matplotlib figure and axis used for plotting.
    """
    # Map string to EVO's PlotMode enum
    mode_map = {
        "xy": plot.PlotMode.xy,
        "xz": plot.PlotMode.xz,
        "yz": plot.PlotMode.yz,
        "xyz": plot.PlotMode.xyz,
    }
    pmode = mode_map.get(plot_mode, plot.PlotMode.xy)

    # Figure / axis setup via EVO helpers
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plot.prepare_axis(fig, pmode)

    # Reference in grey dashed
    plot.traj(ax, pmode, traj_ref, '--', "gray", "reference")

    # Color limits from metric stats (fallback to min/max of error array)
    try:
        stats = ape_metric.get_all_statistics()
        vmin = float(stats.get("min", np.nanmin(ape_metric.error)))
        vmax = float(stats.get("max", np.nanmax(ape_metric.error)))
    except Exception:
        vmin = float(np.nanmin(ape_metric.error))
        vmax = float(np.nanmax(ape_metric.error))

    # Estimated trajectory colored by per-pose APE
    plot.traj_colormap(
        ax,
        traj_est,
        ape_metric.error,   # per-pose translational APE array
        pmode,
        min_map=vmin,
        max_map=vmax
    )

    ax.legend()
    if show:
        plt.show()

    return fig, ax