import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from scipy.stats import ttest_rel,wilcoxon

import os

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

def plot_trajectory(se2_poses, traj_color = 'gray', plot_direction = True, label = 'Trajectory'):
    xs = [x for (_, x, _, _) in se2_poses]
    ys = [y for (_, _, y, _) in se2_poses]
    thetas = [theta for (_, _, _, theta) in se2_poses]

    # Compute dynamic arrow length
    arrow_length = 0.1

    # Normalize index for colormap
    num = len(se2_poses)
    colors = cm.viridis(np.linspace(0, 1, num))  # Use any colormap: viridis, plasma, jet, etc.
    plt.plot(xs, ys, linestyle='-', color=traj_color, label=label)
    if plot_direction:
        for i in range(num):
            dx = arrow_length * np.cos(thetas[i])
            dy = arrow_length * np.sin(thetas[i])
            plt.quiver(xs[i], ys[i], dx, dy,
                    angles='xy', scale_units='xy', scale=1, color=colors[i])

def compute_ape(traj1, traj2):
    np_traj1 = np.array(traj1)
    np_traj2 = np.array(traj2)
    coord_diffs = np_traj1[:,1:3] - np_traj2[:,1:3]
    coord_diffs_sqr = coord_diffs**2
    sqr_dists = coord_diffs_sqr.sum(axis = 1)
    return np.mean(sqr_dists)

def evaluate_batch_results(root_dir="gnd_test_results", num_tests=30, save_path=None):
    results = []

    for i in range(num_tests):
        test_dir = os.path.join(root_dir, f"test_{i}")
        file_gauss = os.path.join(test_dir, "twb_gauss.g2o")
        file_gnd = os.path.join(test_dir, "twb_gnd.g2o")
        file_gt = os.path.join(test_dir, "twb_gt.g2o")

        try:
            poses_gauss = read_se2_vertices(file_gauss)
            poses_gnd = read_se2_vertices(file_gnd)
            poses_gt = read_se2_vertices(file_gt)

            if not (poses_gauss and poses_gnd and poses_gt):
                print(f"[WARNING] Missing or empty file(s) in test_{i}")
                continue

            ape_gauss = compute_ape(poses_gauss, poses_gt)
            ape_gnd = compute_ape(poses_gnd, poses_gt)

            results.append({
                "test_index": i,
                "APE_Gaussian": ape_gauss,
                "APE_GND": ape_gnd
            })

        except Exception as e:
            print(f"[ERROR] Failed to process test_{i}: {e}")

    df = pd.DataFrame(results)
    if save_path:
        df.to_pickle(save_path)
        print(f"[INFO] Results saved to {save_path}")

    return df


import os
import numpy as np
import matplotlib.pyplot as plt

def plot_gnd_vs_gps(root_dir, i, gps_filename=None, save_path=None, show=True):
    """
    Plot GND and GPS/GT trajectories for test i (root_dir/test_i).

    Looks for:
      - GND: 'twb_gnd.g2o'
      - GPS/GT: gps_filename if provided, else tries 'twb_gps.g2o' then 'twb_gt.g2o'

    Assumes read_se2_vertices(path) -> poses in one of:
      dict {id: (x,y,theta)}, list of (id,x,y,theta), list of (id,(x,y,theta)),
      list of (x,y,theta), ndarray Nx4 (id,x,y,theta) or Nx3 (x,y,theta),
      or list of dicts with keys id/x/y/theta.

    Returns:
        (fig, ax): Matplotlib figure and axes.
    """
    test_dir = os.path.join(root_dir, f"test_{i}")
    file_gnd = os.path.join(test_dir, "twb_gnd.g2o")

    # Decide GPS/GT file
    if gps_filename is None:
        gps_filename = "twb_gauss.g2o"
    file_gps = os.path.join(test_dir, gps_filename)
    if not os.path.exists(file_gps):
        raise FileNotFoundError(f"GPS/GT file not found: {file_gps}")

    if not os.path.exists(file_gnd):
        raise FileNotFoundError(f"GND file not found: {file_gnd}")

    # ---- Load poses (expects your existing helper) ----
    poses_gnd = read_se2_vertices(file_gnd)
    poses_gps = read_se2_vertices(file_gps)

    if poses_gnd is None or poses_gps is None:
        raise ValueError("Missing or empty pose data for GND and/or GPS/GT.")

    # ---- Robust coercion to {id: (x,y,theta)} ----
    def to_dict(poses):
        # Dict: {id: (x,y,theta)} or {id: {'x':..,'y':..,'theta':..}}
        if isinstance(poses, dict):
            d = {}
            for k, v in poses.items():
                pid = int(k)
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    x = float(v[0]); y = float(v[1]); th = float(v[2]) if len(v) >= 3 else 0.0
                elif isinstance(v, dict) and 'x' in v and 'y' in v:
                    x = float(v['x']); y = float(v['y']); th = float(v.get('theta', v.get('th', 0.0)))
                else:
                    raise ValueError(f"Unsupported value for key {k}: {type(v)}")
                d[pid] = (x, y, th)
            return d

        # NumPy array
        if hasattr(poses, 'shape'):
            arr = np.asarray(poses)
            if arr.ndim != 2:
                raise ValueError(f"Ndarray must be 2D, got shape {arr.shape}")
            if arr.shape[1] == 4:  # (id,x,y,theta)
                return {int(arr[i,0]): (float(arr[i,1]), float(arr[i,2]), float(arr[i,3]))
                        for i in range(arr.shape[0])}
            if arr.shape[1] == 3:  # (x,y,theta), synthesize ids by index
                return {i: (float(arr[i,0]), float(arr[i,1]), float(arr[i,2]))
                        for i in range(arr.shape[0])}
            raise ValueError(f"Ndarray with unsupported number of columns: {arr.shape[1]}")

        # Iterable (list/tuple)
        d = {}
        for idx, item in enumerate(poses):
            # (id, x, y, theta)
            if isinstance(item, (list, tuple)) and len(item) == 4 and all(
                isinstance(x, (int, float, np.integer, np.floating)) for x in item[1:]
            ):
                pid, x, y, th = item
                d[int(pid)] = (float(x), float(y), float(th))
                continue

            # (id, (x,y,theta))
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (list, tuple)):
                pid = item[0]
                vals = item[1]
                if len(vals) < 2:
                    raise ValueError("Pose tuple must have at least x,y.")
                x, y = vals[0], vals[1]
                th = vals[2] if len(vals) >= 3 else 0.0
                d[int(pid)] = (float(x), float(y), float(th))
                continue

            # (x,y,theta)
            if isinstance(item, (list, tuple)) and len(item) == 3:
                x, y, th = item
                d[idx] = (float(x), float(y), float(th))
                continue

            # (x,y) → th=0
            if isinstance(item, (list, tuple)) and len(item) == 2 and all(
                isinstance(v, (int, float, np.integer, np.floating)) for v in item
            ):
                x, y = item
                d[idx] = (float(x), float(y), 0.0)
                continue

            # dict element
            if isinstance(item, dict):
                pid = int(item.get('id', idx))
                if 'x' in item and 'y' in item:
                    x = float(item['x']); y = float(item['y']); th = float(item.get('theta', item.get('th', 0.0)))
                    d[pid] = (x, y, th)
                    continue

            # object with attributes
            try:
                pid = int(getattr(item, 'id', idx))
                x = float(getattr(item, 'x'))
                y = float(getattr(item, 'y'))
                th = float(getattr(item, 'theta', getattr(item, 'th', 0.0)))
                d[pid] = (x, y, th)
                continue
            except Exception:
                raise ValueError(f"Unrecognized pose format for element #{idx}: {repr(item)}")

        return d

    gnd = to_dict(poses_gnd)
    gps = to_dict(poses_gps)

    common_ids = sorted(set(gnd.keys()) & set(gps.keys()))
    if not common_ids:
        raise ValueError("No common vertex IDs between GND and GPS/GT trajectories.")

    gnd_xy = np.array([[gnd[k][0], gnd[k][1]] for k in common_ids], dtype=float)
    gps_xy = np.array([[gps[k][0], gps[k][1]] for k in common_ids], dtype=float)

    # ---- Plot ----
    fig, ax = plt.subplots()
    ax.plot(gps_xy[:, 0], gps_xy[:, 1], label="GPS trajectory")
    ax.plot(gnd_xy[:, 0], gnd_xy[:, 1], label="GND trajectory")
    ax.scatter([gps_xy[0, 0]], [gps_xy[0, 1]], label="Start", zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"GND vs GPS — test_{i}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


if __name__ == "__main__":
    df_results = evaluate_batch_results(root_dir="test_results/exp1_test_results_2", save_path="test_results/exp1_test_results_2/ape_results.pkl")
    print(df_results)


    results_np = df_results.to_numpy()[:,1:]

    # 1. Compute mean APEs
    ape_diff = results_np[:,1] - results_np[:,0]

    ape_diff = ape_diff / results_np[:,1]

    # y-range limits
    ymin, ymax = -10.5, 2.0

    # If you don't have explicit GPS period values, use 1..N
    x = np.arange(1, len(ape_diff) + 1)

    fig, ax = plt.subplots()
    ax.plot(x, ape_diff, marker='o')
    ax.set_xlabel('gps period')
    ax.set_ylabel('ape difference')
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)

    ape_diff = np.asarray(ape_diff)
    above = ape_diff > ymax
    below = ape_diff < ymin

    # small padding so text sits inside the frame
    pad = 0.02 * (ymax - ymin)

    # annotate values that are above/below the y-limits
    for xi, yi in zip(x[above], ape_diff[above]):
        ax.text(xi, ymax - pad, f'{yi:.2f}', ha='center', va='top')

    for xi, yi in zip(x[below], ape_diff[below]):
        ax.text(xi, ymin + pad, f'{yi:.2f}', ha='center', va='bottom')

    plt.show()


    plot_gnd_vs_gps(root_dir="test_results/exp1_test_results_2", i = 22)