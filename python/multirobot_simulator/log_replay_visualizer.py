"""
Visualize ground truth and odometry-only estimate from Python simulator logs.

Reads gt_log_*.txt and msg_log_*.txt (same format written by simulator.py).
- Ground truth: dotted line (full path from gt_log).
- Odometry estimate: integrated from msg_log "odom" lines only; shown step-by-step
  as solid trail + point + heading, like the controller visualizer.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_gt_log(path: Path):
    """
    Parse gt_log from Python simulator: "t pose x y z yaw, vx, vy, vz, r".
    Returns: (times, poses) with poses (N, 4) = (x, y, z, yaw).
    """
    times = []
    poses = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 9 or parts[1] != "pose":
                continue
            t = float(parts[0])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            yaw = float(parts[5])
            times.append(t)
            poses.append([x, y, z, yaw])
    return np.array(times), np.array(poses)


def load_msg_log_odom(path: Path):
    """
    Parse msg_log from Python simulator; keep only "t odom v_fwd vz omega" lines.
    Returns: (times, odom) with odom (N, 3) = (v_fwd, vz, omega).
    """
    times = []
    odom = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5 or parts[1] != "odom":
                continue
            t = float(parts[0])
            v_fwd, vz, omega = float(parts[2]), float(parts[3]), float(parts[4])
            times.append(t)
            odom.append([v_fwd, vz, omega])
    return np.array(times), np.array(odom)


def integrate_odom(gt_pose0, odom_times, odom_readings):
    """
    Integrate odometry from initial pose (x, y, z, yaw).
    odom_readings: (N, 3) = (v_fwd, vz, omega). Each reading is applied over the
    interval to the next timestamp. Returns: (times, poses) with poses (N, 4) at each odom time.
    """
    x, y, z, yaw = float(gt_pose0[0]), float(gt_pose0[1]), float(gt_pose0[2]), float(gt_pose0[3])
    times_out = [odom_times[0]]
    poses_out = [np.array([x, y, z, yaw])]

    for i in range(1, len(odom_times)):
        dt = odom_times[i] - odom_times[i - 1]
        v_fwd, vz, omega = odom_readings[i - 1, 0], odom_readings[i - 1, 1], odom_readings[i - 1, 2]
        yaw += dt * omega
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        vx = v_fwd * np.cos(yaw)
        vy = v_fwd * np.sin(yaw)
        x += dt * vx
        y += dt * vy
        z += dt * vz
        times_out.append(odom_times[i])
        poses_out.append(np.array([x, y, z, yaw]))

    return np.array(times_out), np.array(poses_out)


def set_equal_3d(ax, X, Y, Z, pad=1.0):
    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)
    zmin, zmax = np.min(Z), np.max(Z)
    r = max(xmax - xmin, ymax - ymin, zmax - zmin) * 0.5 + pad
    xc = (xmax + xmin) * 0.5
    yc = (ymax + ymin) * 0.5
    zc = (zmax + zmin) * 0.5
    ax.set_xlim(xc - r, xc + r)
    ax.set_ylim(yc - r, yc + r)
    ax.set_zlim(zc - r, zc + r)


def discover_bot_ids(log_dir: Path):
    """Find bot IDs from gt_log_*.txt in log_dir."""
    ids = []
    for p in sorted(log_dir.glob("gt_log_*.txt")):
        m = re.match(r"gt_log_(\d+)\.txt", p.name)
        if m:
            ids.append(m.group(1))
    return sorted(ids, key=int)


def run_log_replay_visualizer(
    log_dir: str,
    bot_ids: list = None,
    dt_anim: float = 0.05,
    animation_speed: float = 0.5,
    heading_len: float = 2.0,
):
    log_path = Path(log_dir)
    if not log_path.is_dir():
        raise FileNotFoundError(f"Log directory not found: {log_path}")

    if bot_ids is None:
        bot_ids = discover_bot_ids(log_path)
    if not bot_ids:
        raise FileNotFoundError(f"No gt_log_*.txt found in {log_path}")

    gt_times = {}
    gt_poses = {}
    odom_trajectory_times = {}
    odom_trajectory_poses = {}
    rel_events = {}  # observer_id -> list of (t, target_id, rx, ry, rz)

    for bid in bot_ids:
        gt_file = log_path / f"gt_log_{bid}.txt"
        msg_file = log_path / f"msg_log_{bid}.txt"
        if not gt_file.exists() or not msg_file.exists():
            continue

        # Ground truth
        t_gt, p_gt = load_gt_log(gt_file)
        if len(t_gt) == 0:
            continue
        gt_times[bid] = t_gt
        gt_poses[bid] = p_gt

        # Odometry readings
        t_odom, o = load_msg_log_odom(msg_file)
        if len(t_odom) == 0:
            continue
        gt0 = p_gt[0]
        t_traj, p_traj = integrate_odom(gt0, t_odom, o)
        odom_trajectory_times[bid] = t_traj
        odom_trajectory_poses[bid] = p_traj

        # Relative-pose events for this observer from msg_log
        ev = []
        with msg_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7 or parts[1] != "relpos":
                    continue
                try:
                    t_rel = float(parts[0])
                    target_id = parts[2]
                    rx, ry, rz = float(parts[3]), float(parts[4]), float(parts[5])
                except ValueError:
                    continue
                ev.append((t_rel, target_id, rx, ry, rz))
        rel_events[bid] = ev

    if not odom_trajectory_poses:
        raise RuntimeError("No valid bot data found.")

    bots = list(odom_trajectory_poses.keys())
    D = len(bots)
    N = max(len(odom_trajectory_poses[b]) for b in bots)

    # Pre-assign relpos events to nearest odom frame index per observer
    frame_rel = {bid: {} for bid in bots}  # bid -> frame_idx -> (target_id, rx, ry, rz)
    for bid in bots:
        t_traj = odom_trajectory_times[bid]
        if bid not in rel_events:
            continue
        for (t_rel, target_id, rx, ry, rz) in rel_events[bid]:
            # find nearest index in odom trajectory
            idx = int(np.searchsorted(t_traj, t_rel, side="left"))
            if idx <= 0:
                frame_idx = 0
            elif idx >= len(t_traj):
                frame_idx = len(t_traj) - 1
            else:
                # choose closer of idx-1 or idx
                if abs(t_traj[idx] - t_rel) < abs(t_traj[idx - 1] - t_rel):
                    frame_idx = idx
                else:
                    frame_idx = idx - 1
            frame_rel[bid][frame_idx] = (target_id, rx, ry, rz)

    all_x = np.concatenate([gt_poses[b][:, 0] for b in bots] + [odom_trajectory_poses[b][:, 0] for b in bots])
    all_y = np.concatenate([gt_poses[b][:, 1] for b in bots] + [odom_trajectory_poses[b][:, 1] for b in bots])
    all_z = np.concatenate([gt_poses[b][:, 2] for b in bots] + [odom_trajectory_poses[b][:, 2] for b in bots])

    fig = plt.figure(figsize=(9, 7))
    ax3d = fig.add_subplot(1, 1, 1, projection="3d")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("Log replay: ground truth (dotted) vs odometry estimate (solid)")

    colors = plt.cm.tab10(np.linspace(0, 1, max(D, 10)))

    gt_lines = []
    odom_trail_lines = []
    odom_pts = []
    head_lines = []
    rel_lines_start = []  # first half: observer color
    rel_lines_end = []    # second half: target color

    for d, bid in enumerate(bots):
        c = colors[d % len(colors)]
        p_gt = gt_poses[bid]
        lgt, = ax3d.plot(p_gt[:, 0], p_gt[:, 1], p_gt[:, 2], ":", color=c, lw=2.0, alpha=0.8, label=f"GT {bid}")
        gt_lines.append(lgt)
        trail, = ax3d.plot([], [], [], "-", color=c, lw=2.0, label=f"odom {bid}")
        pt, = ax3d.plot([], [], [], "o", color=c, markersize=8)
        head, = ax3d.plot([], [], [], "-", color=c, lw=1.5)
        odom_trail_lines.append(trail)
        odom_pts.append(pt)
        head_lines.append(head)

        # Relative pose visualization: two segments so color can change along the line
        rl_start, = ax3d.plot([], [], [], "-", color=c, alpha=0.3, lw=1.5)
        rl_end, = ax3d.plot([], [], [], "-", color=c, alpha=0.3, lw=1.5)
        rel_lines_start.append(rl_start)
        rel_lines_end.append(rl_end)

    set_equal_3d(ax3d, all_x, all_y, all_z, pad=2.0)
    ax3d.legend(loc="upper left", ncol=2)

    def set_point3d(line, x, y, z):
        line.set_data([x], [y])
        line.set_3d_properties([z])

    def set_line3d(line, xs, ys, zs):
        line.set_data(xs, ys)
        line.set_3d_properties(zs)

    idx = [0]

    def update(frame):
        for d, bid in enumerate(bots):
            p_odom = odom_trajectory_poses[bid]
            n = len(p_odom)
            k = min(idx[0], n - 1) if n else 0
            if n == 0:
                continue
            x, y, z, yaw = p_odom[k, 0], p_odom[k, 1], p_odom[k, 2], p_odom[k, 3]

            # odometry trail and heading
            set_line3d(odom_trail_lines[d], p_odom[: k + 1, 0], p_odom[: k + 1, 1], p_odom[: k + 1, 2])
            set_point3d(odom_pts[d], x, y, z)
            hx = x + heading_len * np.cos(yaw)
            hy = y + heading_len * np.sin(yaw)
            set_line3d(head_lines[d], [x, hx], [y, hy], [z, z])

            # relative pose visualization at this frame (if any)
            rl_start = rel_lines_start[d]
            rl_end = rel_lines_end[d]
            if k in frame_rel.get(bid, {}):
                target_id, rx, ry, rz = frame_rel[bid][k]
                # relpose is in the robot's body frame; rotate into world frame using yaw
                dx_w = rx * np.cos(yaw) - ry * np.sin(yaw)
                dy_w = rx * np.sin(yaw) + ry * np.cos(yaw)
                dz_w = rz
                x_end = x + dx_w
                y_end = y + dy_w
                z_end = z + dz_w
                xm = 0.5 * (x + x_end)
                ym = 0.5 * (y + y_end)
                zm = 0.5 * (z + z_end)

                # Start segment: observer color
                set_line3d(rl_start, [x, xm], [y, ym], [z, zm])

                # End segment: target bot's color if known, else reuse observer color
                if target_id in bots:
                    tgt_index = bots.index(target_id)
                    tgt_color = colors[tgt_index % len(colors)]
                    rl_end.set_color((*tgt_color[:3], 0.3))
                set_line3d(rl_end, [xm, x_end], [ym, y_end], [zm, z_end])
            else:
                # no relpose at this frame → hide lines
                set_line3d(rl_start, [], [], [])
                set_line3d(rl_end, [], [], [])

        idx[0] += 1
        return odom_trail_lines + odom_pts + head_lines + rel_lines_start + rel_lines_end

    interval_ms = 1000 * dt_anim * (1.0 / animation_speed)
    ani = FuncAnimation(fig, update, frames=N, interval=interval_ms, blit=False)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize GT and odometry-only from Python simulator logs")
    parser.add_argument("log_dir", nargs="?", default="python/multirobot_simulator/sample_data", help="Directory with gt_log_*.txt and msg_log_*.txt")
    parser.add_argument("--bots", nargs="*", help="Bot IDs (default: auto from gt_log_*.txt)")
    parser.add_argument("--dt", type=float, default=0.05, help="Animation frame interval (s)")
    parser.add_argument("--speed", type=float, default=0.5, help="Animation speed multiplier")
    parser.add_argument("--heading-len", type=float, default=2.0, help="Heading arrow length")
    args = parser.parse_args()

    args.log_dir = "test_data/multidrone"

    run_log_replay_visualizer(
        log_dir=args.log_dir,
        bot_ids=args.bots,
        dt_anim=args.dt,
        animation_speed=args.speed,
        heading_len=args.heading_len,
    )


if __name__ == "__main__":
    main()
