"""
Visualize ground truth and odometry-only estimate from Python simulator logs.

Reads gt_log_*.txt and msg_log_*.txt (same format written by simulator.py).
- Ground truth: dotted line (full path from gt_log).
- Odometry estimate: integrated from msg_log "odom" lines only; shown step-by-step
  as solid trail + point + heading.
- relpos: two-tone line (observer → midpoint → target color).
- lmobs_rp: same geometry as relpos (body-frame rel rotated to world), but one segment
  only, entirely in the observer drone's color.

Edit the constants in ``main()`` / module defaults below (no argparse).

- ``DEFAULT_ROBOT_OBS_ON``: draw relpos (robot-to-robot relative) observation rays.
- ``DEFAULT_LM_OBS_ON``: draw lmobs_rp landmark observation rays.
- ``DEFAULT_ROBOT_OBS_LAST_ONLY`` / ``DEFAULT_LM_OBS_LAST_ONLY``: if True, only the last
  robot (highest index in ``bots``) draws that observation type.
- If ``landmarks.json`` exists in the log directory, landmark positions are drawn (static).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------------------------------------------------------------
# Defaults — edit these paths / options (replaces CLI parsing)
# ---------------------------------------------------------------------------
DEFAULT_LOG_DIR = "test_data/multidrone2/unit_test/1_samples/0"
DEFAULT_BOT_IDS = None  # None = discover from gt_log_*.txt in LOG_DIR
DEFAULT_DT_ANIM = 0.05
DEFAULT_ANIMATION_SPEED = 0.5
DEFAULT_HEADING_LEN = 2.0
DEFAULT_ROBOT_OBS_ON = False
DEFAULT_LM_OBS_ON = True
DEFAULT_ROBOT_OBS_LAST_ONLY = True
DEFAULT_LM_OBS_LAST_ONLY = True


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
    Parse msg_log from Python simulator; keep only "t odom ..." lines.
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


def load_msg_log_relpos_events(path: Path):
    """Parse relpos lines: t relpos target_id rx ry rz qx qy qz qw ..."""
    ev = []
    with path.open("r", encoding="utf-8") as f:
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
    return ev


def load_msg_log_lmobs_events(path: Path):
    """Parse lmobs_rp lines: t lmobs_rp landmark_id rx ry rz std_x std_y std_z"""
    ev = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7 or parts[1] != "lmobs_rp":
                continue
            try:
                t_lm = float(parts[0])
                lm_id = parts[2]
                rx, ry, rz = float(parts[3]), float(parts[4]), float(parts[5])
            except ValueError:
                continue
            ev.append((t_lm, lm_id, rx, ry, rz))
    return ev


def load_landmarks_xyz(log_dir: Path):
    """
    Load ``log_dir/landmarks.json`` if present and valid.
    Expected shape: { "id": [x, y, z], ... }. Returns (N, 3) float array or None.
    """
    path = log_dir / "landmarks.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or not data:
        return None
    xs, ys, zs = [], [], []
    for _k, v in data.items():
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            try:
                xs.append(float(v[0]))
                ys.append(float(v[1]))
                zs.append(float(v[2]))
            except (TypeError, ValueError):
                continue
    if not xs:
        return None
    return np.column_stack([np.array(xs), np.array(ys), np.array(zs)])


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
    """Find bot IDs from gt_log_*.txt in log_dir (numeric suffix)."""
    ids = []
    for p in sorted(log_dir.glob("gt_log_*.txt")):
        stem = p.stem  # gt_log_<id>
        if not stem.startswith("gt_log_"):
            continue
        suffix = stem[len("gt_log_") :]
        if suffix.isdigit():
            ids.append(suffix)
    return sorted(ids, key=int)


def _nearest_frame_index(t_traj: np.ndarray, t_event: float) -> int:
    idx = int(np.searchsorted(t_traj, t_event, side="left"))
    if idx <= 0:
        return 0
    if idx >= len(t_traj):
        return len(t_traj) - 1
    if abs(t_traj[idx] - t_event) < abs(t_traj[idx - 1] - t_event):
        return idx
    return idx - 1


def _body_rel_to_world(rx: float, ry: float, rz: float, yaw: float):
    """Relative position in body frame → world frame displacement (same as relpos block)."""
    dx_w = rx * np.cos(yaw) - ry * np.sin(yaw)
    dy_w = rx * np.sin(yaw) + ry * np.cos(yaw)
    dz_w = rz
    return dx_w, dy_w, dz_w


def run_log_replay_visualizer(
    log_dir: str,
    bot_ids: list = None,
    dt_anim: float = 0.05,
    animation_speed: float = 0.5,
    heading_len: float = 2.0,
    robot_obs_on: bool = True,
    lm_obs_on: bool = True,
    robot_obs_last_only: bool = False,
    lm_obs_last_only: bool = False,
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
    lm_events = {}  # observer_id -> list of (t, lm_id, rx, ry, rz)

    for bid in bot_ids:
        gt_file = log_path / f"gt_log_{bid}.txt"
        msg_file = log_path / f"msg_log_{bid}.txt"
        if not gt_file.exists() or not msg_file.exists():
            continue

        t_gt, p_gt = load_gt_log(gt_file)
        if len(t_gt) == 0:
            continue
        gt_times[bid] = t_gt
        gt_poses[bid] = p_gt

        t_odom, o = load_msg_log_odom(msg_file)
        if len(t_odom) == 0:
            continue
        gt0 = p_gt[0]
        t_traj, p_traj = integrate_odom(gt0, t_odom, o)
        odom_trajectory_times[bid] = t_traj
        odom_trajectory_poses[bid] = p_traj

        rel_events[bid] = load_msg_log_relpos_events(msg_file) if robot_obs_on else []
        lm_events[bid] = load_msg_log_lmobs_events(msg_file) if lm_obs_on else []

    if not odom_trajectory_poses:
        raise RuntimeError("No valid bot data found.")

    bots = list(odom_trajectory_poses.keys())
    D = len(bots)
    N = max(len(odom_trajectory_poses[b]) for b in bots)

    frame_rel = {bid: {} for bid in bots}
    if robot_obs_on:
        for bid in bots:
            t_traj = odom_trajectory_times[bid]
            for (t_rel, target_id, rx, ry, rz) in rel_events.get(bid, []):
                frame_idx = _nearest_frame_index(t_traj, t_rel)
                frame_rel[bid][frame_idx] = (target_id, rx, ry, rz)

    frame_lm = {bid: {} for bid in bots}
    if lm_obs_on:
        for bid in bots:
            t_traj = odom_trajectory_times[bid]
            for (t_lm, lm_id, rx, ry, rz) in lm_events.get(bid, []):
                frame_idx = _nearest_frame_index(t_traj, t_lm)
                frame_lm[bid].setdefault(frame_idx, []).append((lm_id, rx, ry, rz))

    landmarks_xyz = load_landmarks_xyz(log_path)

    all_x_parts = [gt_poses[b][:, 0] for b in bots] + [odom_trajectory_poses[b][:, 0] for b in bots]
    all_y_parts = [gt_poses[b][:, 1] for b in bots] + [odom_trajectory_poses[b][:, 1] for b in bots]
    all_z_parts = [gt_poses[b][:, 2] for b in bots] + [odom_trajectory_poses[b][:, 2] for b in bots]
    if landmarks_xyz is not None:
        all_x_parts.append(landmarks_xyz[:, 0])
        all_y_parts.append(landmarks_xyz[:, 1])
        all_z_parts.append(landmarks_xyz[:, 2])
    all_x = np.concatenate(all_x_parts)
    all_y = np.concatenate(all_y_parts)
    all_z = np.concatenate(all_z_parts)

    fig = plt.figure(figsize=(9, 7))
    ax3d = fig.add_subplot(1, 1, 1, projection="3d")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    title_parts = ["Log replay: GT (dotted) vs odom (solid)"]
    if robot_obs_on:
        title_parts.append("relpos (2-tone)")
    if lm_obs_on:
        title_parts.append("lmobs (observer color)")
    if landmarks_xyz is not None:
        title_parts.append("landmarks")
    ax3d.set_title("; ".join(title_parts))

    colors = plt.cm.tab10(np.linspace(0, 1, max(D, 10)))

    if landmarks_xyz is not None:
        ax3d.scatter(
            landmarks_xyz[:, 0],
            landmarks_xyz[:, 1],
            landmarks_xyz[:, 2],
            c="0.35",
            marker="^",
            s=36,
            alpha=0.85,
            label="landmarks",
            depthshade=True,
        )

    gt_lines = []
    odom_trail_lines = []
    odom_pts = []
    head_lines = []
    rel_lines_start = []
    rel_lines_end = []
    lm_lines = []  # single segment, observer color only

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

        rl_start, = ax3d.plot([], [], [], "-", color=c, alpha=0.3, lw=1.5)
        rl_end, = ax3d.plot([], [], [], "-", color=c, alpha=0.3, lw=1.5)
        rel_lines_start.append(rl_start)
        rel_lines_end.append(rl_end)

        lm_ln, = ax3d.plot([], [], [], "-", color=c, alpha=0.35, lw=1.5)
        lm_lines.append(lm_ln)

    set_equal_3d(ax3d, all_x, all_y, all_z, pad=2.0)
    ax3d.legend(loc="upper left", ncol=2)

    def set_line3d(line, xs, ys, zs):
        line.set_data(xs, ys)
        line.set_3d_properties(zs)

    idx = [0]
    last_obs_bid = bots[-1]

    def update(frame):
        for d, bid in enumerate(bots):
            p_odom = odom_trajectory_poses[bid]
            n = len(p_odom)
            k = min(idx[0], n - 1) if n else 0
            if n == 0:
                continue
            x, y, z, yaw = p_odom[k, 0], p_odom[k, 1], p_odom[k, 2], p_odom[k, 3]

            set_line3d(odom_trail_lines[d], p_odom[: k + 1, 0], p_odom[: k + 1, 1], p_odom[: k + 1, 2])
            odom_pts[d].set_data([x], [y])
            odom_pts[d].set_3d_properties([z])
            hx = x + heading_len * np.cos(yaw)
            hy = y + heading_len * np.sin(yaw)
            set_line3d(head_lines[d], [x, hx], [y, hy], [z, z])

            rl_start = rel_lines_start[d]
            rl_end = rel_lines_end[d]
            if robot_obs_on and not (robot_obs_last_only and bid != last_obs_bid):
                if k in frame_rel.get(bid, {}):
                    target_id, rx, ry, rz = frame_rel[bid][k]
                    dx_w, dy_w, dz_w = _body_rel_to_world(rx, ry, rz, yaw)
                    x_end = x + dx_w
                    y_end = y + dy_w
                    z_end = z + dz_w
                    xm = 0.5 * (x + x_end)
                    ym = 0.5 * (y + y_end)
                    zm = 0.5 * (z + z_end)

                    set_line3d(rl_start, [x, xm], [y, ym], [z, zm])

                    if target_id in bots:
                        tgt_index = bots.index(target_id)
                        tgt_color = colors[tgt_index % len(colors)]
                        rl_end.set_color((*tgt_color[:3], 0.3))
                    set_line3d(rl_end, [xm, x_end], [ym, y_end], [zm, z_end])
                else:
                    set_line3d(rl_start, [], [], [])
                    set_line3d(rl_end, [], [], [])
            else:
                set_line3d(rl_start, [], [], [])
                set_line3d(rl_end, [], [], [])

            lm_ln = lm_lines[d]
            if lm_obs_on and not (lm_obs_last_only and bid != last_obs_bid):
                lm_ln.set_color((*colors[d % len(colors)][:3], 0.35))
                if k in frame_lm.get(bid, {}) and frame_lm[bid][k]:
                    xs, ys, zs = [], [], []
                    first_seg = True
                    for (_lm_id, rx, ry, rz) in frame_lm[bid][k]:
                        dx_w, dy_w, dz_w = _body_rel_to_world(rx, ry, rz, yaw)
                        if not first_seg:
                            xs.append(np.nan)
                            ys.append(np.nan)
                            zs.append(np.nan)
                        xs.extend([x, x + dx_w])
                        ys.extend([y, y + dy_w])
                        zs.extend([z, z + dz_w])
                        first_seg = False
                    set_line3d(lm_ln, xs, ys, zs)
                else:
                    set_line3d(lm_ln, [], [], [])
            else:
                set_line3d(lm_ln, [], [], [])

        idx[0] += 1
        return (
            odom_trail_lines
            + odom_pts
            + head_lines
            + rel_lines_start
            + rel_lines_end
            + lm_lines
        )

    interval_ms = 1000 * dt_anim * (1.0 / animation_speed)
    # Must keep a reference: otherwise the animation is GC'd before plt.show() runs.
    _anim = FuncAnimation(fig, update, frames=N, interval=interval_ms, blit=False)
    plt.show()


def main():
    run_log_replay_visualizer(
        log_dir=DEFAULT_LOG_DIR,
        bot_ids=DEFAULT_BOT_IDS,
        dt_anim=DEFAULT_DT_ANIM,
        animation_speed=DEFAULT_ANIMATION_SPEED,
        heading_len=DEFAULT_HEADING_LEN,
        robot_obs_on=DEFAULT_ROBOT_OBS_ON,
        lm_obs_on=DEFAULT_LM_OBS_ON,
        robot_obs_last_only=DEFAULT_ROBOT_OBS_LAST_ONLY,
        lm_obs_last_only=DEFAULT_LM_OBS_LAST_ONLY,
    )


if __name__ == "__main__":
    main()
