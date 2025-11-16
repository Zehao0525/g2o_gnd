from controller import *
from simulator import *

import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - activates 3D projection

import time


# ---------- Utility ----------

def quat_from_yaw(psi):
    """Return quaternion [w, x, y, z] for yaw about +y (matches x–z heading)."""
    c = np.cos(psi / 2.0)
    s = np.sin(psi / 2.0)
    return np.array([c, 0.0, s, 0.0], float)


def set_equal_3d(ax, X, Y, Z, pad=1.0):
    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)
    zmin, zmax = np.min(Z), np.max(Z)
    xr = xmax - xmin
    yr = ymax - ymin
    zr = zmax - zmin
    r = max(xr, yr, zr) * 0.5 + pad
    xc = (xmax + xmin) * 0.5
    yc = (ymax + ymin) * 0.5
    zc = (zmax + zmin) * 0.5
    ax.set_xlim(xc - r, xc + r)
    ax.set_ylim(yc - r, yc + r)
    ax.set_zlim(zc - r, zc + r)


# ---------- N-drone 3D visualizer ----------

def run_visualizer_3d_multi(ctrls, word_sim:WorldSim, T=30.0, dt=0.02, heading_len=2.0, annimation_speed = 1.0, no_animation = False):
    """
    Visualize N yaw-only 3D velocity-controlled drones simultaneously.

    ctrls: list[VelocityXZWaypointController]
    sims:  list[DroneSim]
           Each DroneSim is assumed to:
             - contain its own controller
             - advance with sim.step()
             - expose state s = [x, y, z, yaw, vx, vy, vz, r]
    """
    assert len(ctrls) == len(word_sim.drone_sims), "ctrls and sims must have same length"
    D = len(word_sim.drone_sims)
    N = int(T / dt)

    # buffers: [time, drone, xyz/yaw]
    xs = np.zeros((N, D, 3))
    yaws = np.zeros((N, D))
    idx = 0

    # figure & 3D axis
    fig = plt.figure(figsize=(9, 7))
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("Multi-drone 3D velocity-mode (yaw-only)")

    # --- static paths & global bounds from waypoints ---
    all_wp_x = []
    all_wp_y = []
    all_wp_z = []

    for d, ctrl in enumerate(ctrls):
        if not hasattr(ctrl, "wp"):
            continue
        wps = np.asarray(ctrl.wp)
        if wps.ndim != 2 or wps.shape[1] != 3:
            continue
        ax3d.plot(wps[:, 0], wps[:, 1], wps[:, 2], '--', alpha=0.5)
        all_wp_x.append(wps[:, 0])
        all_wp_y.append(wps[:, 1])
        all_wp_z.append(wps[:, 2])

    # If we have any waypoint info, use it to set bounds; otherwise use initial states.
    if all_wp_x:
        X = np.concatenate(all_wp_x)
        Y = np.concatenate(all_wp_y)
        Z = np.concatenate(all_wp_z)
    else:
        init_states = np.array([sim.s[:3] for sim in sims])
        X, Y, Z = init_states[:, 0], init_states[:, 1], init_states[:, 2]

    set_equal_3d(ax3d, X, Y, Z, pad=2.0)

    # --- artists for each drone ---
    trail_lines = []
    drone_pts = []
    head_lines = []

    # init from current sim states
    for d, sim in enumerate(word_sim.drone_sims):
        x, y, z, yaw, vx, vy, vz, r = sim.s

        # trajectory trail
        trail_line, = ax3d.plot([], [], [], lw=2.0, label=f"drone {d+1}")
        # current position
        drone_pt, = ax3d.plot([x], [y], [z], 'o')
        # heading arrow
        head_line, = ax3d.plot([], [], [], lw=2.0)

        trail_lines.append(trail_line)
        drone_pts.append(drone_pt)
        head_lines.append(head_line)

    ax3d.legend(loc="upper left")

    # --- helpers ---

    def set_point3d(line, x, y, z):
        line.set_data([x], [y])
        line.set_3d_properties([z])

    def set_line3d(line, X, Y, Z):
        line.set_data(X, Y)
        line.set_3d_properties(Z)

    # --- animation update ---

    def update(frame):
        nonlocal idx
        if idx >= N:
            return trail_lines + drone_pts + head_lines

        word_sim.step()
        
        for d, sim in enumerate(word_sim.drone_sims):
            # advance that drone
            #sim.step()
            s = sim.s
            x, y, z, yaw, vx, vy, vz, r = s

            # log
            xs[idx, d, :] = [x, y, z]
            yaws[idx, d] = yaw

            # update trail
            set_line3d(
                trail_lines[d],
                xs[:idx+1, d, 0],
                xs[:idx+1, d, 1],
                xs[:idx+1, d, 2],
            )

            # update position marker
            set_point3d(drone_pts[d], x, y, z)

            # heading arrow (x–z heading, constant y; same convention as your single-drone version)
            hx = x + heading_len * np.cos(yaw)
            hy = y + heading_len * np.sin(yaw)
            set_line3d(head_lines[d], [x, hx], [y, hy], [z, z])

        idx += 1
        return trail_lines + drone_pts + head_lines

    
    ani = FuncAnimation(fig, update, frames=N, interval=1000 * dt * (1/annimation_speed), blit=False)
    plt.show()


# ---------- Example wiring ----------

if __name__ == "__main__":
    DT = 0.02
    N_DRONES = 5  # set as needed

    ctrls = []
    sims = []

    word_sim = WorldSim([])

    for drone_id in range(0, N_DRONES):
        try:
            # Assumes your JSON uses string keys "1", "2", ..., or similar per drone
            ctrl = VelocityXZWaypointController.from_json(
                'config/sim_config.json',
                str(drone_id),
                DT
            )
        except NameError:
            raise RuntimeError(
                "Paste/import VelocityXZWaypointController and LimitsVel above."
            )

        sim = DroneSim(
            str(drone_id),
            ctrl,
            world_sim=word_sim,
            msg_log_path=f'msg_log_{drone_id}.txt',
            gt_log_path=f'gt_log_{drone_id}.txt',
            config_path='config/sim_config.json',
            dt=DT,
            yaw0=0.0,
        )
        ctrls.append(ctrl)
        sims.append(sim)

    word_sim.set_drone_sims(sims)

    word_sim, _, _ = WorldSim.create(trajectory_path='config/trajectories.json')

    for k in range(10000):
        word_sim.step()
        if word_sim.reached_dest_all():
            break
    run_visualizer_3d_multi(ctrls, word_sim, T=35.0, dt=DT, heading_len=2.5, annimation_speed = 0.1, no_animation = False)
