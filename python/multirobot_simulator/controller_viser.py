from controller import *
from simulator import *

import matplotlib
matplotlib.use("TkAgg")

"""
Visualize a yaw-only 3D velocity-controlled drone:
- Single 3D view: path, pose (triangle), and heading.
- Controller outputs *velocities*: v_fwd (body-x), vy (world-y), omega_yaw.
Coordinates:
- Yaw about +y (turn in x–z plane, consistent with the heading arrow drawn in x–z).
- Forward motion projects into world x–z via yaw. Vertical is world-y.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - activates 3D projection

# ---------- Minimal kinematic sim (velocity interface) ----------


# ---------- Utility ----------
def quat_from_yaw(psi):
    """Return quaternion [w, x, y, z] for yaw about +y (matches x–z heading)."""
    c = np.cos(psi/2.0)
    s = np.sin(psi/2.0)
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

# ---------- The 3D visualizer ----------
def run_visualizer_3d(ctrl, sim, T=30.0, dt=0.02, heading_len=2.0):
    """
    ctrl: your VelocityXZWaypointController instance
    sim:  DroneSim instance
    """
    N = int(T/dt)

    # buffers
    xs = np.zeros((N, 3))
    yaws = np.zeros(N)
    idx = 0

    # figure & 3D axis
    fig = plt.figure(figsize=(8, 6))
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("3D velocity-mode drone (yaw-only)")

    # static path
    wps = ctrl.wp
    path_line, = ax3d.plot(wps[:,0], wps[:,1], wps[:,2], '--')

    # artists to update
    trail_line, = ax3d.plot([], [], [], lw=2.0)                      # trail
    drone_pt,   = ax3d.plot([sim.s[0]], [sim.s[1]], [sim.s[2]], 'o')  # current position
    head_line,  = ax3d.plot([], [], [], lw=2.0)                       # heading arrow (x–z at current y)

    # axis limits
    set_equal_3d(ax3d, wps[:,0], wps[:,1], wps[:,2], pad=2.0)

    # init state
    x, y, z, yaw, vx, vy, vz, r = sim.s

    def set_point3d(line, x, y, z):
        line.set_data([x], [y])
        line.set_3d_properties([z])

    def set_line3d(line, X, Y, Z):
        line.set_data(X, Y)
        line.set_3d_properties(Z)

    def update(frame):
        nonlocal x, y, z, yaw, vx, vy, vz, r, idx

        # controller inputs
        pos  = np.array([x, y, z])
        vel  = np.array([vx, vy, vz])
        quat = quat_from_yaw(yaw)

        # --- velocity controller interface ---

        # integrate sim (kinematic)
        sim.step()
        s = sim.s
        x, y, z, yaw, vx, vy, vz, r = s

        # log
        xs[idx] = [x, y, z]
        yaws[idx] = yaw
        idx += 1

        # update artists
        set_line3d(trail_line, xs[:idx,0], xs[:idx,1], xs[:idx,2])
        set_point3d(drone_pt, x, y, z)

        # heading arrow (x–z plane, constant y)
        hx = x + heading_len * np.cos(yaw)
        hy = y + heading_len * np.sin(yaw)
        set_line3d(head_line, [x, hx], [y, hy], [z, z])

        return trail_line, drone_pt, head_line

    ani = FuncAnimation(fig, update, frames=N, interval=1000*dt, blit=False)
    plt.show()

# ---------- Example wiring ----------
if __name__ == "__main__":
    # Example waypoints


    DT = 0.02
    try:

        ctrl = VelocityXZWaypointController.from_json('config/sim_config.json', '1', DT)
    except NameError:
        raise RuntimeError("Paste/import VelocityXZWaypointController and LimitsVel above.")

    # Initial state: at first waypoint, slight yaw
    yaw0 = np.deg2rad(10.0)
    sim = DroneSim('1', ctrl, world_sim = None, 
                 msg_log_path = 'msg_log.txt',
                 gt_log_path = 'gt_log.txt',
                 config_path = 'config/sim_config.json',
                 dt = DT,
                 yaw0 = 0)

    run_visualizer_3d(ctrl, sim, T=35.0, dt=DT, heading_len=2.5)
