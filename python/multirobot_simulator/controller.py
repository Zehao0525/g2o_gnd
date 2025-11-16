from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

def quat_to_yaw_wxyz(q: np.ndarray) -> float:
    """Quaternion -> yaw (rad). q=[w,x,y,z], yaw about +z (turn in x–y plane)."""
    x, y, z, w = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi

@dataclass
class LimitsVel:
    max_lin_speed: float   # [m/s] cap for |v_fwd| and |vy|
    max_ang_speed: float   # [rad/s] cap for |omega_yaw|

class CState(Enum):
    TURNING = 0
    FLYING  = 1

class VelocityXZWaypointController:
    """
    Velocity controller with two states (y-up, x-forward, z-right):
      TURNING: output only omega_yaw (yaw about +y) to face waypoint in x–z; v_fwd=vy=0.
      FLYING:  output only v_fwd (body-x) and vy (world-y) toward waypoint; omega_yaw=0.

    - Rotate in x–z (heading angle around +y).
    - Translate only in x (forward) and y (vertical-up). No commanded lateral z motion.
    - Zero commanded velocity during state transitions.
    - Waypoint considered reached by x–y proximity; then advance and return to TURNING.
    - Fixed simulation step dt is provided at init (used for one-step velocity targets).
    """

    def __init__(
        self,
        waypoints_xyz: np.ndarray,
        limits: LimitsVel,
        dt: float,
        reach_tol_xy: float = 0.30,   # [m] proximity in x–y to accept waypoint
        yaw_tol: float = 1e-2,        # [rad] heading alignment for transition
    ):
        assert waypoints_xyz.shape[1] == 3
        self.wp = waypoints_xyz.astype(float)
        self.lim = limits
        self.dt = float(dt)

        self._i = 0
        self.state = CState.TURNING
        self.step_number = 0

        self._reach_tol_xy = float(reach_tol_xy)
        self._yaw_tol = float(yaw_tol)

        self.terminal = False


    # --- Factory method ---
    @staticmethod
    def from_json(json_path: Union[str, Path], bot_id: str, dt:float, trajectory_path:Union[str, Path] = None) -> "VelocityXZWaypointController":
        """
        Create a controller for the specified bot from a JSON configuration.

        Expected structure (partial):
        {
          "dt": 0.5,
          "bots": {
            "0": {
              "path": [[0, 0, 0], [29, 29, 29]],
              "controller": {
                "max_lin_vel": 3,
                "max_lin_acc": 2,
                "max_rot_vel": 90,
                "max_rot_acc": 45
              }
            }
          }
        }
        """
        path = Path(json_path)
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        if "bots" not in cfg or bot_id not in cfg["bots"]:
            raise KeyError(f"Bot ID '{bot_id}' not found in {json_path}")

        bot_cfg = cfg["bots"][bot_id]
        if trajectory_path is None:
            wp = np.asarray(bot_cfg["path"], dtype=float)
        else:
            traj_path = Path(trajectory_path)
            with traj_path.open("r", encoding="utf-8") as f:
                traj_cfg = json.load(f)
            wp = np.asarray(traj_cfg[bot_id], dtype=float)
        dt = float(cfg["dt"])

        # Extract limits (convert deg/s to rad/s for yaw if needed)
        ctrl_cfg = bot_cfg["controller"]
        # lim_dict = dict(
        #     v_fwd_max=ctrl_cfg["max_lin_vel"],
        #     vy_max=ctrl_cfg["max_lin_vel"],
        #     omega_yaw_max=np.deg2rad(ctrl_cfg["max_rot_vel"]),
        #     a_fwd_max=ctrl_cfg["max_lin_acc"],
        #     ay_max=ctrl_cfg["max_lin_acc"],
        #     alpha_yaw_max=np.deg2rad(ctrl_cfg["max_rot_acc"]),
        # )
        lim = LimitsVel(max_lin_speed=ctrl_cfg["max_lin_vel"], max_ang_speed=np.deg2rad(ctrl_cfg["max_rot_vel"]))
        #lim = LimitsVel(**lim_dict)

        return VelocityXZWaypointController(
            waypoints_xyz=wp,
            limits=lim,
            dt=dt
        )

    # -------- helpers --------
    def _body_axes(self, yaw: float):
        """
        Body axes in world frame for yaw about +y:
          body-x (forward) in x–z: [cos(yaw), 0, sin(yaw)]
          body-y (up)     = world-y: [0, 1, 0]
        """
        x_hat = np.array([np.cos(yaw), np.sin(yaw), 0.0])  # forward in x–z
        z_hat = np.array([0.0, 0.0, 1.0])                  # world/body y (up)
        return x_hat, z_hat

    # -------- main step --------
    def step(
        self,
        pos_xyz: np.ndarray,
        quat_xyzw: np.ndarray,
        vel_xyz: np.ndarray,   # not used for control (kinematic), kept for API
        yaw_rate: float,       # not used; we command omega directly
        dt_unused: float = None,      # external dt; controller uses self.dt
    ) -> dict:
        """
        Returns:
          v_fwd: commanded forward velocity along body-x [m/s]
          vy:    commanded vertical velocity along world-y [m/s]
          omega_yaw: commanded yaw rate [rad/s] about +y
          debug: diagnostics
        """
        self.step_number += 1

        yaw = quat_to_yaw_wxyz(quat_xyzw)
        x_hat, z_hat = self._body_axes(yaw)

        # Current target waypoint
        wp = self.wp[self._i]
        r = wp - pos_xyz

        # Desired yaw faces waypoint projection into x–z (since yaw turns in that plane)
        if abs(r[0]) + abs(r[1]) > 1e-12:
            yaw_des = np.arctan2(r[1], r[0])  # atan2(dz, dx)
        else:
            yaw_des = yaw
        yaw_err = wrap_pi(yaw_des - yaw)

        # Defaults (safe zeros)
        v_fwd_cmd = 0.0
        vz_cmd    = 0.0
        omega_cmd = 0.0
        transitioned = False

        if self.terminal:
            debug = {
                "step": self.step_number,
                "state": self.state.name,
                "i_wp": self._i,
                "yaw": float(yaw),
                "yaw_des": float(yaw),
                "yaw_err": float(0),
                "transitioned_this_step": transitioned,
            }
            return {
                "v_fwd": v_fwd_cmd,            # body-x velocity command
                "vz": vz_cmd,                  # world-z velocity command (z is up)
                "omega_yaw": omega_cmd,        # yaw rate about +z
                "debug": debug,
            }

        # ---------------- TURNING (angular-only) ----------------
        if self.state == CState.TURNING:
            # If already aligned within tolerance, transition *with zero outputs*
            if abs(yaw_err) < self._yaw_tol:
                self.state = CState.FLYING
                transitioned = True
                # outputs remain zero this step by design
            else:
                # Command yaw rate to close the gap in one step (clipped)
                omega_cmd = float(np.clip(yaw_err / self.dt,
                                          -self.lim.max_ang_speed,
                                          self.lim.max_ang_speed))
                v_fwd_cmd = 0.0
                vz_cmd    = 0.0

        # ---------------- FLYING (linear-only) ----------------
        else:
            # Components along allowed axes
            ex = float(np.dot(r, x_hat))  # along body-x (forward)
            ez = float(np.dot(r, z_hat))  # world-y (up) = r[1]
            dist_xz = float(np.hypot(ex, ez))

            # If close enough in x–y, advance waypoint and transition back to TURNING with zeros
            if dist_xz <= self._reach_tol_xy:
                if self._i < len(self.wp) - 1:
                    self._i += 1
                    self.state = CState.TURNING
                    transitioned = True
                    # outputs remain zero on this transition step
                else:
                    # Last waypoint: hold position
                    v_fwd_cmd = 0.0
                    vz_cmd    = 0.0
                    omega_cmd = 0.0
                    self.terminal = True
            else:
                # Command velocities that would reach the waypoint in one step,
                # then clamp the TOTAL speed magnitude (not per-axis).
                vx_des = ex / self.dt         # desired forward speed along body-x
                vz_des = ez / self.dt         # desired vertical speed along world-y

                # Compute total (2D) speed in the allowed motion plane.
                speed = float(np.hypot(vx_des, vz_des))
                vmax  = float(self.lim.max_lin_speed)

                if speed > vmax and speed > 1e-12:
                    scale = vmax / speed
                    vx_des *= scale
                    vz_des *= scale

                v_fwd_cmd = float(vx_des)
                vz_cmd    = float(vz_des)
                omega_cmd = 0.0  # no rotation in FLYING

        debug = {
            "step": self.step_number,
            "state": self.state.name,
            "i_wp": self._i,
            "yaw": float(yaw),
            "yaw_des": float(yaw_des),
            "yaw_err": float(yaw_err),
            "transitioned_this_step": transitioned,
        }
        return {
            "v_fwd": v_fwd_cmd,            # body-x velocity command
            "vz": vz_cmd,                  # world-z velocity command (z is up)
            "omega_yaw": omega_cmd,        # yaw rate about +z
            "debug": debug,
        }
