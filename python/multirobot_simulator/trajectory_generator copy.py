from __future__ import annotations

import heapq
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import os, json, random, re

import numpy as np

# =========================
# I/O: .npz (bit-packed) ↔ 3D bool grid
# =========================

def save_voxels_npz(filepath: str, voxels: np.ndarray) -> None:
    """
    Save a 3D voxel grid to a compact .npz:
      - voxels: 3D numpy array of dtype bool or {0,1}; shape (J,K,L) ≡ (Z,Y,X)
    On disk:
      - shape: int32[3] (Z,Y,X)
      - data:  uint8[...] = numpy.packbits(voxels.ravel())
    """
    if voxels.ndim != 3:
        raise ValueError("voxels must be 3D")
    v = (voxels.astype(np.uint8) != 0).ravel(order="C")
    packed = np.packbits(v)
    np.savez_compressed(
        filepath,
        shape=np.array(voxels.shape, dtype=np.int32),
        data=packed,
    )

def load_voxels_npz(filepath: str) -> np.ndarray:
    """
    Load a voxel grid saved by save_voxels_npz(). Returns bool array shape (Z,Y,X).
    """
    with np.load(filepath, allow_pickle=False) as f:
        shape = tuple(int(x) for x in f["shape"])
        flat_n = int(np.prod(shape))
        bits = np.unpackbits(f["data"])[:flat_n]
        return bits.reshape(shape, order="C").astype(bool)

# =========================
# A* in 3D with 26-neighborhood
# =========================

def _neighbors_26():
    nbrs = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                cost = float(np.sqrt(dx*dx + dy*dy + dz*dz))
                nbrs.append((dz, dy, dx, cost))
    return nbrs

_NEI = _neighbors_26()

def _in_bounds(p, shape) -> bool:
    z, y, x = p
    Z, Y, X = shape
    return (0 <= z < Z) and (0 <= y < Y) and (0 <= x < X)

def _h(a, b) -> float:
    # Euclidean heuristic, consistent with 26-connected moves
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))

def astar_path(voxels: np.ndarray, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """
    A* pathfinding on a 3D occupancy grid.
    Inputs:
      - voxels: bool array shape (Z,Y,X); True/1 = obstacle, False/0 = free
      - start, goal: 3-element np arrays [z,y,x] (ints)
    Output:
      - dense path as int array shape [N,3] including start and goal.
        Returns empty array if no path exists.
    """
    if voxels.ndim != 3:
        raise ValueError("voxels must be 3D")
    start = tuple(int(v) for v in start.tolist())
    goal  = tuple(int(v) for v in goal.tolist())

    if not (_in_bounds(start, voxels.shape) and _in_bounds(goal, voxels.shape)):
        return np.empty((0,3), dtype=np.int64)
    if voxels[start] or voxels[goal]:
        return np.empty((0,3), dtype=np.int64)

    open_heap: List[Tuple[float, int, Tuple[int,int,int]]] = []
    g = {start: 0.0}
    came = {}
    counter = 0
    heapq.heappush(open_heap, ( _h(start, goal), counter, start ))
    closed = set()

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            # reconstruct
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return np.array(path, dtype=np.int64)
        closed.add(cur)
        cz, cy, cx = cur
        for dz, dy, dx, w in _NEI:
            nb = (cz+dz, cy+dy, cx+dx)
            if not _in_bounds(nb, voxels.shape): 
                continue
            if voxels[nb]:
                continue
            ng = g[cur] + w
            if ng < g.get(nb, float("inf")):
                g[nb] = ng
                came[nb] = cur
                counter += 1
                heapq.heappush(open_heap, (ng + _h(nb, goal), counter, nb))

    return np.empty((0,3), dtype=np.int64)

# =========================
# Inflate a path and insert as obstacles
# =========================

def inflate_path_into_voxels(voxels: np.ndarray, path: np.ndarray, n: int, *, euclidean: bool=False) -> None:
    """
    In-place: mark as occupied every voxel within radius n of each path point.
    - voxels: bool array (Z,Y,X), modified in-place (True = obstacle)
    - path: int array [N,3] of [z,y,x]
    - n: non-negative inflation radius (Chebyshev by default, Euclidean if euclidean=True)
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if path.size == 0:
        return
    Z, Y, X = voxels.shape
    rr = range(-n, n+1)
    if euclidean:
        r2 = n*n
        offsets = [(dz,dy,dx) for dz in rr for dy in rr for dx in rr
                   if dz*dz + dy*dy + dx*dx <= r2]
    else:
        offsets = [(dz,dy,dx) for dz in rr for dy in rr for dx in rr
                   if max(abs(dz),abs(dy),abs(dx)) <= n]

    for z, y, x in path.astype(int):
        for dz, dy, dx in offsets:
            zz, yy, xx = z+dz, y+dy, x+dx
            if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X:
                voxels[zz, yy, xx] = True

# =========================
# Path simplification (3D line-of-sight shortcutting)
# =========================

def _bresenham_3d(a: Tuple[int,int,int], b: Tuple[int,int,int]) -> List[Tuple[int,int,int]]:
    # 3D Bresenham over integer voxel cells; returns inclusive list from a to b.
    x1, y1, z1 = a[2], a[1], a[0]
    x2, y2, z2 = b[2], b[1], b[0]
    dx, dy, dz = abs(x2-x1), abs(y2-y1), abs(z2-z1)
    sx = 1 if x2 >= x1 else -1
    sy = 1 if y2 >= y1 else -1
    sz = 1 if z2 >= z1 else -1
    x, y, z = x1, y1, z1
    line = [(z, y, x)]
    if dx >= dy and dx >= dz:
        p1, p2 = 2*dy - dx, 2*dz - dx
        while x != x2:
            x += sx
            if p1 >= 0:
                y += sy; p1 -= 2*dx
            if p2 >= 0:
                z += sz; p2 -= 2*dx
            p1 += 2*dy; p2 += 2*dz
            line.append((z, y, x))
    elif dy >= dx and dy >= dz:
        p1, p2 = 2*dx - dy, 2*dz - dy
        while y != y2:
            y += sy
            if p1 >= 0:
                x += sx; p1 -= 2*dy
            if p2 >= 0:
                z += sz; p2 -= 2*dy
            p1 += 2*dx; p2 += 2*dz
            line.append((z, y, x))
    else:
        p1, p2 = 2*dy - dz, 2*dx - dz
        while z != z2:
            z += sz
            if p1 >= 0:
                y += sy; p1 -= 2*dz
            if p2 >= 0:
                x += sx; p2 -= 2*dz
            p1 += 2*dy; p2 += 2*dx
            line.append((z, y, x))
    return line

def _los_clear(voxels: np.ndarray, a: Tuple[int,int,int], b: Tuple[int,int,int]) -> bool:
    for v in _bresenham_3d(a, b):
        if not _in_bounds(v, voxels.shape) or voxels[v]:
            return False
    return True

def simplify_path(voxels: np.ndarray, dense_path: np.ndarray) -> np.ndarray:
    """
    Greedy line-of-sight simplification.
    - Keeps first and last; removes intermediate points if straight segments are free.
    - Returns sparse path as int array [M,3].
    """
    if dense_path.size == 0:
        return dense_path.astype(np.int64)
    pts = [tuple(int(u) for u in p) for p in dense_path]
    out = [pts[0]]
    anchor = 0
    N = len(pts)
    while anchor < N-1:
        # try to connect anchor → farthest reachable
        j = N-1
        while j > anchor+1 and not _los_clear(voxels, pts[anchor], pts[j]):
            j -= 1
        out.append(pts[j])
        anchor = j
    return np.array(out, dtype=np.int64)


def visualize_voxels(voxels: np.ndarray, show=True) -> None:
    """
    Visualize a 3D voxel grid (bool array [Z,Y,X]) using matplotlib.
    Occupied cells (True) are shown as filled cubes.
    """
    if voxels.ndim != 3:
        raise ValueError("voxels must be 3D")

    Z, Y, X = voxels.shape
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Create 3D coordinates of occupied voxels
    filled = np.argwhere(voxels)  # shape [N,3] with (z,y,x)

    if filled.size == 0:
        print("No occupied voxels to visualize.")
        return

    # Scatter plot
    ax.scatter(filled[:, 2], filled[:, 1], filled[:, 0],
               c="black", marker="s", s=20, alpha=0.8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_zlim(0, Z)
    ax.set_box_aspect((X, Y, Z))  # equal aspect

    plt.tight_layout()
    if show:
        plt.show()


def plot_paths_3d(
    paths: Iterable[np.ndarray],
    labels: Optional[Sequence[str]] = None,
    show: bool = True,
    draw_start_end: bool = True,
    start_marker: str = "o",
    end_marker: str = "X",
    linewidth: float = 2.0,
    alpha: float = 0.95,
    title: Optional[str] = None,
):
    """
    Visualize multiple 3D paths in one plot.

    Args:
        paths: iterable of arrays, each with shape [N,3] (x,y,z). N may differ per path.
        labels: optional list of labels, len == number of paths.
        show: call plt.show() at the end.
        draw_start_end: mark start/end points for each path.
        start_marker, end_marker: matplotlib marker styles for start/end.
        linewidth: line width for path polylines.
        alpha: line transparency (0..1).
        title: optional figure title.
    """
    paths = list(paths)
    if labels is not None and len(labels) != len(paths):
        raise ValueError("labels length must match number of paths")

    # Basic validation and gather bounds
    mins = np.array([np.inf, np.inf, np.inf], dtype=float)
    maxs = -mins
    cleaned_paths = []
    for i, p in enumerate(paths):
        p = np.asarray(p)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError(f"path {i} must have shape [N,3], got {p.shape}")
        # drop rows with NaNs if any
        if np.isnan(p).any():
            p = p[~np.isnan(p).any(axis=1)]
        if p.shape[0] == 0:
            continue
        cleaned_paths.append(p)
        mins = np.minimum(mins, p.min(axis=0))
        maxs = np.maximum(maxs, p.max(axis=0))

    if not cleaned_paths:
        print("No valid paths to plot.")
        return

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot each path
    for idx, p in enumerate(cleaned_paths):
        lbl = labels[idx] if labels is not None else f"path {idx}"
        ax.plot(p[:, 0], p[:, 1], p[:, 2], label=lbl, linewidth=linewidth, alpha=alpha)
        if draw_start_end:
            ax.scatter(p[0, 0], p[0, 1], p[0, 2], marker=start_marker, s=60, label=None)
            ax.scatter(p[-1, 0], p[-1, 1], p[-1, 2], marker=end_marker, s=70, label=None)

    # Axes labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Equal aspect ratio
    ranges = maxs - mins
    max_range = float(np.max(ranges))
    if max_range == 0:
        max_range = 1.0
    centers = (maxs + mins) / 2.0
    ax.set_xlim(centers[0] - max_range/2, centers[0] + max_range/2)
    ax.set_ylim(centers[1] - max_range/2, centers[1] + max_range/2)
    ax.set_zlim(centers[2] - max_range/2, centers[2] + max_range/2)
    ax.set_box_aspect((1, 1, 1))

    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    if show:
        plt.show()

def save_paths(filepath: str, paths: List[np.ndarray]) -> None:
    """
    Save a list of 3D paths (each [N,3]) into one .npz file.
    Paths may have different lengths.
    """
    arrays = {f"path_{i}": p.astype(np.float64) for i, p in enumerate(paths)}
    np.savez_compressed(filepath, **arrays)

def load_paths(filepath: str) -> List[np.ndarray]:
    """
    Load a list of 3D paths from a .npz file saved with save_paths().
    Returns a list of arrays [N,3].
    """
    out = []
    with np.load(filepath, allow_pickle=False) as data:
        # Ensure they come back in original order
        for key in sorted(data.files, key=lambda s: int(s.split("_")[1])):
            out.append(data[key])
    return out

def _is_list_of_lists(x: Any) -> bool:
    return isinstance(x, list) and x and all(isinstance(e, list) for e in x)

def _is_flat_list(x: Any) -> bool:
    return isinstance(x, list) and all(not isinstance(e, (list, dict)) for e in x)

def dumps_outer_list_rows(data: Any, indent: int = 2) -> str:
    def ser(obj: Any, lvl: int) -> str:
        pad = " " * (indent * lvl)
        nxt = " " * (indent * (lvl + 1))

        if isinstance(obj, dict):
            if not obj: return "{}"
            lines = ["{"]
            ol = len(obj.items())
            for i, (k, v) in enumerate(obj.items()):
                vs = ser(v, lvl + 1)
                item = f"{nxt}{json.dumps(k)}: {vs}"
                if i<ol-1:
                    item = item+","
                lines.append(item)
            lines.append(f"{pad}" + "}")
            return "\n".join(lines)

        if _is_list_of_lists(obj):
            if not obj: return "[]"
            rows = [f"{nxt}{json.dumps(row, ensure_ascii=False)}" for row in obj]
            return "[\n" + ",\n".join(rows) + f"\n{pad}]"

        if _is_flat_list(obj):
            return json.dumps(obj, ensure_ascii=False)

        if isinstance(obj, list):
            if not obj: return "[]"
            elems = [f"{nxt}{ser(e, lvl + 1)}" for e in obj]
            return "[\n" + ",\n".join(elems) + f"\n{pad}]"

        return json.dumps(obj, ensure_ascii=False)

    return ser(data, 0)

def dump_outer_list_rows(data: Any, fp: str, indent: int = 4) -> None:
    with open(fp, "w", encoding="utf-8") as f:
        f.write(dumps_outer_list_rows(data, indent))


def bounce(
    pt,
    prev_pt,
    bottom_corner=(0, 0, 0),
    top_corner=(30, 30, 30),
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Outgoing **unit** direction after bouncing off the axis-aligned box at ``pt``.

    Incoming direction is the last segment, ``pt - prev_pt`` (toward the boundary).
    Let **n** be the *outward* unit normal of the hit face (chosen among touching faces
    by the strongest alignment ``v_in·n``). Specular reflection is
    ``v_spec = v_in - 2 (v_in·n) n``.

    We require θᵢ + θₒ > 90° where θᵢ is the angle between **v_in** and **n**, and θₒ
    is the angle between **v_out** and **-n** (outgoing into the box). For mirror
    reflection θₒ = θᵢ. If θᵢ + θₒ ≤ 90° (“steep” / orange case in the sketch), we keep
    **v_out** in the incidence plane but take θₒ = 90° − θᵢ so that θᵢ + θₒ = 90° (+ε),
    i.e. outgoing is rotated away from the pure mirror until the sum exceeds 90°.

    Parameters
    ----------
    pt, prev_pt
        End and previous waypoint (world ``[x,y,z]``, same convention as trajectories).
    bottom_corner, top_corner
        Same half-open box as ``trajectory_generation``: valid lattice cells are
        ``b[i] .. t[i]-1``.

    Returns
    -------
    np.ndarray shape (3,), float64 unit vector along the outgoing ray.
    """
    b = np.asarray(bottom_corner, dtype=float).reshape(3)
    t_top = np.asarray(top_corner, dtype=float).reshape(3)
    t_cell = t_top - 1.0

    p = np.asarray(pt, dtype=float).reshape(3)
    prev = np.asarray(prev_pt, dtype=float).reshape(3)
    seg = p - prev
    seg_norm = np.linalg.norm(seg)
    if seg_norm < tol:
        raise ValueError("bounce: zero-length segment (prev_pt == pt).")
    v_in = seg / seg_norm

    normals: List[np.ndarray] = []
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    if abs(x - b[0]) <= tol:
        normals.append(np.array([-1.0, 0.0, 0.0]))
    if abs(x - t_cell[0]) <= tol:
        normals.append(np.array([1.0, 0.0, 0.0]))
    if abs(y - b[1]) <= tol:
        normals.append(np.array([0.0, -1.0, 0.0]))
    if abs(y - t_cell[1]) <= tol:
        normals.append(np.array([0.0, 1.0, 0.0]))
    if abs(z - b[2]) <= tol:
        normals.append(np.array([0.0, 0.0, -1.0]))
    if abs(z - t_cell[2]) <= tol:
        normals.append(np.array([0.0, 0.0, 1.0]))

    if not normals:
        raise ValueError(f"bounce: pt {pt} is not on the box shell for the given corners.")

    best_i = int(np.argmax([float(np.dot(v_in, nn)) for nn in normals]))
    n = normals[best_i].astype(float)
    n /= np.linalg.norm(n)

    # --- Specular (valid when θᵢ + θₒ = 2θᵢ > 90° for mirror θₒ = θᵢ) ---
    v_spec = v_in - 2.0 * float(np.dot(v_in, n)) * n
    v_spec_n = np.linalg.norm(v_spec)
    if v_spec_n < tol:
        raise ValueError("bounce: degenerate specular direction.")
    v_spec /= v_spec_n

    cos_i = float(np.clip(np.dot(v_in, n), -1.0, 1.0))
    theta_i = float(np.arccos(cos_i))
    cos_o_spec = float(np.clip(-np.dot(v_spec, n), -1.0, 1.0))
    theta_o_spec = float(np.arccos(cos_o_spec))

    min_sum = 0.5 * np.pi + 1e-6
    if theta_i + theta_o_spec > min_sum:
        return v_spec

    # --- Orange case: widen in the incidence plane so θᵢ + θₒ > 90° ---
    sin_i = float(np.sqrt(max(0.0, 1.0 - cos_i * cos_i)))
    if sin_i > 1e-8:
        t_hat = (v_in - cos_i * n) / sin_i
        t_hat /= np.linalg.norm(t_hat)
    else:
        # Nearly normal incidence: build any tangent.
        aux = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(n, aux)) > 0.9:
            aux = np.array([0.0, 1.0, 0.0], dtype=float)
        t_hat = np.cross(n, aux)
        tn = np.linalg.norm(t_hat)
        if tn < tol:
            raise ValueError("bounce: cannot build tangent basis.")
        t_hat = t_hat / tn

    # Target θₒ = 90° − θᵢ (+ε so sum strictly exceeds 90°).
    eta = 1e-4
    theta_o = 0.5 * np.pi - theta_i + eta
    theta_o = float(np.clip(theta_o, eta, 0.5 * np.pi - eta))

    v_out = np.sin(theta_o) * t_hat - np.cos(theta_o) * n
    vo = np.linalg.norm(v_out)
    if vo < tol:
        raise ValueError("bounce: degenerate adjusted direction.")
    return (v_out / vo).astype(np.float64)


def _ray_aabb_exit_t(
    o: np.ndarray,
    d: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    t_eps: float = 1e-8,
) -> Optional[float]:
    """
    Smallest forward parameter t > t_eps where ray o + t d exits the closed AABB
    [b_lo, b_hi], assuming o lies strictly inside or on the boundary and d is unit.
    Uses the standard slab / Kay-Kajiya interval overlap (robust vs per-face tests).
    """
    o = np.asarray(o, dtype=float).reshape(3)
    d = np.asarray(d, dtype=float).reshape(3)
    b_lo = np.asarray(b_lo, dtype=float).reshape(3)
    b_hi = np.asarray(b_hi, dtype=float).reshape(3)
    t_enter = -np.inf
    t_exit = np.inf
    for i in range(3):
        di = float(d[i])
        if abs(di) < 1e-15:
            if o[i] < b_lo[i] - 1e-7 or o[i] > b_hi[i] + 1e-7:
                return None
            continue
        inv = 1.0 / di
        t1 = (b_lo[i] - o[i]) * inv
        t2 = (b_hi[i] - o[i]) * inv
        t_near = min(t1, t2)
        t_far = max(t1, t2)
        t_enter = max(t_enter, t_near)
        t_exit = min(t_exit, t_far)
    if t_enter > t_exit + 1e-9:
        return None
    if t_exit < t_eps:
        return None
    # Origin inside or just within: exit is at t_exit. Origin strictly outside ahead: t_enter.
    if t_enter > t_eps:
        return float(t_enter)
    return float(t_exit)


def bounce_goal(
    pt,
    prev_pt,
    bottom_corner=(0, 0, 0),
    top_corner=(30, 30, 30),
    tol: float = 1e-9,
) -> Tuple[int, int, int]:
    """
    Like ``opposite(pt)`` conceptually as a *next* waypoint: shoot the ``bounce``
    outgoing ray from ``pt`` and return the first hit on the axis-aligned box shell
    (same ``[bottom_corner, top_corner)`` lattice convention as trajectories).

    The shell is the set of lattice bounds ``b[i] .. t[i]-1`` treated as a closed
    axis-aligned box in ℝ³; the returned point is snapped to integer face coords.

    ``pt`` lies on a face, edge, or corner. We nudge the ray origin slightly into the
    interior (sum of inward normals for all touched faces), then use slab-derived
    **exit** time ``t_exit`` so a ray from inside the box always hits the far wall.
    """
    b_lo = np.asarray(bottom_corner, dtype=float).reshape(3)
    t_top = np.asarray(top_corner, dtype=float).reshape(3)
    b_hi = t_top - 1.0
    shell_eps = 1e-6
    step_in = 5e-3
    margin = 1e-5

    o0 = np.asarray(pt, dtype=float).reshape(3)
    p_int = tuple(int(round(x)) for x in o0)
    d = bounce(pt, prev_pt, bottom_corner, top_corner, tol=tol)
    d = np.asarray(d, dtype=float).reshape(3)
    d_norm = np.linalg.norm(d)
    if d_norm < tol:
        raise ValueError("bounce_goal: zero bounce direction.")
    d = d / d_norm

    inward = np.zeros(3, dtype=float)
    touched = False
    for i in range(3):
        if abs(o0[i] - b_lo[i]) <= shell_eps:
            inward[i] += 1.0
            touched = True
        if abs(o0[i] - b_hi[i]) <= shell_eps:
            inward[i] -= 1.0
            touched = True
    if touched and float(np.linalg.norm(inward)) > shell_eps:
        o = o0 + step_in * (inward / np.linalg.norm(inward))
    else:
        o = o0.copy()
    o = np.clip(o, b_lo + margin, b_hi - margin)

    t_hit = _ray_aabb_exit_t(o, d, b_lo, b_hi, t_eps=1e-8)
    if t_hit is None or not np.isfinite(t_hit):
        raise ValueError(
            f"bounce_goal: slab miss from nudged origin for pt={pt} (check box vs direction)."
        )

    q_hit = o + float(t_hit) * d
    snap_tol = 2e-3
    out: List[int] = []
    for j in range(3):
        lo, hi = int(b_lo[j]), int(b_hi[j])
        qj = float(q_hit[j])
        if qj <= lo + snap_tol:
            out.append(lo)
        elif qj >= hi - snap_tol:
            out.append(hi)
        else:
            out.append(int(np.clip(round(qj), lo, hi)))
    result = (out[0], out[1], out[2])
    if result != p_int:
        return result

    b_i = np.asarray(bottom_corner, dtype=int).reshape(3)
    t_i = np.asarray(top_corner, dtype=int).reshape(3)
    p_arr = np.asarray(p_int, dtype=int)
    fallback_arr = (t_i - 1) - (p_arr - b_i) + b_i
    fallback = tuple(int(x) for x in fallback_arr.tolist())
    if fallback != p_int:
        return (int(fallback[0]), int(fallback[1]), int(fallback[2]))

    raise ValueError(
        f"bounce_goal: snapped goal equals pt {pt}; box may be degenerate or bounce axis-aligned to same face."
    )


def trajectory_generation(
    bot_ids,
    traj_path,
    top_corner=(30, 30, 30),
    bottom_corner=(0, 0, 0),
    inflate=3,
    seed=None,
    visualize=False,
    inflate_obstacles_along_path=True,
    n_trajectory_midpoints: int = 0,
):
    """
    Sample opposite / bounce goals on the integer box [bottom_corner, top_corner) and
    plan 3D A* paths with optional corridor inflation.

    ``n_trajectory_midpoints`` chains extra legs from each bot's current endpoint. Legs
    **interleave** ``opposite`` and ``bounce_goal`` (bounce ray to next face): round 0
    uses ``opposite`` only; then odd rounds use ``bounce_goal``, even rounds use
    ``opposite`` (e.g. 2 midpoints → opposite → bounce → opposite).

    **Inflation (when enabled)** uses **only the current leg**: within each midpoint round,
    drone order is **shuffled**; each bot plans against an empty grid plus inflated *dense*
    paths of bots **already planned earlier in that same round** (same segment). No
    cross-leg obstacles.

    Endpoints: if a candidate goal is too close (Chebyshev &lt; ``inflate+1``) to an
    already committed **endpoint**, another goal is drawn from **the same bounding face
    plane(s)** as that candidate, using the same shuffle-and-try pattern as start
    sampling.

    Output JSON: ``{bot_id: [[x,y,z], ...], ...}`` integer world coordinates (as JSON ints).

    Returns ``{"starts": ..., "ends": ..., "paths": ...}`` (``paths`` written to ``traj_path``).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    b = np.array(bottom_corner, dtype=int)
    t = np.array(top_corner, dtype=int)
    if np.any(t <= b):
        raise ValueError("top_corner must be strictly greater than bottom_corner in all dims")

    dims = t - b
    Z, Y, X = int(dims[2]), int(dims[1]), int(dims[0])
    if min(X, Y, Z) <= 0:
        raise ValueError("box must have positive extent in all dimensions")

    def linf(p, q):
        return int(np.max(np.abs(np.array(p, dtype=int) - np.array(q, dtype=int))))

    def opposite(pt):
        pt = np.array(pt, dtype=int)
        return tuple((t - 1) - (pt - b) + b)

    def face_points():
        xs = list(range(b[0], t[0]))
        ys = list(range(b[1], t[1]))
        zs = list(range(b[2], t[2]))
        faces = []
        faces.append([(b[0], y, z) for y in ys for z in zs])
        faces.append([(t[0] - 1, y, z) for y in ys for z in zs])
        faces.append([(x, b[1], z) for x in xs for z in zs])
        faces.append([(x, t[1] - 1, z) for x in xs for z in zs])
        faces.append([(x, y, b[2]) for x in xs for y in ys])
        faces.append([(x, y, t[2] - 1) for x in xs for y in ys])
        return list({tuple(p) for face in faces for p in face})

    def to_grid_idx(p_xyz):
        x, y, z = (np.array(p_xyz, dtype=int) - b).tolist()
        return np.array([z, y, x], dtype=int)

    def to_world_xyz(p_zyx):
        z, y, x = p_zyx.astype(int)
        return [int(x + b[0]), int(y + b[1]), int(z + b[2])]

    def to_world_path(sparse_zyx):
        return [to_world_xyz(p_zyx) for p_zyx in sparse_zyx]

    def same_face_plane(a, c) -> bool:
        """True if lattice points a,c share at least one bounding face of the box."""
        a = tuple(map(int, a))
        c = tuple(map(int, c))
        for ax in range(3):
            for bound in (int(b[ax]), int(t[ax] - 1)):
                if a[ax] == bound and c[ax] == bound:
                    return True
        return False

    def goals_sharing_face(q_seed) -> List[Tuple[int, ...]]:
        pool = [c for c in face_points() if same_face_plane(q_seed, c)]
        random.shuffle(pool)
        return pool

    n_mid = int(max(0, n_trajectory_midpoints))
    n_rounds = n_mid + 1

    starts, ends, paths = {}, {}, {}
    sparse_paths_for_viz = []
    thr = inflate + 1
    bc = (int(b[0]), int(b[1]), int(b[2]))
    tc = (int(t[0]), int(t[1]), int(t[2]))

    for round_idx in range(n_rounds):
        placed_round: List[Tuple[int, ...]] = []
        candidates = face_points()
        order = list(bot_ids)
        random.shuffle(order)
        order_i = {str(b): i for i, b in enumerate(order)}
        dense_this_leg: dict[str, np.ndarray] = {}

        def build_grid_upto(bid: str) -> np.ndarray:
            g = np.zeros((Z, Y, X), dtype=bool)
            if not inflate_obstacles_along_path:
                return g
            for j in range(order_i[bid]):
                ob = str(order[j])
                prev_dense = dense_this_leg.get(ob)
                if prev_dense is not None and prev_dense.size:
                    inflate_path_into_voxels(g, prev_dense, n=inflate)
            return g

        for bid in order:
            bs = str(bid)
            grid = build_grid_upto(bid)
            loc = candidates[:]
            random.shuffle(loc)
            success = False
            while not success:
                if round_idx == 0:
                    if not loc:
                        break
                    p = tuple(map(int, loc.pop()))
                else:
                    p = tuple(map(int, ends[bid]))
                    if any(linf(p, r) < thr for r in placed_round):
                        raise ValueError(
                            f"Start of leg {round_idx} for bot '{bid}' conflicts with "
                            f"placement (inflate={inflate})."
                        )
                if round_idx % 2:
                    prv = paths[bid]
                    if len(prv) < 2:
                        raise ValueError(
                            f"Bot '{bid}' needs at least two waypoints for bounce leg "
                            f"(round {round_idx})."
                        )
                    q_seed = bounce_goal(p, tuple(prv[-2]), bc, tc)
                else:
                    q_seed = opposite(p)
                qt = tuple(map(int, q_seed))
                qopts = (
                    [qt]
                    if round_idx == 0
                    else [qt] + [c for c in goals_sharing_face(q_seed) if c != qt]
                )
                if round_idx:
                    random.shuffle(qopts)
                for q in qopts:
                    if p == q:
                        continue
                    if round_idx == 0:
                        if any(
                            linf(p, r) < thr or linf(q, r) < thr
                            for r in placed_round
                        ):
                            continue
                    elif any(linf(q, r) < thr for r in placed_round):
                        continue
                    dense = astar_path(grid, to_grid_idx(p), to_grid_idx(q))
                    if dense.size == 0:
                        continue
                    sparse = simplify_path(grid, dense)
                    dense_this_leg[bs] = np.array(dense, copy=True)
                    wpath = to_world_path(sparse)
                    if round_idx == 0:
                        paths[bid] = wpath
                        starts[bid] = list(p)
                    elif wpath:
                        paths[bid].extend(wpath[1:])
                    ends[bid] = list(map(int, q))
                    placed_round.extend([p, tuple(q)])
                    candidates = [
                        c for c in candidates if linf(c, p) >= thr and linf(c, q) >= thr
                    ]
                    success = True
                    break
                if round_idx or success:
                    break
            if not success:
                raise ValueError(
                    f"Could not place/plan for bot '{bid}' with inflate={inflate} "
                    f"in box {tuple(b)}..{tuple(t - 1)} (round {round_idx}). "
                    "Try reducing inflate, enlarging the box, or lowering bot count."
                )

        if round_idx == 0:
            for bid in bot_ids:
                sparse_paths_for_viz.append(np.array(paths[bid], dtype=int))
        else:
            for i, bid in enumerate(bot_ids):
                sparse_paths_for_viz[i] = np.array(paths[bid], dtype=int)

    out_dir = os.path.dirname(traj_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2)

    if visualize:
        try:
            plot_paths_3d(sparse_paths_for_viz)
        except Exception:
            pass

    return {"starts": starts, "ends": ends, "paths": paths}


# =========================
# Tiny demo (comment out in production)
# =========================
if __name__ == "__main__":

    file_path, filename = os.path.split(os.path.realpath(__file__))
    Z,Y,X = 30,30,30
    grid = np.zeros((Z,Y,X), dtype=bool)
    # put a random wall
    #grid[15, 5:25, 14:16] = True

    trajectory_generation(
        ["0", "1", "2", "3", "4"],
        "trajectories.json",
        visualize=True,
        n_trajectory_midpoints=3
    )