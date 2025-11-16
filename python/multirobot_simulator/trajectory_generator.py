from __future__ import annotations
import numpy as np
import heapq
from typing import Tuple, List, Optional

import matplotlib.pyplot as plt

import os, json, re, random

from typing import Any
from pathlib import Path

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


import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Sequence

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


def trajectory_generation(
    bot_ids,
    traj_path,
    space_dim = (30,30,30),
    top_corner=(30, 30, 30),
    bottom_corner=(0, 0, 0),
    inflate=3,
    seed=None,
    visualize=False,
    inflate_obstacles_along_path=True
):
    """
    Generate collision-spaced start/end pairs on the faces of an axis-aligned box
    and route 3D A* paths between opposite points. Writes a JSON of trajectories.

    Args
    ----
    bot_ids : list[str]
        IDs for which to create trajectories.
    traj_path : str
        Output JSON path. Format: {bot_id: [[x,y,z], ...], ...}
    top_corner, bottom_corner : tuple[int,int,int]
        Defines the open-top, closed-bottom grid bounds [bottom, top).
        For example: bottom=(0,0,0), top=(30,30,30) → valid coords 0..29.
    inflate : int
        Minimum Chebyshev (L∞) spacing minus one. We enforce
        L∞(p, q) >= inflate + 1 between all starts and ends.
    seed : int|None
        RNG seed for reproducible placement.
    visualize : bool
        If True and helpers exist, will call plot_paths_3d(paths).
    inflate_obstacles_along_path : bool
        If True, calls inflate_path_into_voxels(grid, path, n=inflate).

    Returns
    -------
    dict
        {"starts": {bot_id: [x,y,z]}, "ends": {...}, "paths": {...}}
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    b = np.array(bottom_corner, dtype=int)
    t = np.array(top_corner, dtype=int)
    if np.any(t <= b):
        raise ValueError("top_corner must be strictly greater than bottom_corner in all dims")

    # Grid dimensions and valid coordinate ranges (inclusive indices are b..t-1)
    dims = t - b
    Z, Y, X = dims[2], dims[1], dims[0]  # your demo uses grid[z,y,x]
    grid = np.zeros((Z, Y, X), dtype=bool)

    # Utility: L∞ distance
    def linf(p, q):
        return int(np.max(np.abs(np.array(p) - np.array(q))))

    # Utility: reflect to opposite location across the cube
    # For bottom=(0,0,0), top=(30,30,30), opposite of (0,0,0) → (29,29,29)
    def opposite(pt):
        pt = np.array(pt, dtype=int)
        return (t - 1) - (pt - b) + b

    # List all integer points on each face
    def face_points():
        xs = list(range(b[0], t[0]))
        ys = list(range(b[1], t[1]))
        zs = list(range(b[2], t[2]))

        faces = []
        # x=b0 or x=t0-1
        faces.append([(b[0], y, z) for y in ys for z in zs])         # x-min
        faces.append([(t[0]-1, y, z) for y in ys for z in zs])       # x-max
        # y=b1 or y=t1-1
        faces.append([(x, b[1], z) for x in xs for z in zs])         # y-min
        faces.append([(x, t[1]-1, z) for x in xs for z in zs])       # y-max
        # z=b2 or z=t2-1
        faces.append([(x, y, b[2]) for x in xs for y in ys])         # z-min
        faces.append([(x, y, t[2]-1) for x in xs for y in ys])       # z-max
        # Flatten deduplicated (edges/corners appear multiple times)
        all_pts = {tuple(p) for face in faces for p in face}
        return list(all_pts)

    candidates = face_points()

    starts, ends, paths = {}, {}, {}
    sparse_paths_for_viz = []
    placed = []  # keep both starts and ends for spacing checks (Chebyshev)

    for bid in bot_ids:
        success = False
        # Work on a local shuffled copy so retries don't disturb global order.
        local_candidates = candidates[:]
        random.shuffle(local_candidates)

        while local_candidates:
            p = local_candidates.pop()         # candidate start [x,y,z]
            q = tuple(opposite(p))             # opposite end

            # Check spacing to everything already placed (both starts and ends)
            if any(linf(p, r) < (inflate + 1) or linf(q, r) < (inflate + 1) for r in placed):
                continue  # try another candidate

            # Try plan immediately. Convert to grid indices [z,y,x].
            s = np.asarray(p, dtype=int)
            g = np.asarray(q, dtype=int)
            def to_grid_idx(p_xyz):
                x, y, z = (p_xyz - b).astype(int)
                return np.array([z, y, x], dtype=int)
            s_idx = to_grid_idx(s)
            g_idx = to_grid_idx(g)

            dense_path = astar_path(grid, s_idx, g_idx)
            if dense_path.size == 0:
                # A* failed → retry a different start
                continue

            # Success: simplify & (optionally) inflate obstacles along the dense path
            sparse_path = simplify_path(grid, dense_path)
            if inflate_obstacles_along_path:
                inflate_path_into_voxels(grid, dense_path, n=inflate)

            # Commit placement and record outputs
            starts[bid] = list(map(int, s))
            ends[bid]   = list(map(int, g))

            def to_world_xyz(p_zyx):
                z, y, x = p_zyx.astype(int)
                return [int(x + b[0]), int(y + b[1]), int(z + b[2])]

            def _to_world_list(seq_zyx):
                return [to_world_xyz(p_zyx) for p_zyx in seq_zyx]

            out_path = _to_world_list(sparse_path)
            paths[bid] = out_path
            sparse_paths_for_viz.append(np.array(out_path, dtype=int))

            # Mark these as taken to enforce spacing for later bots
            placed.extend([tuple(s), tuple(g)])

            # Prune global candidates to speed later bots
            candidates = [
                c for c in candidates
                if linf(c, s) >= (inflate + 1) and linf(c, g) >= (inflate + 1)
            ]

            success = True
            break  # done with this bot

        if not success:
            raise ValueError(
                f"Could not place/plan for bot '{bid}' with inflate={inflate} in box {tuple(b)}..{tuple(t-1)}. "
                "Try reducing inflate, enlarging the box, or removing obstacles."
            )
        

    # Write JSON: {bot_id: [[x,y,z], ...], ...}
    os.makedirs(os.path.dirname(traj_path), exist_ok=True) if os.path.dirname(traj_path) else None
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2)

    # Optional visualization if your helpers are available
    if visualize:
        try:
            plot_paths_3d(sparse_paths_for_viz)
        except Exception:
            pass

    return {"starts": starts, "ends": ends, "paths": paths}


import os, json, random
import numpy as np

def trajectory_generation(
    bot_ids,
    traj_path,
    top_corner=(30, 30, 30),
    bottom_corner=(0, 0, 0),
    inflate=3,
    seed=None,
    visualize=False,
    inflate_obstacles_along_path=True,
    space_dim=(30, 30, 30),
):
    """
    Plan in voxel index space of shape space_dim=(X,Y,Z), then map to continuous/world
    space bounded by bottom_corner..top_corner.

    Output JSON format: {bot_id: [[x,y,z], ...], ...} in WORLD coordinates (floats).
    Returns dict with both voxel indices and world paths.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # --- Voxel space setup (index domain) ---
    X, Y, Z = map(int, space_dim)                    # voxel dims (x,y,z)
    if min(X, Y, Z) <= 0:
        raise ValueError("space_dim must be positive in all dimensions")
    grid = np.zeros((Z, Y, X), dtype=bool)           # your A* uses [z,y,x]

    # --- World-space affine mapping (idx -> world) ---
    b = np.array(bottom_corner, dtype=float)
    t = np.array(top_corner, dtype=float)
    if np.any(t <= b):
        raise ValueError("top_corner must be strictly greater than bottom_corner in all dims")
    span = t - b
    denom = np.maximum(np.array([X-1, Y-1, Z-1], dtype=float), 1.0)  # avoid /0 if dim==1
    scale = span / denom

    def idx_to_world(p_xyz_idx):
        """Map voxel index [x,y,z] to world [x,y,z] (float)."""
        p = np.array(p_xyz_idx, dtype=float)
        return (b + p * scale).tolist()

    # --- Helpers in voxel index space ---
    def linf(p, q):  # Chebyshev distance in voxels
        return int(np.max(np.abs(np.array(p) - np.array(q))))

    def opposite_voxel(p):  # reflect across voxel box
        px, py, pz = map(int, p)
        return (X-1 - px, Y-1 - py, Z-1 - pz)

    def face_points_voxel():
        xs = range(X); ys = range(Y); zs = range(Z)
        faces = []
        faces.append([(0, y, z) for y in ys for z in zs])         # x-min
        faces.append([(X-1, y, z) for y in ys for z in zs])       # x-max
        faces.append([(x, 0, z) for x in xs for z in zs])         # y-min
        faces.append([(x, Y-1, z) for x in xs for z in zs])       # y-max
        faces.append([(x, y, 0) for x in xs for y in ys])         # z-min
        faces.append([(x, y, Z-1) for x in xs for y in ys])       # z-max
        # dedupe edges/corners
        return list({tuple(p) for face in faces for p in face})

    candidates = face_points_voxel()

    # --- Planning loop (voxel space) ---
    starts_vox, ends_vox, paths_world = {}, {}, {}
    sparse_paths_for_viz = []
    placed = []  # keep both starts & ends to enforce spacing

    for bid in bot_ids:
        success = False
        local = candidates[:]
        random.shuffle(local)

        while local:
            s_idx_xyz = local.pop()              # voxel start [x,y,z]
            g_idx_xyz = opposite_voxel(s_idx_xyz)

            # spacing in voxel units
            if any(linf(s_idx_xyz, r) < (inflate + 1) or linf(g_idx_xyz, r) < (inflate + 1) for r in placed):
                continue

            # Convert to A* indexing [z,y,x]
            s_zyx = np.array([s_idx_xyz[2], s_idx_xyz[1], s_idx_xyz[0]], dtype=int)
            g_zyx = np.array([g_idx_xyz[2], g_idx_xyz[1], g_idx_xyz[0]], dtype=int)

            dense = astar_path(grid, s_zyx, g_zyx)
            if dense.size == 0:
                continue  # try another candidate

            sparse = simplify_path(grid, dense)
            if inflate_obstacles_along_path:
                inflate_path_into_voxels(grid, dense, n=inflate)

            # Commit
            starts_vox[bid] = list(map(int, s_idx_xyz))
            ends_vox[bid]   = list(map(int, g_idx_xyz))

            # Map sparse voxels [z,y,x] -> [x,y,z] idx -> world
            def zyx_to_xyz_idx(p_zyx):
                z, y, x = map(int, p_zyx)
                return [x, y, z]

            world_path = [idx_to_world(zyx_to_xyz_idx(p)) for p in sparse]
            paths_world[bid] = world_path
            sparse_paths_for_viz.append(np.array(world_path, dtype=float))

            # enforce spacing for later bots & prune global candidates
            placed.extend([tuple(s_idx_xyz), tuple(g_idx_xyz)])
            candidates = [
                c for c in candidates
                if linf(c, s_idx_xyz) >= (inflate + 1) and linf(c, g_idx_xyz) >= (inflate + 1)
            ]
            success = True
            break

        if not success:
            raise ValueError(
                f"Could not place/plan for bot '{bid}' with inflate={inflate} "
                f"in voxel box 0..({X-1},{Y-1},{Z-1}). "
                "Reduce inflate, enlarge space_dim, or relax obstacles."
            )

    # --- Write world-space trajectories ---
    cur_path = Path(__file__).resolve().parent
    traj_path = os.path.join(cur_path, traj_path)
    out_dir = os.path.dirname(traj_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(paths_world, f, indent=2)

    if visualize:
        try:
            plot_paths_3d(sparse_paths_for_viz)
        except Exception:
            pass

    return paths_world,    # world coordinates (floats)


# =========================
# Tiny demo (comment out in production)
# =========================
if __name__ == "__main__":

    file_path, filename = os.path.split(os.path.realpath(__file__))
    Z,Y,X = 30,30,30
    grid = np.zeros((Z,Y,X), dtype=bool)
    # put a random wall
    #grid[15, 5:25, 14:16] = True

    trajectory_generation(['0','1','2','3','4'], "trajectories.json", visualize=True)

    bot_missions = [(np.array([0,0,0]), np.array([29,29,29])),
                    (np.array([0,0,29]), np.array([29,29,0])), 
                    (np.array([0,29,0]), np.array([29,0,29])),
                    (np.array([29,0,0]), np.array([0,29,29])), 
                    (np.array([15,0,0]), np.array([15,29,29]))]

    paths = []

    for s,g in bot_missions:
        path = astar_path(grid, s, g)
        print("dense path len:", len(path))
        if path.size:
            sp = simplify_path(grid, path)
            print("sparse path len:", len(sp))
            inflate_path_into_voxels(grid, path, n=3)
            paths.append(sp)
    
    map_pth = os.path.join(file_path,"map.npz")
    save_voxels_npz(map_pth, grid)
    # reloaded = load_voxels_npz(map_pth)
    # assert np.array_equal(grid, reloaded)

    #visualize_voxels(grid)
    plot_paths_3d(paths)

    output_to_config = False
    if output_to_config:
        config_pth = os.path.join(file_path,"config","sim_config.json")

        if os.path.exists(config_pth):
            with open(config_pth, "r") as f:
                config_dict = json.load(f)
        else:
            config_dict = {"bots" : {}}

        for i,p in enumerate(paths):
            config_dict.setdefault("bots", {})
            config_dict["bots"].setdefault(str(i), {})
            config_dict["bots"][str(i)].update({"path" : np.array(p).tolist()})
        dump_outer_list_rows(config_dict,config_pth)
    else:
        config_pth = os.path.join(file_path,"config","trajectories.json")
        config_dict =  {}

        for i,p in enumerate(paths):
            config_dict[str(i)] = np.array(p).tolist()
        
        dump_outer_list_rows(config_dict,config_pth)

    
    #save_paths(path_pth, paths)
