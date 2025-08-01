import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

if __name__ == "__main__":
    import numpy as np

    choice = 1
    if choice == 0:
        filename = "trajectory_est.g2o"
        se2_poses_after = read_se2_vertices(filename)
        

        filename = "trajectory_before.g2o"
        se2_poses_before = read_se2_vertices(filename)

        filename = "trajectory_gt.g2o"
        se2_poses_gt = read_se2_vertices(filename)

        plt.figure(figsize=(8, 6))

        plot_trajectory(se2_poses_gt, 'green', False, 'ground truth')
        plot_trajectory(se2_poses_after, 'orange', False, 'after')
        plot_trajectory(se2_poses_before, 'blue', False, 'before')
        plt.title('SE2 Trajectory with Colored Orientation Arrows')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()
    if choice == 1:
        filename = "cauchy_cauchy_after.g2o"
        se2_kernel_poses_after = read_se2_vertices(filename)

        filename = "cauchy_cauchy_back_after.g2o"
        se2_kernel_clone_poses_after = read_se2_vertices(filename)
        

        filename = "cauchy_gauss_after.g2o"
        se2_poses_after = read_se2_vertices(filename)
        

        filename = "cauchy_gauss_before.g2o"
        se2_poses_before = read_se2_vertices(filename)

        filename = "cauchy_gauss_gt.g2o"
        se2_poses_gt = read_se2_vertices(filename)
    

        plt.figure(figsize=(8, 6))

        plot_trajectory(se2_poses_gt, 'green', False, 'ground truth')
        plot_trajectory(se2_poses_after, 'orange', False, 'after')
        print("APE of gaussian:", compute_ape(se2_poses_after, se2_poses_gt))
        plot_trajectory(se2_poses_before, 'blue', False, 'before')
        plot_trajectory(se2_kernel_poses_after, 'red', False, 'kernel')
        plot_trajectory(se2_kernel_clone_poses_after, 'purple', False, 'kernel2')
        print("APE of gnd:", compute_ape(se2_kernel_poses_after, se2_poses_gt))
        print("APE of gnd w gauss init est:", compute_ape(se2_kernel_clone_poses_after, se2_poses_gt))
        plt.title('SE2 Trajectory with Colored Orientation Arrows')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()