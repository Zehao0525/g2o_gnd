import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R

import evo
from evo.core import trajectory,metrics,sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core.metrics import APE
from evo.tools import log

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

def read_se3_vertices_as_se2(filename):
    pose_dict = {}  # vertex_id -> (x, y, theta)

    with open(filename, 'r') as file:
        for line in file:
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

                # Store / overwrite to ensure unique IDs
                pose_dict[vertex_id] = (x, y, theta)

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



if __name__ == "__main__":
    import numpy as np

    len_bot0 = 4224

    filename = "test_results/file_trajectory_gt_bot0.g2o"
    results_gt = read_se3_vertices_as_se2(filename)[:len_bot0+1]
    print("len(results_gt)", len(results_gt))

    filename = "test_results/file_trajectory_opt_bot0.g2o"
    results_gnd = read_se3_vertices_as_se2(filename)
    results_gnd_edge = results_gnd[:len_bot0+1]
    print("len(results_gnd)", len(results_gnd))


    filename = "test_results/file_trajectory_opt_wognd_bot0.g2o"
    results_nognd = read_se3_vertices_as_se2(filename)
    results_nognd_edge = results_nognd[:len_bot0+1]
    print("len(results_gnd)", len(results_nognd_edge))

    filename = "test_results/file_trajectory_pre_comm_bot0.g2o"
    result_before = read_se3_vertices_as_se2(filename)[:len_bot0+1]
    print("len(result_before)", len(result_before))

    filename = "test_results/file_trajectory_pre_opt_bot0.g2o"
    result_before_opt = read_se3_vertices_as_se2(filename)[:len_bot0+1]
    print("len(result_before_opt)", len(result_before_opt))


    filename = "test_results/bot0_observation_vtxs_refs.g2o"
    obs_vtxs_bot0 = read_se3_vertices_as_se2(filename)[:4]
    obs_vtxs_ids = np.array(obs_vtxs_bot0)[:,0]
    selected = [p for p in results_gnd if p[0] in obs_vtxs_ids]
    print("len(selected)", len(selected))




    filename = "test1_new_data/bot1/vertices.g2o"
    bot1_gt = read_se3_vertices_as_se2(filename)
    print("len(bot1_gt)", len(bot1_gt))

    filename = "test1_new_data/gt.tum"
    tum_gt_bot0 = read_tum_vertices_as_se2(filename)


    gt_pose_path = list_to_pose_path(tum_gt_bot0)
    pre_opt_pose_path = list_to_pose_path(result_before_opt)
    gauss_opt_pose_path = list_to_pose_path(results_nognd_edge)
    gnd_opt_pose_path = list_to_pose_path(results_gnd_edge)

    # no opt
    traj_ref, traj_est = sync.associate_trajectories(gt_pose_path, pre_opt_pose_path)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est))

    print(f"ATE pre opt: {ape_metric.rmse:.4f}")


    # gauss opt
    traj_ref, traj_est = sync.associate_trajectories(gt_pose_path, gauss_opt_pose_path)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est))

    print(f"ATE pre opt: {ape_metric.rmse:.4f}")


    # gnd opt
    traj_ref, traj_est = sync.associate_trajectories(gt_pose_path, gnd_opt_pose_path)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est))

    print(f"ATE pre opt: {ape_metric.rmse:.4f}")





    plt.figure(figsize=(8, 6))

    #print("gts reviewed:", obs_vtxs_bot0)
    #print("graph vtxs reviewed:", selected)
    obs_source = [results_gnd[2619]]
    plot_landmarks(obs_source, color = 'orange', label = '')
    plot_landmarks(obs_vtxs_bot0, color = 'red', label = 'gt vtxs')
    plot_landmarks(selected, color = 'purple', label = 'bot0 vtxs')

    alpha = 0.7
    #plot_trajectory(bot1_gt, 'red', False, 'Bot1', alpha = 0.5)
    plot_trajectory(tum_gt_bot0, 'green', False, 'ground truth', alpha = alpha)
    #plot_trajectory(result_before, 'orange', False, 'no comms', alpha = alpha)
    #plot_trajectory(result_before_opt, 'purple', False, 'gnd comms edge', alpha = alpha)
    plot_trajectory(results_nognd_edge, 'red', False, 'gnd comms edge', alpha = alpha)
    plot_trajectory(results_gnd_edge, 'blue', False, 'gnd comms edge', alpha = alpha)
    print("APE of gnd:", compute_ape(results_gnd_edge, results_gt))
    print("APE before optimization:", compute_ape(result_before, results_gt))
    plt.title('SE2 Trajectory with Colored Orientation Arrows')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()