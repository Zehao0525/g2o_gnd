import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R

from vis_evaluation_helper import *

from evo.tools import plot


def ape_evaluate(t_ref, t_est, tag = "=== ATE (full graph) ==="):
    ape = compute_ape_with_evo(t_ref, t_est)
    ape_aligned = compute_ape_with_evo(t_ref, t_est, align = True)
    print(tag)
    print("mean: ", ape["mean"], "    aligned mean: ", ape_aligned["mean"])
    print("max: ", ape["max"], "    aligned max: ", ape_aligned["max"])
    print("std: ", ape["std"], "    aligned std: ", ape_aligned["std"])
    return ape, ape_aligned



if __name__ == "__main__":
    import numpy as np

    len_bot0 = 4224

    # full graph optimization
    filename = "test_results/multirobot/fullGraph0.g2o"
    results_gt = read_se3_vertices_as_se2(filename, realign=True)
    print("len(results_gnd)", len(results_gt))

    # Our method
    filename = "test_results/multirobot/file_trajectory_opt_bot0.g2o"
    results_gnd = read_se3_vertices_as_se2(filename, idbound=(0,4225))
    print("len(results_gnd)", len(results_gnd))

        # Our method
    filename = "test_results/multirobot/multi_round/file_trajectory_opt_bot0.g2o"
    results_gnd_sr = read_se3_vertices_as_se2(filename, idbound=(0,4225))
    print("len(results_gnd)", len(results_gnd))

    # No communications
    filename = "test_results/multirobot/file_trajectory_pre_comm_bot0.g2o"
    result_before = read_se3_vertices_as_se2(filename, idbound=(0,4225))
    print("len(result_before)", len(result_before))

    # DPGO
    filename = "test_results/DPGO_results/robot0/trajectory_optimized.csv"
    result_dpgo = csv_to_xytheta_list(filename)




    filename = "test_data/test1_new_data/bot1/vertices.g2o"
    bot1_gt = read_se3_vertices_as_se2(filename, idbound=(0,4225))
    print("len(bot1_gt)", len(bot1_gt))

    # GT
    filename = "test_data/test1_new_data/bot0/gt0.tum"
    tum_gt_bot0 = read_tum_vertices_as_se2(filename)
    


    gt_pose_path = list_to_pose_path(tum_gt_bot0)
    pre_opt_pose_path = list_to_pose_path(result_before)
    gnd_opt_pose_path = list_to_pose_path(results_gnd)
    gnd_sr_opt_pose_path = list_to_pose_path(results_gnd_sr)
    dpgo_pose_path = list_to_pose_path(result_dpgo)


    ape_full,ape_full_aligned = ape_evaluate(tum_gt_bot0, results_gt, tag = "=== ATE (full) ===")

    ape_before,ape_before_aligned = ape_evaluate(tum_gt_bot0, result_before, tag = "=== ATE (before) ===")

    ape_dogo,ape_dpgo_aligned = ape_evaluate(tum_gt_bot0, result_dpgo, tag = "=== ATE (dpgo) ===")

    ape_gnd,ape_gnd_aligned = ape_evaluate(tum_gt_bot0, results_gnd, tag = "=== ATE (gnd) ===")

    ape_gnd,ape_gnd_aligned = ape_evaluate(tum_gt_bot0, results_gnd_sr, tag = "=== ATE (gnd_sr) ===")

    
    plot_ape_colormap(ape_gnd["traj_est"], ape_gnd["traj_ref"], ape_metric=ape_gnd["data"], plot_mode="xy")


    plt.figure(figsize=(8, 6))

    #print("gts reviewed:", obs_vtxs_bot0)
    #print("graph vtxs reviewed:", selected)
    obs_source = [results_gnd[2619]]
    plot_landmarks(obs_source, color = 'orange', label = '')
    # plot_landmarks(obs_vtxs_bot0, color = 'red', label = 'gt vtxs')
    # plot_landmarks(selected, color = 'purple', label = 'bot0 vtxs')

    alpha = 0.7
    #plot_trajectory(bot1_gt, 'red', False, 'Bot1', alpha = 0.5)
    plot_trajectory(tum_gt_bot0, 'green', False, 'ground truth', alpha = alpha)
    plot_trajectory(result_before, 'orange', False, 'no communications', alpha = alpha)
    plot_trajectory(results_gnd, 'blue', False, 'gnd comms edge', alpha = alpha)
    plot_trajectory(results_gt, 'purple', False, 'full graph optimization', alpha = alpha)
    plot_trajectory(result_dpgo, 'pink', False, 'dpgo', alpha = alpha)
    #print("APE of gnd:", compute_ape(results_gnd_edge, results_gt))
    #print("APE before optimization:", compute_ape(result_before, results_gt))
    plt.title('Multi-Robot estimated trajectories')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()