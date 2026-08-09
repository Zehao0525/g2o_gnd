import matplotlib.pyplot as plt

try:
    from evaluators.glenn_multirobot.vis_evaluation_helper import *
except ModuleNotFoundError:
    from vis_evaluation_helper import *


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
    filename = "test_results/glenn_multirobot/multibot_full_graph/full_graph1.g2o"
    results_gt = read_se3_vertices_as_se2(filename, realign=True)
    print("len(results_ggd)", len(results_gt))

    # Our method
    filename = "test_results/glenn_multirobot/multi_round/file_trajectory_opt_bot1.g2o"
    results_ggd = read_se3_vertices_as_se2(filename, idbound=(4225,9000))
    print("len(results_ggd)", len(results_ggd))

    filename = "test_results/glenn_multirobot/single_round/file_trajectory_opt_bot1.g2o"
    results_ggd_sr = read_se3_vertices_as_se2(filename, idbound=(4225,9000))
    print("len(results_ggd)", len(results_ggd))

    # No communications
    filename = "test_results/glenn_multirobot/file_trajectory_pre_comm_bot1.g2o"
    result_before = read_se3_vertices_as_se2(filename, idbound=(4225,9000))
    print("len(result_before)", len(result_before))

    # DPGO
    filename = "test_results/glenn_multirobot/DPGO_results/robot1/trajectory_optimized.csv"
    result_dpgo = csv_to_xytheta_list(filename, realign=True)




    # DPGO
    filename = "test_results/glenn_multirobot/gauss/multi_round/file_trajectory_opt_bot1.g2o"
    result_gauss = read_se3_vertices_as_se2(filename, realign=True)
    len(result_gauss)
    print("len(result_gauss)", len(result_gauss))


    # DPGO
    #filename = "test_results/glenn_multirobot/gauss/file_trajectory_opt_bot1.csv"
    #result_gauss_sr = csv_to_xytheta_list(filename, realign=True)




    filename = "test_data/glenn_multirobot/test1_new_data/bot1/vertices.g2o"
    bot1_gt = read_se3_vertices_as_se2(filename, idbound=(4225,9000))
    print("len(bot1_gt)", len(bot1_gt))

    # GT
    filename = "test_data/glenn_multirobot/test1_new_data/bot1/gt1.tum"
    tum_gt_bot0 = read_tum_vertices_as_se2(filename)
    


    gt_pose_path = list_to_pose_path(tum_gt_bot0)
    pre_opt_pose_path = list_to_pose_path(result_before)
    ggd_opt_pose_path = list_to_pose_path(results_ggd)
    dpgo_pose_path = list_to_pose_path(result_dpgo)





    ape_gauss,ape_gauss_aligned = ape_evaluate(tum_gt_bot0, result_gauss, tag = "=== ATE (gauss) ===")
    #ape_ggd,ape_ggd_aligned = ape_evaluate(tum_gt_bot0, result_gauss_sr, tag = "=== ATE (gauss_sr) ===")

    plot_ape_colormap(ape_gauss["traj_est"], ape_gauss["traj_ref"], ape_metric=ape_gauss["data"], plot_mode="xy")


    ape_full,ape_full_aligned = ape_evaluate(tum_gt_bot0, results_gt, tag = "=== ATE (full) ===")

    ape_before,ape_before_aligned = ape_evaluate(tum_gt_bot0, result_before, tag = "=== ATE (before) ===")

    ape_dogo,ape_dpgo_aligned = ape_evaluate(tum_gt_bot0, result_dpgo, tag = "=== ATE (dpgo) ===")

    ape_ggd,ape_ggd_aligned = ape_evaluate(tum_gt_bot0, results_ggd, tag = "=== ATE (ggd) ===")

    ape_ggd_sr,ape_ggd_sr_aligned = ape_evaluate(tum_gt_bot0, results_ggd_sr, tag = "=== ATE (ggd_sr) ===")




    plot_ape_colormap(ape_ggd["traj_est"], ape_ggd["traj_ref"], ape_metric=ape_ggd["data"], plot_mode="xy")


    plt.figure(figsize=(8, 6))

    #print("gts reviewed:", obs_vtxs_bot0)
    #print("graph vtxs reviewed:", selected)
    obs_source = [results_ggd[2619]]
    plot_landmarks(obs_source, color = 'orange', label = '')
    # plot_landmarks(obs_vtxs_bot0, color = 'red', label = 'gt vtxs')
    # plot_landmarks(selected, color = 'purple', label = 'bot0 vtxs')

    alpha = 0.7
    #plot_trajectory(bot1_gt, 'red', False, 'Bot1', alpha = 0.5)
    plot_trajectory(tum_gt_bot0, 'green', False, 'ground truth', alpha = alpha)
    #plot_trajectory(result_before, 'orange', False, 'no communications', alpha = alpha)
    plot_trajectory(results_ggd, 'blue', False, 'ggd comms edge', alpha = alpha)
    plot_trajectory(results_ggd_sr, 'cyan', False, 'ggd comms edge', alpha = alpha)
    plot_trajectory(results_gt, 'purple', False, 'full graph optimization', alpha = alpha)
    plot_trajectory(result_dpgo, 'pink', False, 'dpgo', alpha = alpha)
    #print("APE of ggd:", compute_ape(results_ggd_edge, results_gt))
    #print("APE before optimization:", compute_ape(result_before, results_gt))
    plt.title('Multi-Robot estimated trajectories')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()