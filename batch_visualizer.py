import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from scipy.stats import ttest_rel,wilcoxon

import os

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

def evaluate_batch_results(root_dir="gnd_test_results", num_tests=30, save_path=None, percentile = 1.0):
    results = []

    for i in range(num_tests):
        test_dir = os.path.join(root_dir, f"test_{i}")
        file_gauss = os.path.join(test_dir, "twb_gauss.g2o")
        file_gnd = os.path.join(test_dir, "twb_gnd.g2o")
        file_gt = os.path.join(test_dir, "twb_gt.g2o")

        try:
            poses_gauss = read_se2_vertices(file_gauss)
            percentile_len = int(percentile * len(poses_gauss))
            poses_gauss = poses_gauss[:percentile_len]
            poses_gnd = read_se2_vertices(file_gnd)[:percentile_len]
            poses_gt = read_se2_vertices(file_gt)[:percentile_len]

            if not (poses_gauss and poses_gnd and poses_gt):
                print(f"[WARNING] Missing or empty file(s) in test_{i}")
                continue

            ape_gauss = compute_ape(poses_gauss, poses_gt)
            ape_gnd = compute_ape(poses_gnd, poses_gt)

            results.append({
                "test_index": i,
                "APE_Gaussian": ape_gauss,
                "APE_GND": ape_gnd
            })

        except Exception as e:
            print(f"[ERROR] Failed to process test_{i}: {e}")

    df = pd.DataFrame(results)
    if save_path:
        df.to_pickle(save_path)
        print(f"[INFO] Results saved to {save_path}")

    return df

if __name__ == "__main__":
    df_results = evaluate_batch_results(root_dir="test_results/exp1_test_results_1", save_path="test_results/exp1_test_results_1/ape_results.pkl", percentile = 0.7)
    print(df_results)


    results_np = df_results.to_numpy()[:,1:]

    # 1. Compute mean APEs
    mean_ape_gaussian = np.mean(results_np[:, 0])
    mean_ape_gnd = np.mean(results_np[:, 1])

    print(f"Mean APE (Gaussian): {mean_ape_gaussian:.6f}")
    print(f"Mean APE (GND):      {mean_ape_gnd:.6f}")

    max_ape_gaussian = np.max(results_np[:, 0])
    max_ape_gnd = np.max(results_np[:, 1])

    print(f"Max APE (Gaussian): {max_ape_gaussian:.6f}")
    print(f"Max APE (GND):      {max_ape_gnd:.6f}")

    # 2. Compute std APEs (sample std; use ddof=0 for population)
    std_ape_gaussian = np.std(results_np[:, 0], ddof=1)
    std_ape_gnd      = np.std(results_np[:, 1], ddof=1)

    print(f"Std APE (Gaussian):  {std_ape_gaussian:.6f}")
    print(f"Std APE (GND):       {std_ape_gnd:.6f}")

    # 2. Evaluate GND improvement (per test, GND must be strictly better)
    gnd_wins = np.sum(results_np[:, 1] < results_np[:, 0])
    total_tests = results_np.shape[0]
    improvement_ratio = gnd_wins / total_tests

    print(f"GND wins in {gnd_wins} out of {total_tests} tests.")
    print(f"GND Improvement Ratio: {improvement_ratio:.2%}")



    # Find indices where GND is not better than Gaussian
    failures = df_results[df_results["APE_GND"] >= df_results["APE_Gaussian"]]

    print(f"Number of failures: {len(failures)}")
    print(failures)
    t_stat, p_value = ttest_rel(df_results["APE_Gaussian"], df_results["APE_GND"])
    print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4e}")
    w_stat, p_value = wilcoxon(df_results["APE_Gaussian"], df_results["APE_GND"])
    print(f"Wilcoxon signed-rank test: W={w_stat}, p={p_value:.4e}")