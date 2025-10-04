input_path = "test_results/multirobot/fullGraph.g2o"
output_path0 = "test_results/multirobot/fullGraph0.g2o"
output_path1 = "test_results/multirobot/fullGraph1.g2o"

id_threshold = 4225


with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path0, "w", encoding="utf-8") as fout0, \
            open(output_path1, "w", encoding="utf-8") as fout1:

        for line in fin:

            parts = line.strip().split()
            # Expected format: VERTEX_SE3:QUAT id tx ty tz qx qy qz qw
            if len(parts) < 9:
                continue

            try:
                vid = int(parts[1])
            except ValueError:
                continue

            if vid < id_threshold:
                fout0.write(line)
            else:
                fout1.write(line)