# glenn_multirobot

Offline plot / APE tools for **Glenn Shimoda** multi-robot (`FileSimulator` / `test1_new_data`) runs.

| Script | Role |
|--------|------|
| `vis_evaluation_helper.py` | Shared I/O + evo APE helpers (SE3/TUM/CSV) |
| `visualizer_multibot_real.py` | Bot0: GT vs GND / pre-comm / full-graph / DPGO |
| `visualizer_multibot_real1.py` | Bot1: same comparison for robot 1 |
| `visualizer_multibot.py` | Earlier bot0 visualizer (older result paths) |
| `vis_data_reshape_helper.py` | Split `fullGraph.g2o` into `fullGraph0/1.g2o` by vertex id |
| `visualize_edges_g2o.py` | Chain/plot odometry edges from `bot*/edges.g2o` |

```bash
PYTHONPATH=python python python/evaluators/glenn_multirobot/visualizer_multibot_real.py
```
