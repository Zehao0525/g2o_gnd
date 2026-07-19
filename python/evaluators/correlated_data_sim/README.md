# correlated_data_sim

Batch evaluation for **Tutorial_slam2d correlated absolute / GPS-like** experiments (`twb_gauss.g2o`, `twb_gnd.g2o`, `twb_gt.g2o`).

| Script | Role |
|--------|------|
| `batch_visualizer.py` | Over `test_*/`: mean/std APE, GND win rate, paired t-test / Wilcoxon |
| `batch_visualizer2.py` | Same family + per-test trajectory overlay helpers |

```bash
PYTHONPATH=python python python/evaluators/correlated_data_sim/batch_visualizer.py
```
