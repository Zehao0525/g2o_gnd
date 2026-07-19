# comp0250_reimplement

Plotters for the **COMP0249 / COMP0250-style** 2D landmark SLAM reimplementation (`incsim_test` and related Tutorial_slam2d outputs).

| Script | Role |
|--------|------|
| `visualizer.py` | Plot SE2 trajectories from `.g2o` (`choice=0` incsim; also has older Cauchy / `twb_*` modes) |

```bash
PYTHONPATH=python python python/evaluators/comp0250_reimplement/visualizer.py
```
