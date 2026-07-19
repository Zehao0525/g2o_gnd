# simulators/multidrone

Python multidrone world / logging layer used to generate data for the C++ Multidrone SLAM stack.

## Batch pipeline (start here)

From the repository root (`PYTHONPATH=python` recommended):

```bash
python python/simulators/multidrone/batch_config_writer.py
python python/simulators/multidrone/generate_batch_scenes.py
python python/simulators/multidrone/batch_simulate_from_scenes.py
```

1. **`batch_config_writer.py`** — fill/update per-bot blocks in `config/sim_config_batch.json`
2. **`generate_batch_scenes.py`** — stage 1: scene folders (`trajectories.json`, optional `landmarks.json`)
3. **`batch_simulate_from_scenes.py`** — stage 2: run `WorldSim` → `gt_log_*` / `msg_log_*` / `bot_ids.txt`

## Layout

| Path | Purpose |
|------|---------|
| entry scripts above | Main pipeline |
| `config/` | Sim / trajectory / landmark JSON |
| `core/` | Library: simulator, controller, trajectory/landmark generators |
| `tools/` | Optional legacy generators and visualizers |

Evaluation: `python/evaluators/multidrone/`
