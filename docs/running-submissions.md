# Running submission pipelines

Run these commands from the **repository root** (the folder that contains `pipeline/`, `data/`, `outputs/`, etc.), unless noted otherwise. Use a virtual environment that has the project dependencies installed (e.g. `pandas`, `numpy`, `lightgbm`, `pyarrow`; PyTorch for the DL experiment scripts).

## Data layout (LightGBM pipelines)

The six LightGBM strategies read Parquet files from:

`data/raw/combined/train.parquet`  
`data/raw/combined/test.parquet`  

They write logs under `logs/`, metrics and CSV outputs under `outputs/`, and may use `models/` at the repo root. Directories are created if missing.

## LightGBM (`pipeline/lgbm/submissions/`)

Each script is self-contained: no CLI flags. Outputs go to `outputs/` as listed.

| Script | Description | Typical outputs under `outputs/` |
|--------|-------------|----------------------------------|
| `09_advanced_lgbm.py` | Advanced features, multi-seed LightGBM per horizon | `advanced_lgbm_results.txt`, `submission.csv` (if test exists) |
| `10_enhanced_lgbm.py` | Enhanced + discovered interaction features | `enhanced_lgbm_results.txt`, `enhanced_submission.csv` |
| `11_perhorizon_ic.py` | Per-horizon models + IC analysis | `perhorizon_ic_results.txt`, `ic_heatmap_data.csv`, `feature_group_ranking.csv`, `perhorizon_ic_submission.csv` |
| `12_weight_decay.py` | Temporal decay on sample weights | `weight_decay_results.txt`, `weight_decay_ic_heatmap.csv`, `weight_decay_group_ranking.csv`, `weight_decay_submission.csv` |
| `13_per_horizon_per_sub_category.py` | One model per (horizon, sub_category) | `per_horizon_per_sub_category_results.txt`, `per_horizon_per_sub_category_submission.csv` |
| `14_per_sub_category_per_horizon.py` | One model per (sub_category, horizon) | `per_sub_category_per_horizon_results.txt`, `per_sub_category_per_horizon_submission.csv` |

Examples:

```bash
python3 pipeline/lgbm/submissions/09_advanced_lgbm.py
python3 pipeline/lgbm/submissions/10_enhanced_lgbm.py
python3 pipeline/lgbm/submissions/11_perhorizon_ic.py
python3 pipeline/lgbm/submissions/12_weight_decay.py
python3 pipeline/lgbm/submissions/13_per_horizon_per_sub_category.py
python3 pipeline/lgbm/submissions/14_per_sub_category_per_horizon.py
```

Shared library code lives in `pipeline/lgbm/` (`01_paths.py` … `08_temporal_decay.py`); the submission scripts load those modules automatically.

PyTorch scripts `15`–`17` load `pipeline/deeplearning/01_paths.py`, `02_logging.py`, and
other numbered modules next to `submissions/` in the same way.

## Deep learning — main LightGBM-style pipeline (`09_main_pipeline.py`)

Tree models on engineered features (eval / full train / inference from saved models).

| Argument | Description |
|----------|-------------|
| `--mode` | `eval` (default): 80/20 chronological eval. `submit`: train on full train + optional test predictions. `infer`: load checkpoints from disk + predict. |
| `--train_data` | Path to training Parquet (default: `train.parquet`, relative to **current working directory**). |
| `--test_data` | Path to test Parquet for `submit` / `infer` (default: `test.parquet`). |

Saved model files: `pipeline/deeplearning/models/lgb_horizon_{1,3,10,25}.txt`.

**Submission file:** `submit` and `infer` write **`submission.csv` in the current working directory** (not under `outputs/`).

Examples:

```bash
# Evaluation only (paths relative to cwd; use absolute paths if needed)
python3 pipeline/deeplearning/submissions/09_main_pipeline.py --mode eval \
  --train_data data/raw/combined/train.parquet

python3 pipeline/deeplearning/submissions/09_main_pipeline.py --mode submit \
  --train_data data/raw/combined/train.parquet \
  --test_data data/raw/combined/test.parquet

python3 pipeline/deeplearning/submissions/09_main_pipeline.py --mode infer \
  --train_data data/raw/combined/train.parquet \
  --test_data data/raw/combined/test.parquet
```

## Deep learning — PyTorch submissions (`15`–`17`, LightGBM-style)

Same conventions as `pipeline/lgbm/submissions/`: **no CLI flags**, data from
`data/raw/combined/*.parquet`, logs under `logs/`, metrics and submission CSVs under
`outputs/`. Validation split uses **`ts_index <= 3500` for training / `> 3500` for
holdout** (same `VAL_THRESHOLD` as LightGBM). Checkpoints (optional) under
`pipeline/deeplearning/models/`.

| Script | Description | Typical outputs under `outputs/` |
|--------|-------------|----------------------------------|
| `15_dl_baseline_torch.py` | Single-seed bottleneck MLP baseline | `dl_baseline_torch_results.txt`, `dl_baseline_torch_submission.csv` |
| `16_dl_horizon_tuned_torch.py` | Per-horizon capacity / dropout tuning | `dl_horizon_tuned_torch_results.txt`, `dl_horizon_tuned_torch_submission.csv` |
| `17_dl_seed_ensemble_torch.py` | Horizon tuning + 4-seed prediction average | `dl_seed_ensemble_torch_results.txt`, `dl_seed_ensemble_torch_submission.csv` |

Examples (from **repository root**):

```bash
python3 pipeline/deeplearning/submissions/15_dl_baseline_torch.py
python3 pipeline/deeplearning/submissions/16_dl_horizon_tuned_torch.py
python3 pipeline/deeplearning/submissions/17_dl_seed_ensemble_torch.py
```

## Quick reference — where files are written

| Area | Location |
|------|----------|
| LightGBM CSVs + result text | `outputs/` at repo root |
| LightGBM logs | `logs/` at repo root |
| `09_main_pipeline.py` submission | `./submission.csv` (cwd) |
| `09_main_pipeline.py` models | `pipeline/deeplearning/models/` |
| PyTorch `15`–`17` submissions | `outputs/dl_*_torch_submission.csv` |
| PyTorch `.pt` checkpoints | `pipeline/deeplearning/models/` (e.g. `15_*_horizon_*.pt`) |

For `09_main_pipeline.py`, if a run fails, confirm `--train_data` / `--test_data`
relative to your cwd or use absolute paths. Scripts `15`–`17` always use
`data/raw/combined/train.parquet` and `test.parquet` via `pipeline/deeplearning/01_paths.py`.
