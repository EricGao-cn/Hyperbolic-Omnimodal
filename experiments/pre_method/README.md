# Pre-method experiments

Code for [PLAN.md](PLAN.md): tests of the evidence-order motivation that run before any method is chosen. Each script is self-contained and writes JSON plus a Markdown table. Nothing here changes the model code under `model/` or the `next_steps` suite.

## Files

| File | Experiment | Needs |
|---|---|---|
| `make_extract_config.py` | Builds the run config for extraction from a checkpoint's own config, the test sets and the holdout split | Config files only |
| `extract_features.py` | Dumps per-clip pooled features, raw projections, retrieval embeddings and lift internals | GPU, checkpoint, media |
| `e0_geometry_audit.py` | **E0**: curvature, √c·r, saturated radius multipliers, scale ceiling, hyperbolic/tangent distortion | Dumps |
| `e1a_caption_fanout.py` | **E1a**: caption fan-out from annotations (TF-IDF), its specificity correlates, optional label tree | Annotation JSON |
| `e1b_itm_subset_ambiguity.py` | **E1b**: ITM accepted counts in a fixed candidate pool under {v}, {a}, {v,a}, {v,a,s} | 1 GPU, VAST checkpoint, media |
| `e2_order_readout.py` | **E2**: correlation of each model's radial readout with E1a/E1b | Dumps, E1 outputs |
| `train_pilot.py` + `pilot/` | **E3/E4**: one head-only pilot cell on cached features, selected on the holdout | VAST dumps (or synthetic) |
| `synthetic.py` | **E5**: synthetic dumps with known per-modality revealed depth | CPU |
| `summarize_pilot.py` | Collects pilot `metrics.json` files into one table | Pilot outputs |
| `check_pre_method.py` | CPU checks of statistics, geometry, cones, centroids and every objective's backward pass | CPU, PyTorch |
| `common.py`, `runtime.py` | Shared I/O, readouts and statistics; model and loader construction outside `run.py` | — |

## Commands

Run everything from the repository root. `S` is the next-steps suite directory that already holds `holdout/` from `experiments/next_steps/holdout_split.py`. `OUT` is any output directory.

```bash
python experiments/pre_method/check_pre_method.py          # CPU, no data
```

**Extraction config.** Use the checkpoint's own config as `--base`, so the model is rebuilt exactly as it was trained. The config keeps the video test sets of `config/pmrl/minimal_hyperbolic_spectral_eval.json` and adds `vast_holdout`, plus `vast_train` with `--with-train`:

```bash
python experiments/pre_method/make_extract_config.py --base TRIAL_EVAL_CONFIG.json \
    --holdout-dir $S/holdout --out $OUT/configs/k_e0.json
python experiments/pre_method/make_extract_config.py --base PMRL_TRIAL_CONFIG.json \
    --holdout-dir $S/holdout --with-train --out $OUT/configs/vast_with_train.json
```

For VAST itself, use a matched-PMRL config as `--base` and give no `--checkpoint`, so the `pretrain_dir` in its `run_cfg` loads the VAST weights. Only the VAST dump needs the training split, so build its config with `--with-train`.

**Dumps (E0, E2, E3).** This takes the usual `run.py` flags. Shards are written per rank, and finished datasets are skipped on rerun.

```bash
torchrun --nproc_per_node 4 experiments/pre_method/extract_features.py \
    --dump-dir $OUT/features/k_e0 --config $OUT/configs/k_e0.json --checkpoint K_E0_CKPT.pt
torchrun --nproc_per_node 4 experiments/pre_method/extract_features.py \
    --dump-dir $OUT/features/vast --config $OUT/configs/vast_with_train.json
```

**E0**

```bash
python experiments/pre_method/e0_geometry_audit.py --out $OUT/e0 \
    k_e0=$OUT/features/k_e0 et=$OUT/features/et m_spectral=$OUT/features/m_spectral
```

**E1a.** Dataset names must match the dump names (`msrvtt_ret`, `vatex_ret`, …). The threshold is calibrated on same-clip captions; for a dataset with one caption per clip, pass `--threshold`. Optional `--labels name=labels.json` takes `{clip_id: [root, ..., leaf]}`.

```bash
python experiments/pre_method/e1a_caption_fanout.py --out $OUT/e1a \
    msrvtt_ret=.../msrvtt/descs_ret_test.json vatex_ret=.../vatex/descs_ret_test.json \
    didemo_ret=.../didemo/descs_ret_test.json activitynet_ret=.../activitynet/descs_ret_test.json
```

**E1b.** One process. Check `checks.single_modality_itm_reliable` in the summary before reading the counts.

```bash
torchrun --nproc_per_node 1 experiments/pre_method/e1b_itm_subset_ambiguity.py --out $OUT/e1b \
    --config $OUT/configs/vast_with_train.json --only msrvtt_ret vatex_ret didemo_ret activitynet_ret
```

**E2**

```bash
python experiments/pre_method/e2_order_readout.py --out $OUT/e2 --e1a $OUT/e1a --e1b $OUT/e1b \
    vast=$OUT/features/vast pmrl=$OUT/features/pmrl k_e0=$OUT/features/k_e0 \
    et=$OUT/features/et m_spectral=$OUT/features/m_spectral
```

**E3/E4.** One cell per call. Selection is on `vast_holdout`; tests are evaluated once with the selected state.

```bash
python experiments/pre_method/train_pilot.py --features $OUT/features/vast \
    --train vast_train --holdout vast_holdout \
    --test msrvtt_ret vatex_ret didemo_ret activitynet_ret --e1a $OUT/e1a --e1b $OUT/e1b \
    --geometry lorentz --objective order --width 64 --out $OUT/pilot/lorentz_order_w64
python experiments/pre_method/summarize_pilot.py $OUT/pilot/*/ --out $OUT/pilot/summary.md
```

E3 cells: `--geometry {euclid, lorentz}` × `--objective {allpair, collapse, subsets, order}` × `--width {512, 64, 16}`, all other flags at their defaults.

E4 cells are Lorentz `order` at the chosen width, changing one flag at a time:

| Flag | Variants |
|---|---|
| `--order` | `meet` |
| `--encoder` | `centroid`, `perron` |
| `--score` | `gromov` |
| `--text-mode` | `query` |

**E5**

```bash
python experiments/pre_method/synthetic.py --out $OUT/synthetic
python experiments/pre_method/train_pilot.py --features $OUT/synthetic --train syn_train \
    --holdout syn_holdout --test syn_test --geometry lorentz --objective order --width 16 \
    --out $OUT/pilot_syn/lorentz_order
```

Read `spearman_radius_vs_true_depth` and `spearman_audio_gain_vs_true_gain` in the synthetic `metrics.json`. Retrieval on synthetic data is low by design, because the private noise is never shared across modalities.

## What each pilot metric means

| Key in `metrics.json` (per test set) | Meaning | Order encoded if |
|---|---|---|
| `retrieval.t2v`, `t2a`, `t2va` (and reverse) | Stage-1 R@1/5/10. Objectives that never train subset embeddings score {v,a} by the mean single-modality score | — |
| `radius_by_size` | Mean radius of h(S) for \|S\| = 1, 2, 3 | Increasing |
| `spearman_radius_vs_size` | Pooled over all subsets | > 0 |
| `monotone_rate` | Rate of r(h(S)) < r(h(S′)) on covering pairs S ⊂ S′ | ≫ 0.5 |
| `cone_satisfied_join` / `_meet` | Cone satisfaction on covering pairs, in each direction | Join high |
| `spearman_text_radius_vs_fanout` | Caption radius against E1a fan-out | < 0 |
| `spearman_audio_gain_vs_itm` | r(h{v,a}) − r(h{v}) against log((1+n_v)/(1+n_va)) from E1b | > 0 |
| `spearman_radius_vs_true_depth` | Synthetic only | > 0 |
| `curvature`, `scaled_radius_median` | Lorentz only: whether the model left the near-flat regime | √c·r ≳ 1 |

## Checked here and not checked

- `check_pre_method.py` passes on CPU (PyTorch 2.14) with 10 tests: rank ties, Spearman, recall, shard merging, exp/log maps, distance agreement with `utils/lorentz_spectral.py`, the CAT(0) distance bound, Gromov-product bounds, cone containment in both spaces, centroids on the manifold and inside the hull, and the backward pass of every objective in both spaces and both text modes.
- E0, E1a and E2 ran end to end on toy dumps and toy annotations. `synthetic.py` → `train_pilot.py` ran end to end on CPU for both geometries.
- `extract_features.py` and `e1b_itm_subset_ambiguity.py` construct the real model and loaders, so they have not been run outside the server. Run one dataset with `--only msrvtt_ret` first.
