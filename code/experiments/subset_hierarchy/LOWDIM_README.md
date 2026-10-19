# Paired low-dimensional end-to-end fine-tuning

**Current handoff decision (2026-10-19):** transfer only source and light annotations
through GitHub; rebuild caches and continue validation on the destination host.
See `LOWDIM_SERVER_HANDOFF.md` for the current procedure. The full-cache transfer
workflow below is optional legacy guidance and is not required for this handoff.

This suite implements the approved width experiment. CPU preparation never starts
CUDA work. No experiment results exist until a run has a validated `DONE.json`.
The 512D references are newly trained controls; earlier 512D results are not reused.

## Fixed experiment

- Datasets: MSRVTT and ActivityNet Captions, using the locally available audio/video
  intersection. ActivityNet `descs_ret_test.json` is the local retrieval evaluation
  split, not a claim of access to the hidden challenge test set.
- Methods: native VAST, GRAM and PMRL; `hyp_cone`, `hyp_none`, `euc_cone`, `euc_none`.
- Dimensions: 8, 32, 128, 512. Seed: 50. Total: 56 runs.
- Train all three encoders for 10 epochs. Global batch 64, fixed across DDP world
  sizes 1/2/4/8. Shuffle once per epoch and omit the incomplete final global batch;
  do not duplicate IDs to fill it. Input sampling is deterministic per video/epoch.
- AdamW: vision LR 5e-7, other trainable parameters 2e-5, betas (0.9,0.98), decay .01,
  clip norm 2, 5% warmup followed by cosine decay. FP16 encoders, FP64 geometry.
  Nonfinite gradients fail the run; updates are never silently skipped.
- Select the highest validation six-direction macro R@1, including epoch zero;
  ties retain the earliest epoch. Evaluate the test split only after selection.
  Six directions are T↔V, T↔A, V↔A; retrieval uses no ITM reranking.

The four subset arms share VAST encoders and exactly identical seven-head weights
and radius initialization (r=1). For each dataset and pretrained model family,
extract frozen 512D features on training IDs only, using deterministic evaluation
views. Fit the uncentered second moment of normalized T/V/A features, use one shared
orthonormal basis, and take nested leading columns for D8/32/128. D512 is identity.
Project native contrastive heads, including biases, into this basis. There is no
separate head pretraining and no compression of an MSRVTT-finetuned checkpoint.

The subset objective averages the fixed nine symmetric contrastive pairs with
τ=.07. Euclidean scores use negative Euclidean distance; hyperbolic scores use
negative geodesic distance with c=1. Radius is bounded to [.21,4]. The nine directed
cone penalties have weight .1 for `*_cone`, zero for `*_none`. Euclidean cones use
angle(p,q−p) and aperture asin(.2/||p||); hyperbolic cones retain K=.1 and the existing
objective. Reflexive Euclidean edges have zero penalty. No UNCHA/HyCo changes are
mixed into this study. Native baseline objectives and ITM/DAM remain intact; PMRL
uses global candidates in DDP so its contrastive pool remains 64. Native VAST/GRAM
retain their original detached candidate-gradient convention.

## Data and resource locations

Current suite: `/data-nfs/xiaoboxia/YuxuanGao/lowdim_finetune_20261019`.
Current source: `/home/xiaobo.xia/YuxuanGao/Hyperbolic-Omnimodal`.
Local logs: `logs/lowdim_finetune_20261019/` under the selected source workspace.
Model/data resources are mapped with `--data-root`, source with `--workspace`,
output with `--suite`, and the VAST/GRAM trees with `--author-root`.

Expected cohorts before decoding: MSRVTT 7427/500/884 and ActivityNet
7580/500/3987 (train/validation/test). The actual frozen cohort is in each
`data/<dataset>/READY.json`; all methods use the same IDs, and exclusions are
explicit in `excluded.json`. Large failure rates stop preparation for inspection.

Each video has 32 uniform candidate frames, stored losslessly as uint8 after
bilinear short-edge resize to 256. The audio cache stores the complete normalized
FP32 Kaldi fbank, not encoder features. Stereo audio follows the existing channel-0
convention. MSRVTT uses 2 frames, 1 audio window, 40 text tokens; ActivityNet uses
8 frames, 2 windows, 160 tokens. BERT tokens are local cached arrays. Training uses
shared deterministic temporal selection, square crop and flip; evaluation uses
fixed views. This is an explicit common input protocol for all new runs.

## CPU workflow (safe to run without GPUs)

Use the existing project environment on this server, with user site-packages off:

```bash
cd /home/xiaobo.xia/YuxuanGao/Hyperbolic-Omnimodal
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
../PMRL/.venv/bin/python -B experiments/subset_hierarchy/lowdim_finetune.py
../PMRL/.venv/bin/python -B experiments/subset_hierarchy/lowdim_schedule.py
# Resume CPU caching if needed; READY datasets are skipped.
../PMRL/.venv/bin/python -u -B experiments/subset_hierarchy/lowdim_data.py --stage cache --workers 8
# CPU validation includes full CPU constructors and checkpoint loads, but no CUDA.
../PMRL/.venv/bin/python -u -B experiments/subset_hierarchy/lowdim_verify.py --stage all
```

`prepare` initially derives the protocol from earlier local experiment manifests;
it is not needed on a new server with the exported protocol and prepared suite.
`CPU_READY.json` requires unit/runtime checks, two-rank Gloo objective/gradient
fixtures, checkpoint/initialization audits and one complete CPU encoder forward per model, all cache array headers, frozen split
hashes and exact sampled audio-window comparisons with the original AudioMapper.
It records source hashes and data hashes. Editing source invalidates readiness.
The distributed fixtures replace encoders with differentiable small models and
fix negative indices; they do not certify complete CUDA encoder execution.

The CPU audit can run as individual stages: `tests`, `models`, `cache`, `seal`,
`export`. The cache must finish before `cache`/`seal`/`export`. Export creates a
local `handoff/lowdim_<timestamp>.tar.gz`; it uploads nothing and includes no
credentials, environments, raw data or model weights.

## Move to another server

The bundle contains `code/`, `authors/VAST`, `authors/GRAM`, protocol, task matrix,
CPU readiness evidence, `environment.json`, `resources.json` and checksums.
Transfer only after destination and transfer scope have been authorized.

1. Copy the entire prepared suite directory (especially `data/`) to a chosen output
   mount. Keep its relative paths. Copy the three initial checkpoints and constructor
   assets listed in `resources.json` under a corresponding new `--data-root`.
   Original videos/WAVs are unnecessary for training from the complete cache, but
   are needed to repeat the original-audio equivalence audit or rebuild the cache.
2. Extract the source bundle and use `code/` as the workspace; point `--author-root`
   at the sibling `authors/`. Keep the exported CPU gate in the suite.
3. Create/use a project-local virtual environment. `requirements-lowdim-finetune.txt`
   records the validated package versions. Choose a PyTorch/CUDA wheel compatible
   with the destination GPU and driver; do not modify the system driver or install
   globally. `environment.json` records the source environment. After changing the
   PyTorch version, rerun CPU validation and GPU preflight before formal training.
4. Check free disk space and current GPU use. Never stop unknown processes. CPU
   gate hashes are portable because source/data references use logical roots.
   Run the default inspection command with all four root options to verify paths.

All following examples assume `python` is from the destination project environment.
Replace the absolute example paths and GPU list with the authorized destination:

```bash
python -B experiments/subset_hierarchy/lowdim_schedule.py \
  --workspace /path/to/bundle/code --author-root /path/to/bundle/authors \
  --data-root /path/to/data --suite /path/to/data/lowdim_finetune_20261019
```

## Explicit GPU workflow (pending until requested)

The same root arguments must be included in every command on a remapped server.
These commands do launch GPU work and must only be run on authorized devices:

```bash
# Frozen train-only feature extraction and PCA (six bases).
python -u -B experiments/subset_hierarchy/lowdim_schedule.py --stage basis --execute-gpu --devices 0,1,2,3,4,5,6,7
# Profile each dataset/family at world sizes 1/2/4/8, then eligible same-card pairs.
python -u -B experiments/subset_hierarchy/lowdim_schedule.py --stage profile --execute-gpu --devices 0,1,2,3,4,5,6,7
# 28 full-encoder smokes at D8 and D512; seals GPU_READY only if all pass.
python -u -B experiments/subset_hierarchy/lowdim_schedule.py --stage preflight --execute-gpu --devices 0,1,2,3,4,5,6,7
# Launch/resume the 56-run queue. Or use --stage all for all four explicit stages.
python -u -B experiments/subset_hierarchy/lowdim_schedule.py --stage train --execute-gpu --devices 0,1,2,3,4,5,6,7
```

Profiling performs 12 optimizer updates, excludes the first two from throughput,
and records each rank's peak allocated/reserved memory. It measures whole-batch
training time, including data loading. Two jobs may share a GPU only if the sum of
measured peaks plus 3 GiB fits and measured aggregate throughput improves ≥10%.
The selected world size maximizes measured throughput per GPU, is common to the
four subset arms per dataset, and remains fixed for each resumed run. At most two
own ranks share a GPU and at most 16 ranks run globally, each with four CPU threads
and two data workers. Current free memory is checked again before launch, with
conservative reservations for processes that have not allocated memory yet.
ActivityNet tasks enter the queue first, with MSRVTT tasks filling available slots.
If no layout fits, tasks wait; the scheduler never reduces global batch size.

`layouts.json` is specific to GPU model/memory and the validated source. Repeat
profiling when hardware changes. Profiles are estimates, so a later allocation
failure marks that run failed and preserves its resumable checkpoint; it does not
switch silently to another world size. No MPS, driver, shared configuration or
unknown process changes are made. All spawned parent/rank PIDs, devices, commands
and logs are recorded. Scheduler interruption terminates only its own process groups.

`last.pt` includes model, AdamW, AMP scaler, epoch/step, per-rank RNG, history and
configuration. Checkpoints commit atomically through two bounded writer slots.
A newly selected best checkpoint is committed before `last.pt`; old best files are
pruned only after the new last state is committed. Best and last are retained.
Resume requires matching source, data, basis, world size and initial AMP scale.
`GPU_READY.json` binds six bases, 28 smokes, paired initialization and measured
layouts to the CPU gate; a placeholder file is insufficient to start training.

## Reports

```bash
python -B experiments/subset_hierarchy/lowdim_report.py
# Add --plots in an environment with matplotlib to export PDF/PNG curves.
python -B experiments/subset_hierarchy/lowdim_report.py --plots
```

Reports under `suite/reports/` include all 56 cell statuses, six-direction R@1/5/10,
macro recall, selected epoch, logged runtime/memory, matched-new-512D retention,
and seven-head radius/cone/angle/parent-compatibility and mismatched-pair controls.
Missing, failed, incomplete and stale runs are explicit; they are never substituted
with historical results. Curves include completed cells only. This is a single-seed
study, so there are no claims of across-seed confidence intervals or robustness.
