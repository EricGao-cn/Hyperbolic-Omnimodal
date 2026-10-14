# Execution on 2026-10-13

## 2026-10-14 restart on GPUs 0, 2, 3, 4

The first eight-rank K trial stopped at 71/1,360 effective updates when GPU 2
fell below the 8 GiB headroom threshold. It had no final checkpoint. Its metrics
and execution records are preserved in
`trials/a_kernel_e0/interrupted_8gpu_20261013/`; they are not combined with the
new training run.

At the user's request the queue restarts from VAST initialization on GPUs
**0,2,3,4**, using 16 examples/rank and unchanged global batch 64. Historical
four-rank measurements support allocator caps of 37 GiB for training and 31 GiB
for evaluation; the 8 GiB runtime headroom protection remains enabled. Frozen
model code is unchanged. The controller itself now also runs from the frozen
source snapshot.

Current logs: project `.tmp/next_steps_20261014_gpu0234/`. The controller PID,
command and port are in `controller_launcher.json`; live status is in
`controller_status.json`. The suite's
`execution_revisions/gpu0234_20261014/` preserves the prior protocol and records
the resource change. The sections below describe the original launch.

Base commit: `5ed0371089` (merged experiment PR). Local execution corrections
are recorded in the suite's `source_changes.diff` and `source_manifest.json`.
These corrections are included in this repository update. The running suite
continues to use its original frozen snapshot; publishing the working tree does
not change an in-flight experiment.

## Protocol and locations

- Suite: `/data0/xiaobo.xia/yuxuan.gao/PMRL/outputs/next_steps_20261013`.
- Logs: project `.tmp/next_steps_20261013/`; commands and PIDs are recorded in
  `smoke_launcher.json`, `controller_launcher.json`, and each stage's
  `execution.json`.
- Environment: existing `/home/xiaobo.xia/YuxuanGao/PMRL/.venv`; PyTorch
  2.8.0+cu128, NumPy 1.26.4. No packages installed.
- Fixed split: 84,659 training clips, 2,048 held-out clips, seed 20261013.
  Source annotations are unchanged; training/validation clip IDs are disjoint.
- Actual VAST video root is `PMRL/datasets/vast150k/videos` under the data root.
  The README's `minimal_causal_assets_20260820/vast150k/videos` does not exist.
- Global batch remains 64, using 8 ranks (8 examples/rank). No gradient
  accumulation change. Schedule remains 1,360 updates; screens stop at 340.
- Training allocator cap: 28 GiB/rank; evaluation: 27 GiB/rank. Start checks
  include GPU headroom; monitoring terminates only this stage's process group
  if any selected GPU falls below 8 GiB free. Other jobs are never terminated.
- Each new stage checks data-disk free space, retaining a 40 GiB reserve plus
  18 GiB working allowance. Save model checkpoints, not optimizer states; all
  promoted trials start again from VAST initialization as required by PLAN.
- WANDB disabled and Hugging Face offline; no external experiment uploads.

## Preflight findings and corrections

1. Historical Lorentz screen/full pairs 01, 03, 05 have Top-50 Spearman rho=1.0
   (only three promoted pairs, with selection bias). `historical_rank_agreement.json`
   records raw values. Original K-grid result files are absent locally; the
   review documents full-length trials, not recoverable matched screen/full pairs.
2. K originally used FP32 lift and spectral computation while M used FP64.
   `hyperbolic_geometry_fp64=true` now opts K into M's exact direction-radius
   lift and pair-distance path. K's default legacy behavior stays unchanged.
   The K baseline is rerun under this matched protocol.
3. Historical native-evaluation logs show CUDA batched-eigensolver failure.
   Bound native scoring to 1,024 small matrices per call across both query
   and candidate axes. A 70x4,096 GPU check agrees with CPU to 2e-15 and uses
   556 MiB peak reserved memory. Scores and model objective are unchanged.
4. Preserve allocator caps and checkpoint locking in the suite. Place logs
   and media traces in this task's workspace log directory, not older runs.
   Record failures and refuse incomplete holdout selection. Promotions finish
   before choosing the symmetric-contrast arm.
5. Use a short, uniquely created workspace TMPDIR for Python multiprocessing
   sockets. Select another free distributed port when needed. A smoke retry
   lowered the training cap from 30 to 28 GiB after startup headroom protection
   fired during a fluctuation in other jobs' memory use.
6. The pre-existing local test rejecting positive entailment weights was stale
   after this PR. It now rejects negative weights and permits positive ones.
7. The PMRL global-negative smoke exposed a noncontiguous SVD-column input to
   NCCL all-gather. Make the gathered input contiguous without changing the
   loss or gradient graph. `check_pmrl_distributed.py` compares two-rank labels,
   loss and remote feature gradients against a single-process reference.
8. Log PMRL's unscaled singular-value gap during matched-baseline training for
   PLAN's stability figure. The scalar diagnostics are detached and do not
   change the objective; recording is opt-in through `log_spectral_diagnostics`.

## Run order and evidence gates

`advance.py` requires the independent two-update smoke suite to finish first.
It runs Gate A plus PMRL screens, promotes the top two screens, completes them,
selects and runs symmetric contrast, then evaluates Gate A. On an M pass it
continues with Gate B. Main three-seed trials follow only when Gate B is decided.

The controller stops with an explicit status if a stage fails, if Gate A needs
the not-yet-implemented K-score/M-Perron fallback, or if a cone candidate passes
retrieval/cone conditions but hierarchy measurements and their measured seed
standard deviation are missing. Missing measurements are not counted as a
negative scientific result. The supplied K fallback silently substituted a
tangent mean for PLAN's Perron proxy; this controller does not execute that
substitution.

The supplied suite does not yet implement the complete external baseline table,
k=4 subtitle lift, width 32, or the full proxy/hierarchy analysis. Its completion
status is explicitly limited to the implemented main trials. Smoke metrics must
not be reported as experiment results.
