# Next-steps experiment suite

These scripts run the plan in `PLAN.md`. The plan commits to one object, the normalized Lorentz Gram M. Its Perron eigenvalue gives the score and its Perron vector gives the proxy. Two gates decide the design:
- **Gate A:** does M match the kernel score under an identical lift?
- **Gate B:** does any cone parent earn its keep?

All selection uses a held-out VAST split. Test numbers are reported only for the selected trial.

## What changed in the model code

All new options default to the existing behaviour, so the 20260929 configs reproduce unchanged. `check_geometry.py` asserts this for the default tuple loss.

| Option (model_cfg) | Values | Where |
|---|---|---|
| `hyperbolic_score_transform` | `distance` (default, −D_M), `inverse` (k/λ₁ ∈ (0,1]), `log` (−log(λ₁/k)) | `utils/lorentz_spectral.py:gram_similarity`, used by `LorentzOmnimodal` for training and native tuple retrieval |
| `hyperbolic_contrast` | `text_vs_va` (default), `symmetric` (each modality as query against the matched other two, averaged) | `LorentzOmnimodal.spectral_loss` |
| `hyperbolic_entailment_parent` on `lorentz_omnimodal` | `perron` (new: on-manifold Perron proxy of the same Gram, floored to the cone domain), `modality_edges`, `tangent_mean`, `svd_learned_radius` | `LorentzOmnimodal.perron_parent` / `entailment_loss`. Nonzero entailment weight was previously refused. |
| `pmrl_global_negatives` on `pmrl` | `false` (default, rank-local 16), `true` (1 positive + all 63 global negatives, matching the hyperbolic losses) | `PMRL.forward_ret` inter loss |

The new training diagnostics are written to `metrics.jsonl` under `geometry`:
- `perron_gap_over_k_min` (expected ≥ 1)
- `perron_identity_rel_error_max` (λ₁ = Σ cosh² d(P, Xₘ))
- `perron_manifold_error_max`
- `perron_proxy_radius_mean`
- `perron_weight_{0,1,2}_mean` (T/V/A agreement weights)
- `cone_satisfied_<edge>`
- `parent_radius_mean`
- `parent_floor_active_fraction`

Allpair keeps its 0929 behaviour and rejects transforms and contrast modes.

## Workflow

Run `python experiments/next_steps/check_geometry.py` first. It runs on CPU and needs no data.

```bash
S=/data0/xiaobo.xia/yuxuan.gao/PMRL/outputs/next_steps_20261013
OLD=/data0/xiaobo.xia/yuxuan.gao/PMRL/outputs/lorentz_omnimodal_20260929
VAST=/data0/xiaobo.xia/yuxuan.gao/PMRL/minimal_causal_assets_20260820/vast150k

# Step 0: fixed holdout (2048 clips) removed from training for every new trial
python experiments/next_steps/holdout_split.py --annotations $VAST/VASTannotations150k.accepted.json \
    --vision $VAST/videos --audio /data0/xiaobo.xia/yuxuan.gao/PMRL/datasets/vast150k/audios --out $S/holdout

# Snapshot the code, reuse the 0929 templates so media/batch/eval protocol match
python experiments/next_steps/suite.py --suite $S init --templates $OLD/templates \
    --pretrain-dir /data0/xiaobo.xia/yuxuan.gao/PMRL/models/model_weights/pretrain_vast --holdout-dir $S/holdout

# Phase 1 (Gate A): K vs M under the same lift, plus 9 transform x tau screens
python experiments/next_steps/suite.py add gate_a
python experiments/next_steps/suite.py add pmrl_screen          # baselines do not depend on the gates
python experiments/next_steps/suite.py run --gpus 0,1,2,3
python experiments/next_steps/suite.py add gate_a_promote        # top-2 screens by holdout -> 1360 steps
python experiments/next_steps/suite.py add gate_a_contrast       # best text_vs_va arm, symmetric contrast
python experiments/next_steps/suite.py run --gpus 0,1,2,3
python experiments/next_steps/suite.py rank-agreement            # screen vs full rank agreement
python experiments/next_steps/suite.py gate-a                    # -> $S/gate_a.json

# Phase 2 (Gate B): E0 = Gate-A winner; Perron (0.03/0.1/0.3), Text, SVD parents at the same settings
python experiments/next_steps/suite.py add gate_b && python experiments/next_steps/suite.py run
python experiments/next_steps/suite.py gate-b --hierarchy hierarchy.json   # {trial: ActivityNet ancestor F1}

# Phase 3/4: three seeds of the final method, kernel and matched PMRL; width sweep
python experiments/next_steps/suite.py add main && python experiments/next_steps/suite.py run
python experiments/next_steps/suite.py add lowdim --basis /data0/.../lowdim_parent_20260929/pca/basis.pt

python experiments/next_steps/suite.py summarize                 # -> $S/summary.md
```

`run` is resumable. It skips stages with `DONE.json`, records failures in `FAILED.json` without retrying them unless `--retry-failed` is passed, and `--dry-run` prints the commands. Each trial runs these stages in order:
1. `train`
2. `holdout_video` (TV-only candidates, Top-50 ITM)
3. `holdout_audio` (Stage-1 T↔A)

Full runs (1,360 steps) then add:
1. `video` (four test sets, 8 directions)
2. `audio` (AudioCaps and Clotho Stage-1)
3. `native` (T → (V, A) tuple scores; M arms only)

Screens stop at 340 updates on the 1,360-step schedule.

Gates are decided on the holdout split; `gate_a.json` and `gate_b.json` also report the test deltas, but those deltas never decide anything. Thresholds live at the top of `suite.py`:
- **Gate A:** relative to the kernel score, Top-50 ≥ −0.3, Stage-1 ≥ −1.0 and audio ≥ −1.0.
- **Gate B:** mean cone satisfaction ≥ 0.3, Top-50 ≥ −0.3 relative to E0, and a hierarchy gain larger than the seed std.

## Not covered yet

- **Native tuple evaluation.** The 0929 `native_evaluation` failure reason is not in the repository. This suite runs the stage and keeps going if it fails. Read `trials/<name>/native/stdout.log` on the first full M run before trusting C1's tuple claim.
- **Hierarchy metrics.** The ActivityNet, MSR-VTT and AudioSet category-tree scripts are not in this repository. Produce `hierarchy.json` with them and pass it to `gate-b`.
- **k = 4 (TVAS).** `LorentzOmnimodal` is fixed to T/V/A, and the subtitle branch is not lifted. This needs a model change and is not included.
- **GRAM and HyperGRAM.** They live outside this repository. Re-evaluate them under the TV-only and native protocols separately.
- **Low-dimension widths.** Only 64 and 16 are supported, because `utils/lowdim_parent.project_checkpoint` accepts only those widths.
