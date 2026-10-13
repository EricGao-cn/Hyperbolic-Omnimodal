# Next-steps plan: hyperbolic omni-modal representation learning

Date: 2026-10-13. Grounded in the 2026-10-13 experiment review `HYPERBOLIC_EXPERIMENT_REVIEW_20261013.md` (section numbers below refer to it). How each step maps to code is in `README.md` next to this file.

## 0. Decision in one paragraph

Commit to **one object, one eigenpair**: the normalized Lorentz Gram `M = -c XᵀJX`. Its Perron eigenvalue λ₁ gives the tuple score, and its Perron vector u gives the shared proxy `P = Xu/√λ₁`. Keep the reciprocal-kernel K-eigengap line (A2) as the strongest internal ablation and the fallback. Do not keep tuning it as the method. The record does **not** show M losing to K on the matrix itself. The 6-point Stage-1 gap and 7-point audio gap of the direct-M suite are mostly explained by the **scaled-expmap lift**. With the old direction-radius lift, M-spectral reaches 44.91 / 59.35 / 18.85 against K's 46.13 / 59.43 / 19.55 (§10). That comparison has never been run as a single-variable experiment, so it is the first thing to run.

Why commit to M rather than K. K applies the elementwise sech to the Gram matrix. It has no characterization theorem, no definiteness guarantee and no on-manifold proxy. It is exactly the "PMRL with a trick" that reviewers will attack. M has all three properties, shown below. Each was numerically verified on 4,000 random tuples (k = 2..5, c ∈ [0.1, 10], D ∈ [2, 32]); the errors are at machine precision.

## 1. The central quantity (must be airtight)

For k points on the hyperboloid with curvature −c, write `M_mn = cosh(√c·d(X_m, X_n))` and note `M_mm = 1`.

| # | Statement | Role in paper |
|---|---|---|
| P1 | M has exactly one positive eigenvalue (Sylvester). Then λ₁ = k + Σ_{i≥2}\|λ_i\| ≥ k, with equality iff X₁ = … = X_k. | Characterization of the optimum. It gives the same trace as PMRL with the opposite bound, so the naive transfer (maximize λ₁ or its softmax) provably rewards separation. |
| P2 | The spectral gap λ₁ − λ₂ ≥ k always holds, so the Perron pair is simple and its derivative is bounded. | Designs out the SVD degeneracy that PMRL and our SVD parent need custom backward passes and hard errors for (§4.6). |
| P3 | u > 0 and ⟨P, P⟩_L = −1/c, so P lies exactly on the manifold. P is a positive-weight Lorentzian centroid, so it lies in the geodesic convex hull and r(P) ≤ max_m r(X_m). | An intrinsic proxy, unlike PMRL's SVD direction or our tangent mean or SVD parents. |
| P4 (new) | `−c⟨P, X_m⟩_L = √λ₁·u_m`, hence **λ₁ = Σ_m cosh²(√c·d(P, X_m))**. | The score *is* the hyperbolic dispersion of the modalities around the proxy, so alignment and proxy are one object rather than two modules. u_m is a per-modality agreement weight. |
| P5 | `D_M = (2/√c)·asinh√(λ₁/k − 1)` reduces to the geodesic distance at k = 2. Similarly, k/λ₁ ∈ (0, 1] reduces to sech²(√c·d/2). | Anchor-free k-ary generalization of the geodesic distance and of a bounded similarity. |

Numerical note: at large √c·r, computing distances via `acosh` and then `cosh` loses up to about 47% relative accuracy. Compute λ₁ and P4 directly from Lorentz products (FP64 for the k×k `eigh`), never from distances.

## 2. Claims to freeze now (everything is triaged against these)

- **C1 (formulation).** A general, anchor-free, any-k omni-modal alignment objective defined entirely in hyperbolic space. Its optimum is characterized (P1). The naive Euclidean-spectral transfer is provably wrong (P1 corollary).
- **C2 (proxy).** The same eigenpair yields an on-manifold shared representation whose distances to the modalities define the score (P3, P4). It is well-conditioned by construction (P2).
- **C3 (empirical).** C1 and C2 are competitive with GRAM and PMRL at 512-d and also cover k = 4 and audio. The regimes where hyperbolic geometry helps are reported honestly.
- **C4 (hierarchy). Conditional.** Keep this claim only if Gate B passes. Otherwise demote it to analysis and future work.

Currently unsupported claims that must not appear: "entailment improves retrieval", "the parent learns semantic hierarchy", and "hyperbolic is better in low dimension". The evidence is in §9 (ET +0.05, cone satisfaction 0%), §12.1 (gains ≤ 0.21) and §12.2 (D64 reverses to Euclidean).

## 3. Triage of known issues

| Issue (review §) | Class | Action |
|---|---|---|
| Selection on test (§0 evidence rules) | Must fix | Select every final config on a validation split. Use 340-step screens only for ranking. |
| PMRL uses rank-local 16 negatives while ours uses global 64 (§A0) | Must fix | Run PMRL with global-64 gather (or ours with local-16) in the main table. |
| Native TVA tuple evaluation failed for M (§10) | Must fix | Repair `ret_native_tuple_candidates`. C1 is a tuple claim, so its evaluation must include tuple retrieval. |
| HyperGRAM is compared under different candidate protocols (§13) | Must fix | Re-evaluate it under our TV-only and native-tuple protocols. |
| Audio below PMRL on all hyperbolic routes (−2.9 for K, §10) | Must answer | Test symmetric outer contrast (Exp 1c). If the gap remains, report it as a measured limitation. |
| Cone entailment never satisfied (0%). The SVD parent radius sits at 0.333, consistent with its multiplier saturating at the 1/3 floor (inferred) (§9). | Must fix or drop | Gate B. |
| Euclidean backbone, only the embedding is hyperbolic; inherited ITM reranker | Tolerate and disclose | Shared with MERU and HyperGRAM. Report Stage-1 (no ITM) as the primary geometry metric. |
| Text anchor in outer InfoNCE | Tolerate if Exp 1c is neutral | One sentence. P1 is anchor-free; the anchor is in the negative sampling only. |
| SVD continuation, consensus loss, scaled-expmap, Gromov semantic hierarchy, learned TupleInfoNCE | Stop | Already measured as non-improving (§10, §11.2, §13). |
| Fully hyperbolic encoders, from-scratch pretraining, more than 4 modalities | Future work | Changes no current claim. |

## 4. Experiments, in order, with decision gates

Defaults are inherited from the K main line unless stated otherwise: VAST init, global batch 64, 1,360 updates, new-module LR 1e-4, ITM 0.1, direction-radius lift, learnable c (init 1, or 0.3 as selected for M). Screens use 340 steps on a fixed schedule. Full runs use 1,360 steps from VAST init. Seed 50 unless stated.

### Step 0. Infrastructure (no new training)
1. **Validation split.** Use the MSR-VTT, VATEX and ActivityNet validation sets if they are already available; otherwise hold out about 2k training samples. Past selections stay as they are; the split applies to every selection from now on.
2. **Screen reliability.** Compute the Spearman correlation of screen versus full-run ranks from the existing Lorentz suite (12 screens and 3 promotions, §10) and the K minimal grid (§7). If ρ is at least 0.7, trust screens for the rest of the plan. If it is lower, screen at 680 steps.
3. **Fix native tuple evaluation.**
4. **Unit tests.** Encode P1 to P5 as unit tests on the training code path, and log λ₁ − λ₂ and P's manifold error during training.

### Phase 1. Commit the core object (Gate A)
| Exp | What varies | Runs |
|---|---|---|
| 1a | Matrix only: K-eigengap vs M with −D_M, under an **identical** direction-radius lift, precision and init | 2 full |
| 1b | Score transform of the same λ₁: {−D_M (unbounded, distance-like), k/λ₁ (bounded, cosine-like), −log(λ₁/k)} × τ ∈ {0.03, 0.07, 0.15} | 9 screens, then the top 2 as full runs |
| 1c | Outer contrast: text_vs_va vs symmetric (each modality as query against the matched rest), using the best 1b score | 2 full |

The rationale for 1b is a hypothesis, not a finding. The bounded K score (46.1) and the unbounded distance scores (allpair 38.7, M scaled-expmap 40.0) split by score shape, which may matter when fine-tuning a cosine-pretrained VAST. k/λ₁ tests this with zero new parameters.

**Gate A** (validation set, then the test number reported). The best M variant passes if, against K, it reaches at least −0.3 on Top-50 R@1, at least −1.0 on Stage-1 and at least −1.0 on audio. The 3-seed std is 0.1 to 0.55 (§10).
- **Pass:** M is the method and K becomes an ablation row.
- **Fail:** Keep P1 to P5 as the analysis that explains why naive transfer fails. Use K as the score but take the proxy from M's Perron pair, and state this two-object design as a disclosed limitation. This is a weaker paper; decide it explicitly instead of drifting into it.

### Phase 2. Proxy and hierarchy (Gate B)
| Exp | What varies | Runs |
|---|---|---|
| 2a | Parent: E0 / Perron P / Text / SVD-learned, on the Gate-A winner, λ_ent ∈ {0.03, 0.1, 0.3} for Perron only | 3 + 3 full |
| 2b | Zero-cost diagnostics on the E0 model: correlation of r(P) with cross-modal dispersion; distribution of the Perron weights u_m per modality (is audio the lowest-agreement modality?); P from subsets {TV, TA, VA, TVA} versus missing-modality retrieval | Evaluation only |
| 2c | Hierarchy evaluation: ActivityNet / MSR-VTT / AudioSet category-tree Hit@1 and ancestor F1 (existing evaluation), comparing P against single-modality embeddings | Evaluation only |

**Gate B.** Keep a cone-entailment term only if Perron entailment reaches non-trivial cone satisfaction (more than about 30% on training edges) **and** improves at least one hierarchy metric beyond the seed std without costing more than 0.3 of Top-50.
- **Otherwise drop the cone loss.** The method is then score plus proxy only, and C4 is restated as "P is the intrinsic shared point and its radius and weights expose agreement", supported by 2b and 2c. Inference from the geometry: cones need a radial ordering (children outside P), while alignment drives the children onto P, so the two terms are in tension at the optimum. This is a plausible explanation for the 0% satisfaction, not a verified one.

### Phase 3. Main tables (three seeds; these carry C3)
Methods: VAST, PMRL (global-64 matched), GRAM, HyperGRAM hybrid and pure (re-evaluated under our protocols), MERU-style geodesic allpair, K-eigengap (internal), and Ours. All use the same init, budget and validation selection, with each baseline's geometry-specific axis tuned within the same screen budget.
Tasks:
- Video T↔V retrieval on the four datasets, 8 directions, reporting Stage-1 and Top-50 separately.
- **Native tuple retrieval** (T → (V, A)).
- AudioCaps and Clotho T↔A retrieval.

Estimated cost is about 24 full runs.

### Phase 4. Generality and regimes (single seed unless a claim emerges)
- **k = 4 (TVAS).** Set `drop_subtitles=false`. This is a config change for M (anchor-free) and directly demonstrates "any k". Compare against GRAM and PMRL at k = 4.
- **Width sweep.** D ∈ {64, 32, 16} for Ours, PMRL and allpair, with τ re-screened per width on validation. A positive low-dimension result becomes a headline; a negative one is a disclosed finding. Both outcomes change what the paper says, so this is worth running.
- **Second backbone (optional).** Run frozen ImageBind TVA if time remains; it is cheap because the features are cached.

### Phase 5. Analysis figures (from logged runs, no new training)
- λ₁ − λ₂ over training, ours (≥ k) versus PMRL's σ₁ − σ₂, showing well-conditioning by theorem.
- Gradient-norm stability.
- The P4 identity holding throughout training.
- Radius and agreement plots from 2b.

## 5. Budget and order

| Phase | Full-run equivalents | Can it start now? |
|---|---|---|
| 0 | ~0 (evaluation and scripts) | Yes |
| 1 | ~6 full + 9 screens (about 8 equivalents) | After Step 0.1 and 0.2 |
| 2 | ~6 full + evaluation | After Gate A |
| 3 | ~24 | After Gate B; baselines (PMRL global-64, GRAM, HyperGRAM re-evaluation) can run in parallel with Phase 1 |
| 4 | ~10 | After Phase 3 configs freeze |

Total: about 50 full-run equivalents. The baselines in Phase 3 do not depend on either gate, so they are the best use of idle GPUs while Phase 1 runs.

After Gate A, freeze the method and keep a change log. From then on, only must-fix items justify a method change.
