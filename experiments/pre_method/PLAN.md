# Pre-method experiments: test the evidence-order motivation before designing the method

Date: 2026-10-14. The commands, inputs and outputs for each experiment are in [README.md](README.md).

## Motivation being tested

Every objective tried so far is a collapse objective: PMRL, the reciprocal-kernel eigengap and the Lorentz-Gram λ₁ all reach their optimum when an instance's modalities coincide. Once they coincide, InfoNCE treats every negative as equally wrong, so curvature can only add packing capacity. That predicts a tie with PMRL at 512-d, a reversal at D64, small gains only at very low width, and radii that barely move. This is what the 20261013 review records. It is an inference, and E0 tests it.

The proposed motivation is an **evidence order**. Let C(S) be the set of instances consistent with observing modality subset S. If S ⊂ S′ then C(S′) ⊆ C(S): adding modalities can only narrow what the instance could be. Across a dataset, coarse descriptions are shared by many clips. Symmetric alignment maps this order to an equivalence. Hyperbolic space can represent it, with radius measuring how much evidence a representation commits to. Under this order the full tuple is the deepest point and each modality is an ancestor of it (the join). The earlier shared proxy sat at the meet and entailed the modalities, which is the opposite direction.

Four claims must hold before a method built on this is worth designing:

| # | Claim | Experiment | Cost |
|---|---|---|---|
| Q0 | The earlier routes ran near-flat, so they say nothing about curvature | E0 | Eval only |
| Q1 | Our benchmarks contain an evidence order with fan-out | E1a, E1b | Eval only |
| Q2 | Existing symmetric models (VAST, PMRL, our ports) erase that order | E2 | Eval only |
| Q3 | A minimal order objective recovers the order, and hyperbolic geometry does it better than a Euclidean twin | E3, E4 (E5 as a control with known truth) | Head-only training |

## E0. Near-flat audit of existing checkpoints

On the K-E0, ET, ESlearn, M-spectral and allpair checkpoints, report:
- the final c;
- √c·r per modality (median and 90th percentile);
- the fraction of direction-radius multipliers saturated at their (1/3, 3) bound;
- whether each modality log-scale sits at its ceiling (it is clamped ≤ 0 and initialized at 0);
- the distortion d_H / ‖log₀x − log₀y‖ on matched caption–clip pairs and on each caption's nearest non-matching clip.

**Reading.** If the median √c·r is below 1 and the median distortion is below 1.1, the space was used as if it were flat, and the earlier results say nothing about curvature. If the radii are large, the collapse explanation is incomplete and must be revisited before E3.

## E1. Does the data contain an evidence order?

Learned embeddings were trained with symmetric losses, so they are not a neutral measure of structure. This is a likely reason the earlier diagnostic D1 found no tree-likeness. E1 measures the order from annotations and from the strongest available matcher instead.

**E1a. Caption fan-out (annotations only).** For each clip's first caption, count the other clips that have a caption at least as similar to it as a typical paraphrase of the same clip. Similarity is TF-IDF cosine over content words, and the threshold is the median similarity between two captions of the same clip. Report:
- the fan-out distribution per dataset;
- its correlation with specificity proxies: caption length, content words, IDF mass and WordNet depth;
- with category labels, the number of clips per node at each depth.

**E1b. Ambiguity under modality subsets (ITM).** Use the ITM head of the VAST checkpoint as a near-oracle matcher. For each caption, fix a pool of 50 Stage-1 candidates that always includes the positive. Count how many candidates the ITM head accepts when the candidate exposes {v}, {a}, {v,a} and, where subtitles exist, {v,a,s}.
- **Prediction:** count(v,a) ≤ min(count(v), count(a)), and count(a) ≫ count(v) for most clips.
- **Reliability check:** if the head accepts the positive under a single-modality condition at less than half its {v,a} rate, the measure for that subset is unreliable, and the summary flags this.

**Decision.** If fan-out is about 1 almost everywhere (each caption matches essentially one clip under every subset), these benchmarks contain no hierarchy for geometry to exploit. Stop there, and either move to data with real fan-out or reposition the paper. If fan-out is heavy-tailed and shrinks as modalities are added, the order exists in our data and E1 becomes the motivating figure.

## E2. Do existing models erase the order?

For VAST, matched PMRL, K-E0, ET and M-spectral, take each model's only radial quantity: the geodesic radius for hyperbolic models, or the raw projection norm before L2 normalization for Euclidean ones. Correlate it with the E1 measures:
- caption readout against E1a fan-out (expected ρ < 0 if the order is encoded);
- v (or a) readout against log(1 + E1b count) under {v} (or {a}), with the same expected sign;
- whether the more ambiguous modality of a clip gets the smaller readout. This is only meaningful for hyperbolic models, because Euclidean norms come from different heads.

**Prediction.** ρ ≈ 0 for every model, since every objective so far is symmetric. That turns the problem statement into a measurement. If some model already shows a clear correlation, the gap is smaller than claimed and the contribution must be restated.

## E3. Head-only pilot on cached features

**Setup.** Cache VAST pooled features for t, v and a once, for the VAST training split, the 2,048-clip holdout and the test sets. Train only small heads, with the same data, budget and selection on the holdout for every cell.

**Subset encoder (identical in every cell).** Each modality is one token. A two-layer transformer runs over the tokens of subset S with absent modalities masked, mean-pools them, and a linear head produces a tangent vector. Euclidean cells use that vector directly; Lorentz cells apply expmap₀ with a learnable c and no ceiling on the scale.

**Grid.**

| Axis | Levels |
|---|---|
| Geometry | Euclidean; Lorentz |
| Objective | O1 pairwise InfoNCE (allpair); O2 collapse (PMRL in Euclidean space, M-matrix −D_M tuple InfoNCE in Lorentz space); O3 InfoNCE between every pair of subsets; O4 = O3 + subset→superset cone entailment |
| Width | 512, 64, 16 |

The Euclidean O4 cell uses Euclidean entailment cones (Ganea et al. 2018), so geometry is the only difference from Lorentz O4. That makes 24 cells at a single seed. Repeat the O3/O4 cells at the chosen width with three seeds.

**Metrics** (Stage-1 only):
1. Retrieval T↔{v}, {a}, {v,a}, and v↔a.
2. Missing-modality retrieval, read as T→{v} against T→{v,a}.
3. Order:
   - mean radius by subset size;
   - Spearman(radius, |S|);
   - how often r(h(S)) < r(h(S′)) on covering pairs;
   - join and meet cone satisfaction.
4. Data alignment: Spearman of the text radius against E1a fan-out, and of the audio radius gain r(h{v,a}) − r(h{v}) against the E1b count drop log((1+n_v)/(1+n_va)).

**Decision rules.**
- If Lorentz O4 keeps retrieval within noise of the best O1/O2 cell and clearly wins on metrics 2–4, proceed to full-model training with O4 as the method skeleton.
- If Euclidean O4 matches Lorentz O4 at every width, the contribution is the order objective, not the geometry. Decide explicitly whether the paper stays a hyperbolic paper.
- If O4 does not raise the order metrics over O3 in either geometry, the order term is not working. Revisit the subset encoder (E4) before anything else.
- The gap between O1/O2 and O3/O4 within one geometry measures how much of any gain comes from the subset objective alone.

## E4. Design ablations (extra cells of E3, Lorentz O4 only)

| Question | Variants | Prediction |
|---|---|---|
| Meet or join | `--order join` (subset ⇒ superset) vs `--order meet` (superset ⇒ subset, the earlier proxy ⇒ modality direction) | Join satisfies the cones; meet reproduces the earlier ~0% |
| Subset encoder | `--encoder fusion` vs `centroid` vs `perron` | Centroid and Perron move toward the origin and cannot represent a join; fusion can |
| Score | `--score dist` vs `gromov` | Picks the retrieval score before any full-model sweep |
| Where text sits | `--text-mode lattice` (text is one more modality) vs `query` (text contrasts with clip subsets and must entail the full clip, MERU direction) | Decides whether text belongs inside the lattice or above it |

## E5. Synthetic control with known truth

Each instance is a leaf of a complete tree (branching 4, depth 6). Each modality reveals a random subset of the levels on the leaf's root path:
- text reveals mostly shallow levels;
- video reveals many levels;
- audio reveals few levels.

Each modality also carries private noise that no other modality shares. A fixed random network maps the revealed level embeddings to features. The true depth of a subset is the number of distinct levels its modalities reveal. Running the E3 cells on this data checks two things before GPU time is spent: that O4 recovers depth when the order exists, and that the readouts mean what the plan claims. Retrieval is not the point here, because private noise is never shared across modalities.

## Dependencies

- E0, E1a and E5 need no training and can start immediately.
- E1b and E2 need one evaluation pass per checkpoint, using `extract_features.py` and `e1b_itm_subset_ambiguity.py`.
- E3 and E4 need the cached VAST features (one extraction pass over the training split, holdout and test sets).
- The method is chosen from the E3/E4 decision rules, then trained full-model with the existing infrastructure: holdout split, matched PMRL negatives and FP64 geometry.
