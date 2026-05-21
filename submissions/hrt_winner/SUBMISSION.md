# HRT Winner — Macro Placement Challenge 2026 Submission

## Submission-form metrics

| Field | Value |
|-------|-------|
| **Average proxy cost (17 IBM benchmarks)** | **1.3376** |
| **Average runtime per IBM benchmark** | **100.4 s** |
| **Total wall time (17 benchmarks)** | 1706.1 s |
| **Zero-overlap rate** | 17/17 |
| **WNS on ariane133 NG45** | not measured |
| **Area on ariane133 NG45** | not measured |
| **Hardware used for the numbers above** | 16-vCPU Linux box, 32 GB RAM, CPU-only PyTorch, 8 parallel proxy workers |
| **Per-benchmark budget for the numbers above** | 120 s (smoke-test budget; submission default is 3300 s) |

These are the numbers produced by `verify_replication.py --budget 120`
(see REPLICATION.md). They are reproducible end-to-end from the .py +
.md artifacts in this folder.

## Per-benchmark breakdown (smoke test, 120 s budget)

| Benchmark | Proxy | Overlap | Wall (s) |
|-----------|-------|---------|----------|
| ibm01 | 0.9814 | 0 |  99.6 |
| ibm02 | 1.4887 | 0 |  99.8 |
| ibm03 | 1.2021 | 0 |  99.7 |
| ibm04 | 1.1477 | 0 |  99.6 |
| ibm06 | 1.4322 | 0 |  99.5 |
| ibm07 | 1.3657 | 0 | 100.0 |
| ibm08 | 1.4587 | 0 | 100.0 |
| ibm09 | 0.9418 | 0 |  99.6 |
| ibm10 | 1.3299 | 0 | 101.4 |
| ibm11 | 1.0424 | 0 | 100.2 |
| ibm12 | 1.4604 | 0 | 101.1 |
| ibm13 | 1.1382 | 0 | 100.6 |
| ibm14 | 1.4293 | 0 | 101.1 |
| ibm15 | 1.5121 | 0 | 101.0 |
| ibm16 | 1.3322 | 0 | 100.7 |
| ibm17 | 1.6969 | 0 | 101.6 |
| ibm18 | 1.7804 | 0 | 100.6 |
| **mean** | **1.3376** | **0** | **100.4** |

Baseline reference (`initial.plc → shake-apart only`, no DPL, no SA): 1.4551 →
this submission gives a **-8.1% reduction** in average proxy.

The submission's default per-benchmark budget is **3300 s** (the
challenge's 1-hour hard limit with a 300 s safety margin). The numbers
above use a 120 s smoke-test budget so they fit in a single 30-minute
verification run. The legalize-cascade improvement (DREAMPlace seed +
best-of-4 cascade vs init-only baseline) is fully captured at this
budget; the 3300 s budget would add ~3200 s more of SA refinement on
top of the same starting point.

## Method — short description (≤ 200 words)

**Two-seed best-of-four legalize cascade, then parallel TILOS-validated SA.**

Stage 0 clamps soft macros into the canvas. Stage 0.5 runs DREAMPlace
(Lin et al.'s FFT-electrostatics analytical placer) on the netlist to
produce a *second* macro seed, independent of the challenge's
`initial.plc`. Stage 1 runs **two legalizers** — shake-apart (iterative
pairwise repulsion, ≤ 0.04 µm displacement on ibm01) and min-disturb
(greedy spiral) — on **both seeds**, producing four zero-overlap
candidates, and keeps the one with the lowest TILOS proxy. On 14/17
benchmarks the DREAMPlace seed wins; on ibm08/ibm10/ibm18 init+shake
wins and the cascade picks it automatically — so the addition is
strictly never-worse.

Stage 2 is a parallel-proxy SA refinement loop. 8–16 worker processes
each hold a `PlacementCost` instance and evaluate one candidate move
per batch against the **exact TILOS proxy** — no surrogate-only commits.
Moves mix five types: Gaussian backbone (4-sigma bank), uniform
big-jumps, low-density-cell targeting, two-macro swaps, and stagnation
reheats. A pre-built dict in `_plc_patches.py` replaces
`PlacementCost.get_ref_node_id`'s O(n) pin-membership scan with O(1),
giving a 9–29× speedup on proxy evaluation.

## Method — long description

### Pipeline
```
Stage 0    Clamp soft macros into canvas                   (µs)
Stage 0.5  DREAMPlace analytical global placement          (~10–15 s)
Stage 1    Cascade legalize on BOTH seeds:
             {init, dpl} × {shake-apart, min-disturb}      (seconds)
Stage 2    Parallel TILOS-proxy SA refinement              (remaining budget)
```

### Why each stage helps
- **DREAMPlace seed.** DPL minimizes HPWL + density via FFT-accelerated
  gradient descent on an electrostatic-system relaxation. It finds a
  fundamentally different basin than `initial.plc`'s RL-derived layout,
  and dominates it on 14/17 IBM benchmarks. The losses (ibm08/10/18)
  correspond to circuits where DPL's analytical model overweights
  wirelength vs the TILOS proxy's congestion term.
- **Best-of-four legalize cascade.** Light-touch legalizers (≤ 0.04 µm
  on ibm01) preserve the seed's global structure. Running both
  shake-apart and min-disturb on both seeds and keeping the lowest
  zero-overlap proxy makes the addition strictly never-worse than
  baseline.
- **Parallel-proxy SA.** Every accepted move is verified against the
  exact TILOS evaluator — no surrogate gap. Sharding across 8–16 worker
  processes lets us evaluate 16 different macro candidates per batch
  and commit the best improving move.
- **`PlacementCost` O(1) patch.** A bit-exact monkey-patch replacing
  the upstream O(n) `get_ref_node_id` membership scan with a pre-built
  dict gives a 9× speedup on ibm01 and 29× on ibm10. The placer falls
  through to the original implementation on any cache-miss.

### Move set (Stage 2)
1. **Gaussian backbone.** σ bank {15%, 6%, 2%, 0.8%} of canvas; sigmas
   are held constant across sweeps so late iterations still explore.
2. **Uniform big-jumps** in a random canvas tile.
3. **Low-density-cell targeting** — picks a destination from the top
   10% lowest-density grid cells.
4. **Two-macro swap** — escapes "wrong neighbor" local minima.
5. **Stagnation reheat / basin-hop** — on wall-clock stagnation,
   relocates 25% of macros to random legal slots.

Each batch mixes types across its 16 worker slots; the best improving
move commits, the others are discarded.

## Replication (one paragraph)

Drop `submissions/hrt_winner/` into a fresh clone of the challenge
repo, `uv sync`, install four extra deps for DREAMPlace
(`shapely cairocffi torch_optimizer ncg-optimizer`), make sure conda
`base` has `cmake bison flex zlib boost-cpp`, then run
`uv run python submissions/hrt_winner/setup_dreamplace.py` (~5 min on
16 cores). That fetches DREAMPlace + 5 submodules from GitHub (or from
the mirror in `DPL_SOURCE_BASE` if set), applies four small build
patches encoded in the script, and compiles 28 CPU operators.
`verify_replication.py --budget 120` then runs the full pipeline on
all 17 IBM benchmarks (~30 min) and prints the table above.
Full details in `REPLICATION.md`.
