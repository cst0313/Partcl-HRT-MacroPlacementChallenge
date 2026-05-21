# HRT Winner — Macro Placement Challenge 2026 Submission

A two-stage placer combining a minimum-perturbation legalizer with a
parallel true-proxy-validated SA refinement loop.

## Key ideas

1. **Shake-apart legalization.** The `initial.plc` we receive is *almost*
   legal — usually fewer than 100 pairwise overlaps. We resolve those
   overlaps with iterative repulsive forces along the *shorter* overlap
   axis, splitting the displacement equally between movable pairs. The
   resulting placement preserves the initial global structure with
   sub-micron displacements (`≤ 0.04 µm` on ibm01) and matches the
   pre-legalization proxy cost to within fp32 tolerance. A min-disturb
   spiral legalizer and the original global greedy legalizer remain as
   fallbacks; the placer chooses whichever zero-overlap result has the
   lowest validated proxy cost.

2. **Parallel TRUE-proxy validation.** Every candidate move is verified by
   the official TILOS evaluator — no surrogate-only commits, no quiet
   regressions. We achieve high throughput by sharding 16 parallel
   workers (one process per worker, each holding its own
   `PlacementCost`) and dispatching one candidate per worker per batch.
   Each batch tests **16 different macros simultaneously** and commits
   the best improving move.

3. **TILOS evaluator monkey-patch.** Profiling `compute_proxy_cost`
   showed `PlacementCost.get_ref_node_id` performing a linear membership
   scan against `soft_macro_pin_indices` / `hard_macro_pin_indices` for
   every pin lookup. We replace it with a pre-built `dict` lookup,
   yielding a **9× speedup on small benchmarks (ibm01: 2.3 s → 0.26 s)
   and 29× on large ones (ibm10: 44 s → 1.5 s)**. The patch is bit-exact
   with the upstream implementation; we fall through to the original
   for any cache-miss.

4. **Metropolis SA with decoupled sigmas.** Four sigma buckets
   {15%, 6%, 2%, 0.8%} of canvas are kept constant across sweeps so that
   late iterations still propose exploratory moves; only the temperature
   decays.

5. **Diversified moves.** Each batch mixes five move types across its
   16 slots: Gaussian backbone (sigma bank), uniform big-jumps,
   low-density-cell targeting (attacks top-10% density), two-macro
   swaps (escapes "wrong neighbor" local minima), and reheats /
   basin-hops on wall-clock stagnation that perturb 25% of macros to
   random legal slots.

## Pipeline

```
Stage 0    Clamp soft macros into canvas                   (µs)
Stage 0.5  DREAMPlace analytical global placement          (~10–15s)
Stage 1    Cascade legalize on BOTH seeds (init + DPL):
             {init, dpl} × {shake-apart, min-disturb}      (seconds)
Stage 2    Parallel true-proxy SA refinement               (≤ 55 min default)
```

Stage 0.5 runs DREAMPlace (Lin et al.'s FFT-electrostatics analytical
placer) on the benchmark to obtain a second seed independent of
`initial.plc`. On 14/17 IBM benchmarks DPL's seed finds a noticeably
better basin (e.g. ibm04 1.31→1.17, ibm09 1.11→0.97); on ibm17/18 it
loses to the human-tuned `initial.plc`. Stage 1 runs both legalizers on
both seeds (4 candidates) and picks the lowest-proxy zero-overlap
result, so DPL is strictly additive — never worse than the original
init-only cascade.

## File layout

| File              | Role |
|-------------------|------|
| `placer.py`       | Pipeline orchestrator; `HrtPlacer.place(benchmark) -> Tensor`. |
| `legalize.py`     | Three legalizers: shake-apart, min-disturb, global greedy. |
| `pproxy_refine.py`| Parallel-proxy SA refinement. Main optimization loop. |
| `gpu_refine.py`   | Net index + incremental HPWL / density / spatial-hash state. |
| `parallel_proxy.py`| Multi-process worker pool for batch proxy-cost evaluation. |
| `_plc_patches.py` | `PlacementCost.get_ref_node_id` O(1) monkey-patch. |
| `dpl_wrapper.py`  | DREAMPlace integration: Benchmark→bookshelf→Placer.py→positions. |
| `setup_dreamplace.py` | One-shot replication script: fetch DPL sources, patch, build, verify. |

## Replicating from scratch

1. Clone this repo and set up the venv exactly as the upstream challenge
   `pyproject.toml` describes (`uv sync` is enough). CPU-only PyTorch
   is sufficient — GPUs are not used.

2. Install the extra Python dependencies DREAMPlace needs:
   ```bash
   uv pip install shapely cairocffi torch_optimizer ncg-optimizer
   ```

3. Make sure `conda`'s `base` environment has `cmake>=3.20`, `bison`,
   `flex`, `zlib`, and `boost-cpp`. On the challenge image we used:
   ```bash
   conda install -n base -y -c conda-forge cmake bison flex zlib boost-cpp
   ```

4. Build DREAMPlace from source. This fetches limbo018/DREAMPlace plus
   its five third-party submodules, applies a handful of small patches
   (NumPy 2 compat, CMake 4 policy, a quadmath.h stub for boost's
   `cpp_bin_float`), and compiles the CPU operators into
   `/tmp/dreamplace_src/DREAMPlace-master/build/`:
   ```bash
   uv run python submissions/hrt_winner/setup_dreamplace.py
   ```
   On networks without direct GitHub access, override the fetch source
   with `DPL_SOURCE_BASE`:
   ```bash
   DPL_SOURCE_BASE="https://<host>/<path>/{user}/{repo}/archive/refs/heads/{branch}.tar.gz" \
       uv run python submissions/hrt_winner/setup_dreamplace.py
   ```
   The build takes ~5 minutes on a 16-core box. A successful run ends
   with `[verify] OK:`.

5. Run the placer.
   ```bash
   uv run evaluate submissions/hrt_winner/placer.py --all     # 17 IBM benchmarks
   uv run evaluate submissions/hrt_winner/placer.py -b ibm01  # single benchmark
   ```
   `placer.py` imports `dpl_wrapper.py` lazily and falls back to the
   init-only Stage-1 cascade if the DREAMPlace build is missing, so the
   placer still works (just less well) without step 4.

## Running

The default time budget is 3300 s per benchmark (55 min); the judges'
1-hour hard limit leaves a 300 s buffer. The placer auto-sizes the
worker pool to `min(16, cpu_count - 2)` so it adapts to whatever the
judging machine provides.

## Hardware

Built and tuned on a 96-core AMD EPYC with 16 worker processes (the
judge's CPU count). The GPU is not currently used — wirelength and
density are both maintained incrementally on CPU, and the dominant cost
of refinement is parallel TILOS evaluations which are CPU-bound. Targets
the official AMD EPYC 9655P + RTX 6000 Ada machine.

## Why this beats SA / RePlAce baselines

- **Initial-placement preservation.** `initial.plc` already encodes a
  near-baseline global structure (proxy ~1.04 on ibm01). Shake-apart
  legalization preserves it; the classic SA baseline throws it away.
- **Cost-aligned optimization.** Every accepted move is verified against
  the exact proxy the judges use, so improvements on this submission's
  internal counter translate exactly to improvements on the leaderboard.
- **Bounded regression.** Best-ever tracking guarantees the returned
  placement is the lowest legal proxy cost seen across all sweeps; the
  placer **cannot** return worse than its legalized input.
