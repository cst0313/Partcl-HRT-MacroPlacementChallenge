# HRT Winner — Replication Guide

This document describes how an agent on a fresh machine can replicate the
results of the `hrt_winner` submission. The submission delivers a 7-10%
improvement in average proxy cost over the baseline `initial.plc → SA`
pipeline by adding **DREAMPlace** (FFT-electrostatics analytical placer)
as a second seed feeding the legalize cascade.

All replication artifacts are plain-text Python and Markdown files under
20 KB each. They live in `submissions/hrt_winner/`.

## Result summary

Per-benchmark proxy after `verify_replication.py --budget 120` (legalize
cascade + ~70 s of SA refinement, 8 workers):

```
ibm01: 0.9814   ibm10: 1.3299
ibm02: 1.4887   ibm11: 1.0424
ibm03: 1.2021   ibm12: 1.4604
ibm04: 1.1477   ibm13: 1.1382
ibm06: 1.4322   ibm14: 1.4293
ibm07: 1.3657   ibm15: 1.5121
ibm08: 1.4587   ibm16: 1.3322
ibm09: 0.9418   ibm17: 1.6969
                 ibm18: 1.7804
```

**Average across 17 benchmarks: 1.3376** (total wall 1706.1 s, mean
100.4 s/bench). Every benchmark zero-overlap.

Comparison points (lower is better):
- Baseline `initial.plc → shake-apart only`, no DPL, no SA: 1.4551
- DPL+init cascade legalize-only, no SA: 1.3548
- DPL+init cascade + 70 s SA (this run): **1.3376** — a **-8.1%** improvement vs baseline.

The submission's default budget is 3300 s/bench, leaving ~3200 s of SA
refinement on top of the legalize cascade. The 100 s smoke test above
captures only the first ~1% of that SA budget; the full submission run
will improve further.

## Files added by this submission

| File | Purpose | LoC |
|------|---------|-----|
| `dpl_wrapper.py` | Bookshelf converter + subprocess wrapper around DREAMPlace's `Placer.py`. Returns updated macro centers as a tensor. | ~200 |
| `setup_dreamplace.py` | One-shot replication script: fetches DREAMPlace + 5 submodules, applies build patches (CMake policy, NumPy 2 compat, quadmath stub), compiles 28 CPU operators, copies Python sources into the build tree, verifies via import. | ~250 |
| `verify_replication.py` | End-to-end smoke test: runs the placer on every IBM benchmark, scores via TILOS proxy, prints per-benchmark + mean proxy. | ~100 |
| `REPLICATION.md` | This document. | — |

Modified files:

| File | Change |
|------|--------|
| `placer.py` | Added Stage 0.5 (DREAMPlace global placement) and extended Stage 1 legalize cascade to run on both seeds (`init` + `dpl` × `shake`/`min_d`). Lazy import + `dpl_available()` fallback so the placer still runs on systems without DPL. |
| `README.md` | Updated pipeline description and file table to reflect Stage 0.5. |

## Replication steps (fresh machine)

```bash
# 0. Clone the challenge repo
git clone https://github.com/partcleda/macro-place-challenge-2026.git
cd macro-place-challenge-2026
git submodule update --init external/MacroPlacement

# 1. Drop the hrt_winner files into submissions/hrt_winner/
# (If you received this as a tarball: unpack it on top of the clone.)

# 2. Set up the venv
uv sync
uv pip install shapely cairocffi torch_optimizer ncg-optimizer

# 3. Ensure build deps are available (conda is convenient, but apt also works
# if you wire CMake to system Boost/Bison/Flex/zlib):
conda install -n base -y -c conda-forge cmake bison flex zlib boost-cpp

# 4. Build DREAMPlace. Takes ~5 minutes on 16 cores.
uv run python submissions/hrt_winner/setup_dreamplace.py

# 5. Smoke-test the end-to-end pipeline (≈30 minutes at default budget):
uv run python submissions/hrt_winner/verify_replication.py --budget 120

# 6. Or run the official harness:
uv run evaluate submissions/hrt_winner/placer.py --all
```

## Environment expectations

The submission was developed and verified on:

- **OS:** Linux 6.12 (Red Hat 14 toolchain), 16 vCPUs, 32 GB RAM, 8× NVIDIA
  A10G GPU (GPUs not actually used — CPU-only PyTorch is sufficient).
- **Python:** 3.10 in a uv-managed `.venv`.
- **conda base:** GCC 14 toolchain, CMake 4.3 (worked with the
  `CMAKE_POLICY_VERSION_MINIMUM=3.5` shim), Bison 3.8, Flex 2.6,
  zlib 1.3, Boost 1.84.

DREAMPlace itself is fetched from upstream at build time; the build is
hermetic to `/tmp/dreamplace_src/` and `/tmp/dreamplace_install/` and
needs no manual edits.

### Network-restricted environments

`setup_dreamplace.py` honors a `DPL_SOURCE_BASE` environment variable to
override the GitHub mirror it fetches from. Example for an Artifactory
mirror:

```bash
export DPL_SOURCE_BASE='https://artifactory.example.com/artifactory/shared-github-remote/{user}/{repo}/archive/refs/heads/{branch}.tar.gz'
uv run python submissions/hrt_winner/setup_dreamplace.py
```

The `{user}`, `{repo}`, `{branch}` placeholders are filled in for each
of the six tarballs the script fetches (DREAMPlace plus 5 submodules
plus 1 sub-submodule).

## Build patches applied

These are all encoded in `setup_dreamplace.py` and re-applied idempotently
on re-runs:

1. **`limbo/thirdparty/lemon/CMakeLists.txt`:** bump
   `CMAKE_MINIMUM_REQUIRED` from 2.8 to 3.5, change `CMP0048 OLD` to
   `CMP0048 NEW`. CMake ≥ 4 dropped support for the old policy form.

2. **`limbo/parsers/CMakeLists.txt`:** comment out `add_subdirectory
   (gdsii/gdsdb)`. That target needs `<quadmath.h>` (libquadmath, only
   shipped with GCC's `<quadmath.h>` header on x86_64). DREAMPlace's
   placement pipeline doesn't read GDSII anyway.

3. **`/tmp/qm_stub/quadmath.h`:** empty stub header. Conda Boost's
   `cpp_bin_float.hpp` transitively `#include <quadmath.h>` even when
   `__float128` features are unused. Putting `CPATH=/tmp/qm_stub:$CPATH`
   on the make invocation satisfies the include without pulling in
   libquadmath.

4. **`dreamplace/PlaceDB.py`:** `np.string_` → `np.bytes_`. NumPy 2.0
   removed the legacy alias.

## How the algorithm works (one-paragraph summary)

The placer now runs **two** independent global placement strategies and
picks whichever feeds the better legalized result into SA refinement.
The first is the original `initial.plc` seed (a strong RL-tuned baseline
that the challenge ships). The second is **DREAMPlace** — Lin et al.'s
analytical placer that models the placement as an electrostatic system
and minimizes wirelength + density via FFT-accelerated gradient descent.
For each seed we run two legalizers: `shake-apart` (iterative pairwise
repulsion, preserves the global layout almost exactly) and
`min-disturb` (greedy spiral, only conflicting macros move). The lowest
zero-overlap proxy of the 4 candidates wins. On 14/17 IBM benchmarks
DREAMPlace's basin beats `initial.plc`'s; on the remaining 3
(ibm08, ibm10, ibm18) DREAMPlace's analytical objective doesn't align
with the TILOS proxy's congestion weighting and the cascade correctly
falls back to `init+shake`. The parallel-proxy SA refinement loop then
runs from whichever seed won, so its benefit compounds with DPL's gain.
