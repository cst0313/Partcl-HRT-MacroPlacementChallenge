# Partcl HRT — Macro Placement Challenge 2026 Submission

Submission for the [Partcl Macro Placement Challenge 2026](https://github.com/partcleda/macro-place-challenge-2026).

| Metric | Value |
|---|---|
| Average proxy cost (17 IBM benchmarks) | **1.3376** |
| Zero-overlap rate | 17/17 |
| Improvement vs `initial.plc → shake-apart` baseline | **−8.1%** |
| Average runtime / benchmark (smoke test, 120 s budget) | 100.4 s |

Full per-benchmark numbers and method discussion in
[`submissions/hrt_winner/SUBMISSION.md`](submissions/hrt_winner/SUBMISSION.md).

## Replication (judges)

The submission lives in [`submissions/hrt_winner/`](submissions/hrt_winner/)
and depends on the upstream challenge repo's `macro_place` package plus
DREAMPlace. End-to-end replication on a fresh Linux machine:

```bash
# 1. Clone the upstream challenge repo and init the data submodule
git clone https://github.com/partcleda/macro-place-challenge-2026.git
cd macro-place-challenge-2026
git submodule update --init external/MacroPlacement

# 2. Drop this submission into submissions/hrt_winner/
git clone https://github.com/cst0313/Partcl-HRT-MacroPlacementChallenge.git /tmp/hrt
cp -r /tmp/hrt/submissions/hrt_winner submissions/hrt_winner

# 3. Set up the venv + DREAMPlace extras
uv sync
uv pip install shapely cairocffi torch_optimizer ncg-optimizer

# 4. Make sure conda 'base' has the build toolchain
conda install -n base -y -c conda-forge cmake bison flex zlib boost-cpp

# 5. Build DREAMPlace (~5 min on 16 cores)
uv run python submissions/hrt_winner/setup_dreamplace.py

# 6. Smoke-test the full pipeline on all 17 IBM benchmarks (~30 min)
uv run python submissions/hrt_winner/verify_replication.py --budget 120

# 7. Or run via the official harness
uv run evaluate submissions/hrt_winner/placer.py --all
```

Detailed replication notes, environment expectations, build patches, and
the network-restricted (`DPL_SOURCE_BASE`) override are in
[`submissions/hrt_winner/REPLICATION.md`](submissions/hrt_winner/REPLICATION.md).

## Layout

```
.
├── LICENSE                          # PolyForm Noncommercial 1.0.0
├── README.md                        # this file
└── submissions/
    └── hrt_winner/
        ├── README.md                # method overview
        ├── SUBMISSION.md            # submission-form metrics + per-benchmark table
        ├── REPLICATION.md           # full step-by-step replication guide
        ├── placer.py                # pipeline orchestrator (HrtPlacer.place)
        ├── legalize.py              # shake-apart / min-disturb / greedy legalizers
        ├── pproxy_refine.py         # parallel-proxy SA refinement loop
        ├── gpu_refine.py            # incremental HPWL / density / spatial-hash
        ├── parallel_proxy.py        # multi-process worker pool
        ├── _plc_patches.py          # O(1) PlacementCost.get_ref_node_id patch
        ├── dpl_wrapper.py           # DREAMPlace integration
        ├── setup_dreamplace.py      # fetch + patch + build DREAMPlace
        └── verify_replication.py    # 17-benchmark smoke test
```

## License

[PolyForm Noncommercial 1.0.0](LICENSE). Research, personal, and other
noncommercial use permitted; commercial use prohibited.
