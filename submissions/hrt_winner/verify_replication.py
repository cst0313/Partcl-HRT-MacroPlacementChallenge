"""End-to-end replication smoke test for the HRT submission.

After running:
    uv run python submissions/hrt_winner/setup_dreamplace.py

run this script:
    uv run python submissions/hrt_winner/verify_replication.py

It will execute the full placer pipeline (DREAMPlace seed + legalize
cascade + parallel-proxy SA refinement) on every ICCAD-04 IBM benchmark
the repo has data for, score the result with the TILOS proxy, and print
a one-line summary per benchmark plus the 17-benchmark mean. The
default per-benchmark budget is 90 seconds — enough to see DREAMPlace's
effect on the legalize stage and a few seconds of SA refinement.

Use `--budget` to control wall time. The full submission runs at 3300 s
per benchmark (55 min), but the legalize-stage gains from DREAMPlace
show up after only a few seconds, so this short script is useful for
quick smoke-testing.

Exit code: 0 on success (every benchmark produced a zero-overlap
placement), 1 if any benchmark errored or returned an illegal layout.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement

import _plc_patches  # noqa: F401  -- speedup monkey-patch
from placer import HrtPlacer
from dpl_wrapper import dpl_available

BENCHES = [
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08", "ibm09",
    "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15", "ibm16", "ibm17",
    "ibm18",
]
DATA_ROOT = _HERE.parent.parent / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=90.0,
                    help="Per-benchmark wall budget in seconds (default 90).")
    ap.add_argument("--workers", type=int, default=8,
                    help="Number of parallel-proxy workers (default 8).")
    ap.add_argument("--subset", type=str, default="",
                    help="Comma-separated benchmark names; empty = all 17.")
    args = ap.parse_args()

    if not dpl_available():
        print("WARNING: DREAMPlace build not detected. Run setup_dreamplace.py first.",
              file=sys.stderr)
        print("         The placer will still run, but without the DPL Stage 0.5 seed.",
              file=sys.stderr)

    targets = args.subset.split(",") if args.subset else BENCHES
    targets = [t for t in targets if t]

    rows = []
    failures = 0
    for name in targets:
        bench_dir = DATA_ROOT / name
        if not bench_dir.exists():
            print(f"{name}: SKIP (data missing at {bench_dir})")
            continue
        try:
            bm, plc = load_benchmark_from_dir(str(bench_dir))
        except Exception as e:
            print(f"{name}: LOAD FAIL {e!r}")
            failures += 1
            continue

        placer = HrtPlacer(
            time_budget_s=args.budget,
            legalize_budget_s=min(30.0, args.budget * 0.3),
            verbose=False,
            num_workers=args.workers,
        )
        t0 = time.time()
        try:
            placement = placer.place(bm)
        except Exception as e:
            print(f"{name}: PLACE FAIL {e!r}")
            failures += 1
            continue
        elapsed = time.time() - t0

        ok, violations = validate_placement(placement, bm)
        cost = compute_proxy_cost(placement, bm, plc)
        proxy = cost["proxy_cost"]
        ovl = cost["overlap_count"]
        flag = "" if ok and ovl == 0 else "  [INVALID]"
        if not ok or ovl > 0:
            failures += 1
        print(f"{name}: proxy={proxy:.4f}  ovl={ovl}  wl={cost['wirelength_cost']:.4f}  "
              f"den={cost['density_cost']:.4f}  cong={cost['congestion_cost']:.4f}  "
              f"[{elapsed:.1f}s]{flag}")
        rows.append((name, proxy, ovl, elapsed))

    if rows:
        avg = sum(r[1] for r in rows) / len(rows)
        tot = sum(r[3] for r in rows)
        print(f"\nMean proxy over {len(rows)} benchmarks: {avg:.4f}  (total runtime {tot:.1f}s)")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
