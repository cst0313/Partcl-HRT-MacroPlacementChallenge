"""
HRT Macro Placement Challenge — Multi-stage hybrid placer.

Pipeline (per benchmark, 1-hour budget):
  Stage 0: Bring soft macros in-bounds (the initial benchmark may have a few OOB)
  Stage 1: Legalize hard macros from initial.plc (min-displacement)
  Stage 2: Optional analytical refinement (skipped — anchor to initial dominates)
  Stage 3: Simulated annealing refinement (HPWL incremental + true-proxy calibration)
  Stage 4: Soft macro force-directed re-optimization (skipped if initial is already good)

Time budget split (default 3300s out of 3600s):
  legalize:   ≤ 60s
  SA refine:  ≤ 3000s
  soft opt:   ≤ 240s
  buffer:     300s
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost

# Make sibling modules importable
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _plc_patches  # noqa: F401, E402  -- monkey-patches PlacementCost for speed
from legalize import legalize, legalize_min_disturb, legalize_shake_apart  # noqa: E402
from pproxy_refine import pproxy_refine  # noqa: E402

try:
    from dpl_wrapper import run_dpl_global_placement, dpl_available
    _DPL_OK = dpl_available()
except Exception:
    _DPL_OK = False
    def run_dpl_global_placement(*a, **k):  # type: ignore[no-redef]
        return None


def _clamp_to_canvas(placement: torch.Tensor, benchmark: Benchmark, gap: float = 0.0) -> torch.Tensor:
    """Clamp every macro center so the full bounding box stays inside the canvas."""
    cw, ch = benchmark.canvas_width, benchmark.canvas_height
    half_w = benchmark.macro_sizes[:, 0] / 2 + gap
    half_h = benchmark.macro_sizes[:, 1] / 2 + gap
    placement = placement.clone()
    placement[:, 0] = torch.clamp(placement[:, 0], min=half_w, max=cw - half_w)
    placement[:, 1] = torch.clamp(placement[:, 1], min=half_h, max=ch - half_h)
    # Re-pin fixed macros
    fixed = benchmark.macro_fixed
    if fixed.any():
        placement[fixed] = benchmark.macro_positions[fixed]
    return placement


def _safe_cost(placement, benchmark, plc):
    try:
        return float(compute_proxy_cost(placement, benchmark, plc)["proxy_cost"])
    except Exception:
        return float("inf")


class HrtPlacer:
    """
    Multi-stage placer combining minimum-displacement legalization and SA refinement.

    Args:
        time_budget_s: Total wall-clock budget for `place()`. Default 3300s (55 min).
        sa_fraction: Fraction of budget devoted to SA refinement. Default 0.85.
        verbose: Print stage timings. Default True.
    """

    def __init__(
        self,
        time_budget_s: float = 3300.0,
        sa_fraction: float = 0.85,
        legalize_budget_s: float = 60.0,
        soft_opt_budget_s: float = 180.0,
        verbose: bool = True,
        seed: int = 42,
        num_workers: Optional[int] = None,
    ):
        self.time_budget_s = float(time_budget_s)
        self.sa_fraction = float(sa_fraction)
        self.legalize_budget_s = float(legalize_budget_s)
        self.soft_opt_budget_s = float(soft_opt_budget_s)
        self.verbose = bool(verbose)
        self.seed = int(seed)
        # Default: leave 2 cores for OS / main process, cap at 16 (judge box has 16).
        if num_workers is None:
            num_workers = min(16, max(1, (os.cpu_count() or 4) - 2))
        self.num_workers = int(num_workers)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _load_plc(self, benchmark: Benchmark):
        """Re-load PlacementCost for the benchmark (needed for true proxy)."""
        from macro_place.loader import load_benchmark_from_dir, load_benchmark
        name = benchmark.name
        # IBM benchmarks
        ibm = Path("external/MacroPlacement/Testcases/ICCAD04") / name
        if ibm.exists():
            _, plc = load_benchmark_from_dir(str(ibm))
            return plc
        # NG45 lookups
        ng45 = {
            "ariane133": "ariane133", "ariane136": "ariane136",
            "nvdla": "nvdla", "mempool_tile": "mempool_tile",
            "ariane133_ng45": "ariane133", "ariane136_ng45": "ariane136",
            "nvdla_ng45": "nvdla", "mempool_tile_ng45": "mempool_tile",
        }
        d = ng45.get(name)
        if d:
            base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
            if (base / "netlist.pb.txt").exists():
                _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
                return plc
        return None

    def _log(self, msg):
        if self.verbose:
            print(f"[HrtPlacer] {msg}", flush=True)

    # ── main entry ───────────────────────────────────────────────────────────

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_start = time.time()
        torch.manual_seed(self.seed)

        # Load PlacementCost for true proxy evaluations
        plc = self._load_plc(benchmark)
        if plc is None:
            self._log(f"WARN: could not load PlacementCost for {benchmark.name}; SA calibration disabled")

        # ── Stage 0: clamp soft macros (initial placement sometimes has OOB) ──
        placement = benchmark.macro_positions.clone()
        placement = _clamp_to_canvas(placement, benchmark, gap=0.0)
        cost_init = _safe_cost(placement, benchmark, plc) if plc else float("inf")
        self._log(f"{benchmark.name}: hard={benchmark.num_hard_macros} soft={benchmark.num_soft_macros} nets={benchmark.num_nets}")
        self._log(f"Stage 0 (clamp): proxy={cost_init:.4f}")

        # ── Stage 0.5: DREAMPlace analytical seed (optional) ──
        # DPL gives a much better global minimum for hard macros than initial.plc
        # on most benchmarks. We pass the DPL output as an ALTERNATIVE seed into
        # Stage 1's legalize cascade, then the cascade picks the best of all
        # candidates (init-seeded and dpl-seeded). On ibm17/18 DPL is worse than
        # initial.plc, but those candidates simply won't win.
        dpl_seed: Optional[torch.Tensor] = None
        if _DPL_OK:
            t0_dpl = time.time()
            try:
                new_pos = run_dpl_global_placement(benchmark, iters=1500)
                if new_pos is not None:
                    fixed = benchmark.macro_fixed.bool()
                    dpl_seed = new_pos.to(placement.device, dtype=placement.dtype)
                    # Keep fixed macros at their original positions
                    dpl_seed = torch.where(fixed.unsqueeze(1), placement, dpl_seed)
                    dpl_seed = _clamp_to_canvas(dpl_seed, benchmark, gap=0.0)
                    c_dpl = _safe_cost(dpl_seed, benchmark, plc) if plc else float("inf")
                    self._log(f"Stage 0.5 (DPL seed): {time.time()-t0_dpl:.1f}s  proxy={c_dpl:.4f} (raw, has overlaps)")
            except Exception as e:
                self._log(f"Stage 0.5 (DPL) failed: {e!r}")
                dpl_seed = None

        # ── Stage 1: legalize hard macros ──
        # Cascade three legalizers, fastest/best-quality first:
        #   1a) shake-apart: iterative pairwise repulsion. Preserves the
        #       initial layout almost exactly when overlaps are mild.
        #   1b) min-disturb: greedy spiral, only conflicting macros move.
        #   1c) global greedy: full re-placement, last-resort.
        # Run on BOTH seeds: initial.plc and (if available) DPL.
        # Pick the legalization with the lowest validated proxy cost that
        # has zero overlaps.
        t0 = time.time()
        placement_legal = None
        cost_legal = float("inf")

        candidates: list[tuple[str, torch.Tensor, float]] = []

        seeds: list[tuple[str, torch.Tensor]] = [("init", placement)]
        if dpl_seed is not None:
            seeds.append(("dpl", dpl_seed))

        for seed_tag, seed_p in seeds:
            try:
                tsa = time.time()
                p = legalize_shake_apart(
                    seed_p, benchmark,
                    gap=0.005,
                    time_budget_s=self.legalize_budget_s * 0.4,
                    max_iters=400,
                )
                p = _clamp_to_canvas(p, benchmark, gap=0.0)
                c = _safe_cost(p, benchmark, plc) if plc else float("inf")
                self._log(f"Stage 1a ({seed_tag}+shake): {time.time()-tsa:.1f}s  proxy={c:.4f}")
                candidates.append((f"{seed_tag}+shake", p, c))
            except Exception as e:
                self._log(f"Stage 1a {seed_tag}+shake failed ({e!r})")

            try:
                tmd = time.time()
                p = legalize_min_disturb(
                    seed_p, benchmark,
                    gap=0.005,
                    time_budget_s=self.legalize_budget_s * 0.3,
                )
                p = _clamp_to_canvas(p, benchmark, gap=0.0)
                c = _safe_cost(p, benchmark, plc) if plc else float("inf")
                self._log(f"Stage 1b ({seed_tag}+min_d): {time.time()-tmd:.1f}s  proxy={c:.4f}")
                candidates.append((f"{seed_tag}+min_d", p, c))
            except Exception as e:
                self._log(f"Stage 1b {seed_tag}+min_d failed ({e!r})")

        if not candidates:
            try:
                t1 = time.time()
                p = legalize(
                    placement, benchmark,
                    gap=0.05,
                    time_budget_s=self.legalize_budget_s,
                )
                p = _clamp_to_canvas(p, benchmark, gap=0.0)
                c = _safe_cost(p, benchmark, plc) if plc else float("inf")
                self._log(f"Stage 1c (global): {time.time()-t1:.1f}s  proxy={c:.4f}")
                candidates.append(("global", p, c))
            except Exception as e:
                self._log(f"Stage 1c global failed ({e!r})")

        if candidates:
            # Pick the cheapest legal candidate (cost==inf means missing plc;
            # in that case just use the first).
            candidates.sort(key=lambda x: x[2])
            label, placement_legal, cost_legal = candidates[0]
            self._log(f"Stage 1 chose '{label}'  proxy={cost_legal:.4f}")
            placement = placement_legal

        # Best-seen tracking
        best_placement = placement.clone()
        best_cost = _safe_cost(best_placement, benchmark, plc) if plc else float("inf")

        # ── Stage 2: True-proxy-validated refinement (parallel) ──
        elapsed = time.time() - t_start
        # Skip soft opt by default since initial soft positions are already great.
        soft_budget_planned = 0.0
        refine_budget = (self.time_budget_s - elapsed) - soft_budget_planned - 15.0
        refine_budget = max(refine_budget, 10.0)
        t0 = time.time()
        try:
            placement_refined = pproxy_refine(
                placement, benchmark, plc,
                time_budget_s=refine_budget,
                num_workers=self.num_workers,
                seed=self.seed,
                verbose=self.verbose,
            )
            placement_refined = _clamp_to_canvas(placement_refined, benchmark, gap=0.0)
            cost_refined = _safe_cost(placement_refined, benchmark, plc) if plc else float("inf")
            self._log(f"Stage 2 (pproxy_refine): {time.time()-t0:.1f}s  proxy={cost_refined:.4f}")
            if cost_refined < best_cost:
                best_cost, best_placement = cost_refined, placement_refined.clone()
        except Exception as e:
            self._log(f"Stage 2 FAILED: {e!r}")
            import traceback; traceback.print_exc()

        placement = best_placement.clone()

        # Stage 3 (soft opt) disabled: initial.plc already has near-optimal soft macros,
        # and aggressive moves consistently hurt density+congestion in testing.

        # Final clamp + fixed-macro restoration
        final = _clamp_to_canvas(best_placement, benchmark, gap=0.0)
        self._log(f"FINAL: {time.time()-t_start:.1f}s  best_proxy={best_cost:.4f}")
        return final
