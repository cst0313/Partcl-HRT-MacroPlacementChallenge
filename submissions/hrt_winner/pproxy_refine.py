"""
Parallel-proxy SA refinement.

Architecture: each batch evaluates one candidate move for each of *K*
different macros simultaneously. With K=num_workers the whole batch takes
~one TILOS evaluator call worth of wall-clock time, so we cover ~K macros
per ~2 s instead of 1.

  * **TRUE-cost validation** — every accepted move is verified by
    `compute_proxy_cost`. We never trust the surrogate alone.
  * **Mixed candidate sigmas** — each in-batch candidate samples its
    displacement from a different sigma in {20%, 10%, 5%, 2%} of canvas,
    giving each batch both exploratory and exploitative shots without
    bookkeeping schedules.
  * **Metropolis acceptance + reheat + best-ever snap-back** — for the one
    move we commit per batch.

Note we only commit ONE move per batch — the best of the K evaluated. The
other K-1 candidates are computed from the same starting state, so the
moment we apply a move, the K-1 deltas become stale. This is a sequential
SA loop that just uses parallel evaluations to enrich each step's candidate
pool with proposals from K different macros.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost

from gpu_refine import _build_net_index, HpwlState, DensityState, SpatialHash
from parallel_proxy import ParallelProxyPool


def _locate_benchmark_dir(benchmark: Benchmark) -> Tuple[Optional[str], Optional[str]]:
    name = benchmark.name
    ibm = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if ibm.exists():
        return str(ibm), None
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
            return None, str(base)
    return None, None


def pproxy_refine(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    time_budget_s: float = 1500.0,
    num_workers: int = 16,
    seed: int = 42,
    enable_metropolis: bool = True,
    verbose: bool = False,
) -> torch.Tensor:
    t_start = time.time()
    rng = np.random.default_rng(seed)
    random.seed(seed)

    pos_np = placement.detach().cpu().numpy().astype(np.float64).copy()
    n_hard = benchmark.num_hard_macros
    sizes_np = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    half_w = sizes_np[:, 0] * 0.5
    half_h = sizes_np[:, 1] * 0.5
    fixed = benchmark.macro_fixed.cpu().numpy()
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    movable_hard = np.array([i for i in range(n_hard) if not fixed[i]], dtype=np.int64)
    if movable_hard.size == 0:
        return placement.clone()

    bdir, ng45_dir = _locate_benchmark_dir(benchmark)
    if bdir is None and ng45_dir is None:
        # Without an on-disk benchmark, we cannot spawn worker processes that
        # rebuild PlacementCost. Return the input unchanged so the caller
        # gets back a known-legal placement rather than crashing.
        if verbose:
            print(f"[pproxy_refine] no benchmark on disk for {benchmark.name!r}; "
                  "returning legalized placement unchanged", flush=True)
        return placement.clone()

    idx = _build_net_index(benchmark)
    hpwl = HpwlState(idx, pos_np, cw, ch)
    dens = DensityState(benchmark, pos_np)
    median_size = float(np.median(np.maximum(sizes_np[:n_hard, 0], sizes_np[:n_hard, 1])))
    cell_overlap = max(median_size, max(cw, ch) / 64.0)
    shash = SpatialHash(cell_overlap, n_hard, half_w[:n_hard], half_h[:n_hard],
                        pos_np[:n_hard], fixed[:n_hard])

    baseline = compute_proxy_cost(torch.from_numpy(pos_np).float(), benchmark, plc)
    cur_true = float(baseline["proxy_cost"])
    if int(baseline.get("overlap_count", 0)) > 0:
        if verbose:
            print(f"[pproxy_refine] starting has overlaps, returning unchanged", flush=True)
        return placement.clone()

    best_true = cur_true
    best_placement = pos_np.copy()
    initial_true = cur_true

    pool = ParallelProxyPool(benchmark_dir=bdir, ng45_dir=ng45_dir, num_workers=num_workers)
    if verbose:
        print(f"[pproxy_refine] start true={cur_true:.4f}  workers={num_workers}", flush=True)

    canvas_scale = max(cw, ch)
    # SA temperature in proxy-cost units. T0 ≈ 3% of starting cost.
    T0_base = max(initial_true * 0.03, 0.005)
    T_min = T0_base * 0.001

    # Sigma bank — one of these is used per batch candidate to ensure both
    # exploratory and exploitative samples in every batch. We deliberately
    # do NOT decay the upper sigmas with sweep index (only the temperature
    # decays). Keeping a constant 15% bucket lets us escape local minima
    # late in the run; the smaller buckets handle fine refinement.
    sigma_bank = [
        canvas_scale * 0.15,
        canvas_scale * 0.06,
        canvas_scale * 0.02,
        canvas_scale * 0.008,
    ]

    # Macro queue — we iterate through all movable macros, batching K of them
    # per parallel evaluation step. Reshuffled each full sweep.
    queue: List[int] = list(movable_hard)
    rng.shuffle(queue)
    queue_ptr = 0

    sweep_idx = 0   # number of complete sweeps over movable macros
    batches = 0
    total_true_calls = 0
    accepted_total = 0
    accepted_worse = 0
    moves_since_best = 0
    reheats = 0
    basin_hops = 0
    last_best_time = time.time()
    # Basin-hop / reheat fires after this fraction of the total budget
    # without a new best. Wallclock-based so it scales with benchmark size.
    stagnation_secs = max(15.0, 0.18 * time_budget_s)

    # Track per-macro recent acceptance rate to skip cold macros after a while
    # (avoids burning batches on already-optimal macros once we've cooled).
    last_improved_sweep = {m: 0 for m in movable_hard}

    # Cache of low-density cells, refreshed periodically. Used by the
    # density-aware move type to propose targets that explicitly reduce
    # top-10% mean density.
    cell_w = dens.cell_w
    cell_h = dens.cell_h
    grid_cols = dens.grid_cols
    grid_rows = dens.grid_rows
    n_cells = dens.n_cells
    low_density_cells: List[Tuple[float, float]] = []
    cells_refresh_every_batches = 8  # cheap (O(cells))

    def _refresh_low_density_cells() -> None:
        nonlocal low_density_cells
        occ = dens.occ  # per-cell occupied area
        # Bottom 30% by occupancy. Take cell centers as targets.
        if occ.size == 0:
            low_density_cells = []
            return
        k = max(8, int(n_cells * 0.30))
        # argpartition is O(N)
        idx_sorted = np.argpartition(occ, k)[:k]
        out = []
        for ci in idx_sorted:
            r = int(ci) // grid_cols
            c = int(ci) % grid_cols
            out.append(((c + 0.5) * cell_w, (r + 0.5) * cell_h))
        low_density_cells = out

    _refresh_low_density_cells()

    def _do_reheat_or_basin_hop():
        """Run a reheat (snap to best) or basin-hop (perturb 25% of macros).
        Mutates pos_np / cur_true / hpwl / dens / shash / reheats / basin_hops
        / last_best_time / moves_since_best in the enclosing scope.
        """
        nonlocal pos_np, cur_true, hpwl, dens, shash, reheats, basin_hops
        nonlocal last_best_time, moves_since_best
        reheats += 1
        do_basin = (reheats % 4 == 0)  # 4th, 8th, 12th reheat is a basin-hop;
                                       # other reheats just snap back to best.
        if do_basin:
            basin_hops += 1
            # Perturb a small slice of movable macros to random legal slots.
            # 8% — large enough to escape the local basin, small enough that
            # subsequent sweeps can recover quickly.
            pos_np = best_placement.copy()
            cur_true = best_true
            perturb_n = max(2, int(0.08 * movable_hard.size))
            chosen = rng.choice(movable_hard, size=perturb_n, replace=False)
            shash = SpatialHash(cell_overlap, n_hard,
                                half_w[:n_hard], half_h[:n_hard],
                                pos_np[:n_hard], fixed[:n_hard])
            for mm in chosen:
                cur_x = pos_np[mm, 0]; cur_y = pos_np[mm, 1]
                shash.remove(int(mm))
                placed = False
                for _t in range(64):
                    rx = float(rng.uniform(half_w[mm], cw - half_w[mm]))
                    ry = float(rng.uniform(half_h[mm], ch - half_h[mm]))
                    if not shash.has_overlap(int(mm), rx, ry,
                                             half_w[mm], half_h[mm],
                                             pos_np[:n_hard], gap=0.005):
                        pos_np[mm, 0] = rx; pos_np[mm, 1] = ry
                        shash.add(int(mm), rx, ry)
                        placed = True
                        break
                if not placed:
                    shash.add(int(mm), cur_x, cur_y)
            hpwl = HpwlState(idx, pos_np, cw, ch)
            dens = DensityState(benchmark, pos_np)
            try:
                check = compute_proxy_cost(
                    torch.from_numpy(pos_np).float(), benchmark, plc)
                if int(check.get("overlap_count", 0)) > 0:
                    raise RuntimeError("basin hop produced overlaps")
                cur_true = float(check["proxy_cost"])
            except Exception:
                pos_np = best_placement.copy()
                cur_true = best_true
                hpwl = HpwlState(idx, pos_np, cw, ch)
                dens = DensityState(benchmark, pos_np)
                shash = SpatialHash(cell_overlap, n_hard,
                                    half_w[:n_hard], half_h[:n_hard],
                                    pos_np[:n_hard], fixed[:n_hard])
            moves_since_best = 0
            last_best_time = time.time()
            if verbose:
                print(f"[pproxy_refine] BASIN-HOP #{basin_hops} "
                      f"perturbed={perturb_n} cur={cur_true:.4f}",
                      flush=True)
        else:
            pos_np = best_placement.copy()
            cur_true = best_true
            hpwl = HpwlState(idx, pos_np, cw, ch)
            dens = DensityState(benchmark, pos_np)
            shash = SpatialHash(cell_overlap, n_hard,
                                half_w[:n_hard], half_h[:n_hard],
                                pos_np[:n_hard], fixed[:n_hard])
            moves_since_best = 0
            last_best_time = time.time()
            if verbose:
                print(f"[pproxy_refine] REHEAT #{reheats}", flush=True)

    try:
        while time.time() - t_start < time_budget_s - 6.0:
            # Refresh sweep / schedule when queue runs out.
            if queue_ptr >= len(queue):
                sweep_idx += 1
                rng.shuffle(queue)
                queue_ptr = 0
                if verbose:
                    rate = total_true_calls / max(1e-6, time.time() - t_start)
                    print(
                        f"[pproxy_refine] sweep {sweep_idx} done. "
                        f"elapsed={time.time()-t_start:.1f}s "
                        f"cur={cur_true:.4f} best={best_true:.4f} "
                        f"acc={accepted_total} worse={accepted_worse} "
                        f"calls={total_true_calls} ({rate:.1f}/s)",
                        flush=True,
                    )

            # Temperature cools with sweep, sigmas stay constant — see comment
            # on sigma_bank above. Reheats bump T0 to recover exploration.
            sigmas = sigma_bank
            T0_eff = T0_base * (1.6 ** reheats) * (0.6 ** sweep_idx)
            T = max(T_min, T0_eff)

            # Collect K macros for this batch.
            K = num_workers
            batch_macros: List[int] = []
            while len(batch_macros) < K and queue_ptr < len(queue):
                batch_macros.append(queue[queue_ptr])
                queue_ptr += 1
            if not batch_macros:
                continue

            # Refresh low-density cell pool occasionally.
            if batches % cells_refresh_every_batches == 0:
                _refresh_low_density_cells()

            # Generate one candidate per slot. Each batch mixes move types by
            # slot index k:
            #
            #   k % 8 == 0,1,2,3 → Gaussian at sigma_bank[k % len(sigma_bank)]
            #   k % 8 == 4        → big jump: uniform random in canvas
            #   k % 8 == 5,6      → low-density target: pick a cold cell and
            #                        jitter inside it
            #   k % 8 == 7        → swap with a random other movable macro
            #
            # The Gaussian backbone keeps fine-refinement working; big jumps
            # and low-density targets attack density/congestion; swap escapes
            # local minima where two macros are in each other's better slots.
            placements_to_try: List[np.ndarray] = []
            # Each entry is a list of (macro_idx, nx, ny) — single-macro moves
            # have length 1, swaps have length 2.
            meta: List[List[Tuple[int, float, float]]] = []
            for k, m in enumerate(batch_macros):
                old_x, old_y = pos_np[m, 0], pos_np[m, 1]
                # Cycle sigma through bank for diversity.
                s = sigmas[k % len(sigmas)]
                slot = k % 8
                want_big_jump = (slot == 4)
                want_low_density = (slot == 5 or slot == 6) and len(low_density_cells) > 0
                want_swap = (slot == 7)

                if want_swap and movable_hard.size > 1:
                    # Pick a partner; need both to fit at each other's positions.
                    partner = None
                    for _t in range(16):
                        cand = int(rng.choice(movable_hard))
                        if cand == m:
                            continue
                        partner = cand
                        break
                    if partner is None:
                        continue
                    px, py = pos_np[partner, 0], pos_np[partner, 1]
                    # Try m at partner's spot — temporarily remove partner.
                    shash.remove(int(partner))
                    m_fits = not shash.has_overlap(int(m), float(px), float(py),
                                                    half_w[m], half_h[m],
                                                    pos_np[:n_hard], gap=0.005)
                    shash.add(int(partner), float(px), float(py))
                    if not m_fits:
                        continue
                    # Now check partner at m's old spot.
                    shash.remove(int(m))
                    p_fits = not shash.has_overlap(int(partner), float(old_x), float(old_y),
                                                    half_w[partner], half_h[partner],
                                                    pos_np[:n_hard], gap=0.005)
                    shash.add(int(m), float(old_x), float(old_y))
                    if not p_fits:
                        continue
                    # Build candidate placement with both swapped.
                    p_cand = pos_np.copy()
                    p_cand[m, 0] = float(px); p_cand[m, 1] = float(py)
                    p_cand[partner, 0] = float(old_x); p_cand[partner, 1] = float(old_y)
                    placements_to_try.append(p_cand.astype(np.float32))
                    meta.append([(int(m), float(px), float(py)),
                                 (int(partner), float(old_x), float(old_y))])
                    continue

                nx = ny = None
                # More tries → fuller batches → higher throughput.
                # Each try is just an overlap check (microseconds).
                for _try in range(24):
                    if want_big_jump:
                        cx = float(rng.uniform(half_w[m], cw - half_w[m]))
                        cy = float(rng.uniform(half_h[m], ch - half_h[m]))
                    elif want_low_density:
                        tx, ty = low_density_cells[rng.integers(0, len(low_density_cells))]
                        # Jitter inside the cell (cells are ~canvas/grid_dim ~0.5 µm).
                        cx = tx + rng.normal(0.0, max(cell_w * 0.4, 1e-3))
                        cy = ty + rng.normal(0.0, max(cell_h * 0.4, 1e-3))
                    else:
                        cx = old_x + rng.normal(0.0, s)
                        cy = old_y + rng.normal(0.0, s)
                    cx = min(max(cx, half_w[m]), cw - half_w[m])
                    cy = min(max(cy, half_h[m]), ch - half_h[m])
                    if abs(cx - old_x) < 1e-6 and abs(cy - old_y) < 1e-6:
                        continue
                    if shash.has_overlap(m, cx, cy, half_w[m], half_h[m],
                                         pos_np[:n_hard], gap=0.005):
                        continue
                    nx, ny = cx, cy
                    break
                if nx is None:
                    continue
                p = pos_np.copy()
                p[m, 0] = nx; p[m, 1] = ny
                placements_to_try.append(p.astype(np.float32))
                meta.append([(int(m), float(nx), float(ny))])

            if not placements_to_try:
                continue

            batches += 1
            results = pool.evaluate_batch(placements_to_try)
            total_true_calls += len(placements_to_try)

            # Pick the best legal candidate across all macros in the batch.
            best_idx = -1
            best_tp = float("inf")
            for k, res in enumerate(results):
                if res is None:
                    continue
                proxy, ovl, _wl, _den, _cong = res
                if ovl > 0:
                    continue
                if proxy < best_tp:
                    best_tp = proxy
                    best_idx = k

            if best_idx < 0:
                moves_since_best += 1
                continue

            moves_list = meta[best_idx]
            delta = best_tp - cur_true
            accept = delta < 0
            if not accept and enable_metropolis and delta > 0:
                p_acc = math.exp(-delta / max(T, 1e-9))
                if random.random() < p_acc:
                    accept = True
                    accepted_worse += 1

            if not accept:
                moves_since_best += 1
                continue

            # Commit all moves in the candidate (1 for single, 2 for swap).
            # For swaps: remove both from hash first, then apply each move
            # (HPWL/density), then add both back. This avoids the move-into-
            # occupied-slot issue when partner hasn't moved yet.
            for (mi, _nx, _ny) in moves_list:
                shash.remove(int(mi))
            for (mi, nx, ny) in moves_list:
                old_x, old_y = pos_np[mi, 0], pos_np[mi, 1]
                _, _nets, nminx, nmaxx, nminy, nmaxy = hpwl.delta_for_move(
                    int(mi), np.array([nx, ny]), pos_np, sizes_np)
                hpwl.apply_move(int(mi), np.array([nx, ny]), pos_np,
                                nminx, nmaxx, nminy, nmaxy)
                dens.move_macro(int(mi), (old_x, old_y), (nx, ny))
                last_improved_sweep[int(mi)] = sweep_idx
            for (mi, nx, ny) in moves_list:
                shash.add(int(mi), float(nx), float(ny))

            cur_true = best_tp
            accepted_total += 1
            if best_tp < best_true:
                best_true = best_tp
                best_placement = pos_np.copy()
                moves_since_best = 0
                last_best_time = time.time()
            else:
                moves_since_best += 1

            # Time-based stagnation check: large benchmarks may finish 0-1
            # sweeps in their budget, so a per-sweep check would never fire.
            # Check every batch.
            if (time.time() - last_best_time) > stagnation_secs and reheats < 6:
                _do_reheat_or_basin_hop()
    finally:
        pool.shutdown()

    out = torch.from_numpy(best_placement).to(placement.dtype)
    if verbose:
        rate = total_true_calls / max(1e-6, time.time() - t_start)
        print(
            f"[pproxy_refine] DONE best={best_true:.4f} "
            f"start={initial_true:.4f} delta={initial_true - best_true:+.4f} "
            f"sweeps={sweep_idx} batches={batches} reheats={reheats} "
            f"calls={total_true_calls} ({rate:.1f}/s) "
            f"acc={accepted_total} worse_acc={accepted_worse}",
            flush=True,
        )
    return out
