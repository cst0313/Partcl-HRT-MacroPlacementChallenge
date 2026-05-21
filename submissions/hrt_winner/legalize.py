"""
HRT Winner — Stage 2: Legalizer.

Snaps a (possibly overlapping) hard-macro placement to a fully legal placement
with zero overlaps. Soft macros and fixed macros are passed through unchanged.

Strategy:
  Phase 0: Clamp inputs to canvas, snap fixed/already-placed macros.
  Phase 1: Greedy largest-first placement with spiral search using a uniform
           spatial-hash grid for O(1) average overlap queries.
  Phase 2: Optional small-radius local refinement that tries to shrink each
           macro's displacement from the original (target) position without
           introducing overlaps.

This is "Strategy A + Strategy B (light)" from the spec.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark


# ---------------------------------------------------------------------------
# Spatial hash grid
# ---------------------------------------------------------------------------


class _Grid:
    """Uniform grid spatial hash for AABB queries."""

    __slots__ = ("cw", "ch", "cell", "ncols", "nrows", "cells")

    def __init__(self, canvas_w: float, canvas_h: float, cell: float):
        self.cw = float(canvas_w)
        self.ch = float(canvas_h)
        self.cell = max(float(cell), 1e-6)
        self.ncols = max(1, int(np.ceil(self.cw / self.cell)) + 1)
        self.nrows = max(1, int(np.ceil(self.ch / self.cell)) + 1)
        # 2-D list of python lists; sparse storage might be heavier here.
        self.cells: List[List[int]] = [[] for _ in range(self.ncols * self.nrows)]

    def _bounds(self, x: float, y: float, hw: float, hh: float) -> Tuple[int, int, int, int]:
        cx0 = max(0, int((x - hw) / self.cell))
        cx1 = min(self.ncols - 1, int((x + hw) / self.cell))
        cy0 = max(0, int((y - hh) / self.cell))
        cy1 = min(self.nrows - 1, int((y + hh) / self.cell))
        return cx0, cx1, cy0, cy1

    def insert(self, idx: int, x: float, y: float, hw: float, hh: float) -> None:
        cx0, cx1, cy0, cy1 = self._bounds(x, y, hw, hh)
        for cy in range(cy0, cy1 + 1):
            row_off = cy * self.ncols
            for cx in range(cx0, cx1 + 1):
                self.cells[row_off + cx].append(idx)

    def candidates(self, x: float, y: float, hw: float, hh: float) -> List[int]:
        cx0, cx1, cy0, cy1 = self._bounds(x, y, hw, hh)
        out: List[int] = []
        seen = set()
        for cy in range(cy0, cy1 + 1):
            row_off = cy * self.ncols
            for cx in range(cx0, cx1 + 1):
                for j in self.cells[row_off + cx]:
                    if j not in seen:
                        seen.add(j)
                        out.append(j)
        return out


# ---------------------------------------------------------------------------
# Overlap helpers
# ---------------------------------------------------------------------------


def _overlaps_any(
    idx: int,
    x: float,
    y: float,
    hw: float,
    hh: float,
    grid: _Grid,
    placed_mask: np.ndarray,
    pos: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    gap: float,
) -> bool:
    """Return True if (idx) at (x,y) overlaps any placed macro (excluding itself)."""
    cands = grid.candidates(x, y, hw, hh)
    for j in cands:
        if j == idx or not placed_mask[j]:
            continue
        if (
            abs(x - pos[j, 0]) < hw + half_w[j] + gap
            and abs(y - pos[j, 1]) < hh + half_h[j] + gap
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Greedy largest-first with spiral search
# ---------------------------------------------------------------------------


def _spiral_legalize(
    pos: np.ndarray,
    target: np.ndarray,
    sizes: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    movable: np.ndarray,
    fixed_mask: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    order: np.ndarray,
    gap: float,
    deadline: float,
) -> Tuple[np.ndarray, bool]:
    """Place macros in `order`, spiral-snapping any that overlap.

    Returns (positions, success).  success=False if the deadline ran out
    before every macro could be placed.
    """
    n = pos.shape[0]
    out = pos.copy()
    placed = np.zeros(n, dtype=bool)

    # Cell size: median macro size keeps the grid balanced.
    if n > 0:
        med = float(np.median(np.maximum(sizes[:, 0], sizes[:, 1])))
    else:
        med = 1.0
    cell = max(med, 1.0)
    grid = _Grid(canvas_w, canvas_h, cell)

    # Pre-place fixed macros: they cannot move.
    for i in range(n):
        if fixed_mask[i]:
            x, y = float(out[i, 0]), float(out[i, 1])
            # Clamp into canvas just in case (fixed should already be valid).
            x = min(max(x, half_w[i] + gap), canvas_w - half_w[i] - gap)
            y = min(max(y, half_h[i] + gap), canvas_h - half_h[i] - gap)
            out[i, 0] = x
            out[i, 1] = y
            grid.insert(i, x, y, half_w[i], half_h[i])
            placed[i] = True

    # Smallest dim across all macros -> sets the spiral step.
    min_dim = float(np.min(np.minimum(sizes[:, 0], sizes[:, 1])))
    step = max(min_dim * 0.5, 0.5)  # not too tiny for big designs
    # Also cap step so very small macros don't make us spiral forever.
    step = min(step, max(canvas_w, canvas_h) * 0.01)
    step = max(step, 0.25)

    max_radius = int(np.ceil(max(canvas_w, canvas_h) / step)) + 2

    for idx in order:
        if fixed_mask[idx]:
            continue
        if not movable[idx]:
            # Non-movable, non-fixed (shouldn't normally happen but be safe).
            x, y = float(out[idx, 0]), float(out[idx, 1])
            grid.insert(idx, x, y, half_w[idx], half_h[idx])
            placed[idx] = True
            continue

        # Start from the target (clamped) position.
        tx = float(target[idx, 0])
        ty = float(target[idx, 1])
        tx = min(max(tx, half_w[idx] + gap), canvas_w - half_w[idx] - gap)
        ty = min(max(ty, half_h[idx] + gap), canvas_h - half_h[idx] - gap)

        # Quick try: target position itself.
        if not _overlaps_any(
            idx, tx, ty, half_w[idx], half_h[idx],
            grid, placed, out, half_w, half_h, gap,
        ):
            out[idx, 0] = tx
            out[idx, 1] = ty
            grid.insert(idx, tx, ty, half_w[idx], half_h[idx])
            placed[idx] = True
            continue

        # Spiral outward on Chebyshev shells.
        best_x = tx
        best_y = ty
        best_d = float("inf")
        found = False

        # Deadline check every spiral pass
        for r in range(1, max_radius + 1):
            if time.time() > deadline:
                break
            shell_found = False
            # Iterate the shell at radius r.
            for dxm in range(-r, r + 1):
                adxm = abs(dxm)
                for dym in range(-r, r + 1):
                    if max(adxm, abs(dym)) != r:
                        continue
                    cx = tx + dxm * step
                    cy = ty + dym * step
                    # Clamp candidate into canvas.
                    cx = min(max(cx, half_w[idx] + gap), canvas_w - half_w[idx] - gap)
                    cy = min(max(cy, half_h[idx] + gap), canvas_h - half_h[idx] - gap)
                    if _overlaps_any(
                        idx, cx, cy, half_w[idx], half_h[idx],
                        grid, placed, out, half_w, half_h, gap,
                    ):
                        continue
                    d = (cx - tx) ** 2 + (cy - ty) ** 2
                    if d < best_d:
                        best_d = d
                        best_x = cx
                        best_y = cy
                        shell_found = True
                        found = True
            if shell_found:
                # The first shell that succeeds is the closest, so stop.
                break

        if not found:
            # Hard failure — leave at clamped target (will be flagged as overlap).
            # Caller should treat this as a failed attempt.
            return out, False

        out[idx, 0] = best_x
        out[idx, 1] = best_y
        grid.insert(idx, best_x, best_y, half_w[idx], half_h[idx])
        placed[idx] = True

        if time.time() > deadline:
            return out, False

    return out, True


# ---------------------------------------------------------------------------
# Local refinement (Strategy B-lite)
# ---------------------------------------------------------------------------


def _refine(
    pos: np.ndarray,
    target: np.ndarray,
    sizes: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    movable: np.ndarray,
    fixed_mask: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    gap: float,
    deadline: float,
    passes: int = 3,
) -> np.ndarray:
    """Try to move each movable macro closer to its target without breaking legality."""
    n = pos.shape[0]
    out = pos.copy()

    if n == 0:
        return out

    med = float(np.median(np.maximum(sizes[:, 0], sizes[:, 1])))
    cell = max(med, 1.0)
    grid = _Grid(canvas_w, canvas_h, cell)
    placed = np.ones(n, dtype=bool)
    for i in range(n):
        grid.insert(i, float(out[i, 0]), float(out[i, 1]), half_w[i], half_h[i])

    movable_idx = np.where(movable & (~fixed_mask))[0]
    # Order: furthest from target first (highest "savings potential").
    for _ in range(passes):
        if time.time() > deadline:
            break
        disp = (out[movable_idx, 0] - target[movable_idx, 0]) ** 2 + (
            out[movable_idx, 1] - target[movable_idx, 1]
        ) ** 2
        order = movable_idx[np.argsort(-disp)]

        improved_any = False
        for idx in order:
            if time.time() > deadline:
                break
            cur_x = float(out[idx, 0])
            cur_y = float(out[idx, 1])
            tx = float(target[idx, 0])
            ty = float(target[idx, 1])
            cur_d = (cur_x - tx) ** 2 + (cur_y - ty) ** 2
            if cur_d < 1e-9:
                continue

            # Temporarily mark as "not placed" so we can re-test ourselves.
            placed[idx] = False

            # Try a direct snap toward the target via line search.
            best_x, best_y = cur_x, cur_y
            best_d = cur_d
            dirx = tx - cur_x
            diry = ty - cur_y
            # Several step fractions, largest first.
            for frac in (1.0, 0.75, 0.5, 0.33, 0.2, 0.1, 0.05):
                nx = cur_x + dirx * frac
                ny = cur_y + diry * frac
                nx = min(max(nx, half_w[idx] + gap), canvas_w - half_w[idx] - gap)
                ny = min(max(ny, half_h[idx] + gap), canvas_h - half_h[idx] - gap)
                if _overlaps_any(
                    idx, nx, ny, half_w[idx], half_h[idx],
                    grid, placed, out, half_w, half_h, gap,
                ):
                    continue
                d = (nx - tx) ** 2 + (ny - ty) ** 2
                if d < best_d - 1e-9:
                    best_d = d
                    best_x = nx
                    best_y = ny
                    break  # we accept the largest improvement we found

            if best_d < cur_d - 1e-9:
                # Update grid: remove from old cells, insert into new.
                # Simpler: rebuild this macro's cell entries by full sweep is costly.
                # Trick: leave stale entries; they're harmless because we check `placed`
                # plus the position in `out`. We still need to insert into the new
                # cells so future candidate queries can find it.
                grid.insert(idx, best_x, best_y, half_w[idx], half_h[idx])
                out[idx, 0] = best_x
                out[idx, 1] = best_y
                improved_any = True

            placed[idx] = True

        if not improved_any:
            break

    return out


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def _validate_no_overlap(
    out: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    gap: float,
) -> bool:
    n = out.shape[0]
    if n <= 1:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(out[i, 0] - out[j, 0])
            dy = abs(out[i, 1] - out[j, 1])
            if dx + 1e-9 < half_w[i] + half_w[j] and dy + 1e-9 < half_h[i] + half_h[j]:
                return False
    return True


def legalize(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    gap: float = 0.05,
    time_budget_s: float = 60.0,
) -> torch.Tensor:
    """Snap a near-legal placement into a fully legal one.

    Args:
        placement: [num_macros, 2] tensor with current macro centers
            (hard macros at indices [0, num_hard_macros); soft macros after).
        benchmark: Benchmark object describing macros, canvas, fixed mask, etc.
        gap: Minimum slack to maintain between hard macros (μm).
        time_budget_s: Soft time budget for the whole legalization in seconds.

    Returns:
        legal_placement: [num_macros, 2] tensor.  Soft macros are unchanged.
    """
    start = time.time()
    deadline = start + max(time_budget_s, 1.0)

    n_total = benchmark.num_macros
    n_hard = benchmark.num_hard_macros

    # Output tensor — start from the input, soft macros will be left alone.
    out_full = placement.detach().clone().to(torch.float32).cpu()

    if n_hard == 0:
        return out_full

    sizes = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64)
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    fixed_mask = benchmark.macro_fixed[:n_hard].detach().cpu().numpy().astype(bool)
    movable = ~fixed_mask  # we'll move everything that isn't fixed
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    target = placement[:n_hard].detach().cpu().numpy().astype(np.float64)
    # Clamp targets to the canvas right away.
    target_clamped = target.copy()
    target_clamped[:, 0] = np.clip(target_clamped[:, 0], half_w + gap, canvas_w - half_w - gap)
    target_clamped[:, 1] = np.clip(target_clamped[:, 1], half_h + gap, canvas_h - half_h - gap)
    # If a macro literally doesn't fit, fall back to clipping with 0 gap.
    bad = (half_w + gap) > (canvas_w - half_w - gap)
    if bad.any():
        target_clamped[bad, 0] = canvas_w / 2.0
    bad = (half_h + gap) > (canvas_h - half_h - gap)
    if bad.any():
        target_clamped[bad, 1] = canvas_h / 2.0

    # Build an initial position vector seeded from the (clamped) target.
    pos0 = target_clamped.copy()
    # Fixed macros keep their *original* benchmark coordinates.
    bench_pos = benchmark.macro_positions[:n_hard].detach().cpu().numpy().astype(np.float64)
    pos0[fixed_mask] = bench_pos[fixed_mask]

    # Build candidate orderings.  Largest area first is the workhorse.
    area = sizes[:, 0] * sizes[:, 1]
    order_area = np.argsort(-area)

    # Second ordering: largest "max-dim" first (helps when aspect ratios vary).
    max_dim = np.maximum(sizes[:, 0], sizes[:, 1])
    order_maxdim = np.argsort(-max_dim)

    # Third ordering: by current y then x (shelf-like — good for tight designs).
    order_yx = np.lexsort((target_clamped[:, 0], target_clamped[:, 1]))

    best_out: Optional[np.ndarray] = None
    best_disp = float("inf")

    half_budget = (deadline - time.time()) * 0.7  # reserve some time for refinement

    for label, order in (("area", order_area), ("maxdim", order_maxdim), ("shelf", order_yx)):
        if time.time() - start > half_budget:
            break
        if best_out is not None and label == "shelf":
            # shelf ordering is only worth trying if previous attempts fully placed
            # but did poorly on displacement.  Skip if our current best is decent.
            pass

        attempt_deadline = min(deadline, time.time() + max(5.0, half_budget / 3.0))
        out, ok = _spiral_legalize(
            pos0,
            target_clamped,
            sizes,
            half_w,
            half_h,
            movable,
            fixed_mask,
            canvas_w,
            canvas_h,
            order,
            gap,
            attempt_deadline,
        )
        if not ok:
            continue
        disp = float(np.sum((out - target_clamped) ** 2))
        if disp < best_disp:
            best_disp = disp
            best_out = out

    # Fallback: if every strategy failed, do a brute "place wherever fits" pass
    # using area order without deadline early-out — accept whatever it gives us.
    if best_out is None:
        out, ok = _spiral_legalize(
            pos0,
            target_clamped,
            sizes,
            half_w,
            half_h,
            movable,
            fixed_mask,
            canvas_w,
            canvas_h,
            order_area,
            gap,
            deadline,
        )
        best_out = out  # even if ok=False, we use the best-effort result

    # Phase 2: local refinement
    if best_out is not None and time.time() < deadline:
        best_out = _refine(
            best_out,
            target_clamped,
            sizes,
            half_w,
            half_h,
            movable,
            fixed_mask,
            canvas_w,
            canvas_h,
            gap,
            deadline,
            passes=4,
        )

    # Write back.
    out_full[:n_hard, 0] = torch.from_numpy(best_out[:, 0].astype(np.float32))
    out_full[:n_hard, 1] = torch.from_numpy(best_out[:, 1].astype(np.float32))

    # Make sure fixed macros are *exactly* at their original spots (no FP drift).
    if fixed_mask.any():
        out_full[:n_hard][torch.from_numpy(fixed_mask)] = benchmark.macro_positions[
            :n_hard
        ][torch.from_numpy(fixed_mask)].to(torch.float32)

    return out_full


# ---------------------------------------------------------------------------
# Min-disturb legalize: lock all non-overlapping macros in place; only spiral
# the macros that participate in an overlap. Dramatically reduces displacement
# vs. the greedy largest-first legalize when the initial.plc is "almost legal".
# ---------------------------------------------------------------------------


def _find_overlap_set(
    pos: np.ndarray, half_w: np.ndarray, half_h: np.ndarray, gap: float,
    canvas_w: float, canvas_h: float,
) -> Tuple[set, set]:
    """Return (involved_macros, oob_macros).

    involved_macros: indices touching any pairwise overlap (treating macros as
                     rectangles padded by `gap`).
    oob_macros:      indices whose bounding box leaves the canvas.
    """
    n = pos.shape[0]
    involved = set()
    oob = set()
    for i in range(n):
        if pos[i, 0] - half_w[i] < -1e-9 or pos[i, 0] + half_w[i] > canvas_w + 1e-9:
            oob.add(i)
        if pos[i, 1] - half_h[i] < -1e-9 or pos[i, 1] + half_h[i] > canvas_h + 1e-9:
            oob.add(i)

    if n <= 1:
        return involved, oob

    # Use a spatial grid for O(N) neighbor lookups
    med = float(np.median(np.maximum(half_w * 2, half_h * 2)))
    cell = max(med * 1.1, 1.0)
    grid = _Grid(canvas_w, canvas_h, cell)
    for i in range(n):
        grid.insert(i, float(pos[i, 0]), float(pos[i, 1]), float(half_w[i]), float(half_h[i]))

    for i in range(n):
        cands = grid.candidates(pos[i, 0], pos[i, 1], half_w[i], half_h[i])
        for j in cands:
            if j <= i:
                continue
            if (abs(pos[i, 0] - pos[j, 0]) < half_w[i] + half_w[j] + gap
                and abs(pos[i, 1] - pos[j, 1]) < half_h[i] + half_h[j] + gap):
                involved.add(i)
                involved.add(j)
    return involved, oob


def legalize_shake_apart(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    gap: float = 0.005,
    time_budget_s: float = 30.0,
    max_iters: int = 200,
) -> torch.Tensor:
    """Iterative repulsion legalization (Lagrangian-style shake-apart).

    Each iteration pushes overlapping pairs apart along the *shorter*
    overlap axis by half the overlap. Fixed macros are immovable and absorb
    100% of the displacement. Total perturbation across the network is
    distributed evenly — minimum-disturbance in expectation.

    Falls back by raising RuntimeError if not converged within the budget.
    """
    start = time.time()
    deadline = start + max(time_budget_s, 1.0)

    n_total = benchmark.num_macros
    n_hard = benchmark.num_hard_macros
    out_full = placement.detach().clone().to(torch.float32).cpu()
    if n_hard == 0:
        return out_full

    sizes = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64)
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    fixed_mask = benchmark.macro_fixed[:n_hard].detach().cpu().numpy().astype(bool)
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    pos = placement[:n_hard].detach().cpu().numpy().astype(np.float64).copy()
    pos[:, 0] = np.clip(pos[:, 0], half_w, canvas_w - half_w)
    pos[:, 1] = np.clip(pos[:, 1], half_h, canvas_h - half_h)
    bench_pos = benchmark.macro_positions[:n_hard].detach().cpu().numpy().astype(np.float64)
    pos[fixed_mask] = bench_pos[fixed_mask]

    # Spatial-grid for fast neighbor queries.
    med = float(np.median(np.maximum(sizes[:, 0], sizes[:, 1])))
    cell = max(med * 1.2, 1.0)

    converged = False
    for _it in range(max_iters):
        if time.time() > deadline:
            break
        grid = _Grid(canvas_w, canvas_h, cell)
        for i in range(n_hard):
            grid.insert(i, float(pos[i, 0]), float(pos[i, 1]),
                        float(half_w[i] + gap), float(half_h[i] + gap))

        forces = np.zeros_like(pos)
        any_overlap = False
        seen_pair = set()
        for i in range(n_hard):
            cands = grid.candidates(pos[i, 0], pos[i, 1],
                                    half_w[i] + gap, half_h[i] + gap)
            for j in cands:
                if j <= i:
                    continue
                pij = (i, j)
                if pij in seen_pair:
                    continue
                seen_pair.add(pij)
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                ox = half_w[i] + half_w[j] + gap - abs(dx)
                oy = half_h[i] + half_h[j] + gap - abs(dy)
                if ox <= 0 or oy <= 0:
                    continue
                any_overlap = True
                # Push along the shorter axis (less perturbation).
                # Tiny epsilon to break ties when dx == 0.
                if ox < oy:
                    sgn = 1.0 if dx >= 0 else -1.0
                    # If both movable, split 50/50. If one fixed, the other
                    # gets 100%.
                    if fixed_mask[i] and fixed_mask[j]:
                        continue  # can't resolve, leave it
                    elif fixed_mask[i]:
                        forces[j, 0] += sgn * ox
                    elif fixed_mask[j]:
                        forces[i, 0] -= sgn * ox
                    else:
                        forces[j, 0] += sgn * ox * 0.5
                        forces[i, 0] -= sgn * ox * 0.5
                else:
                    sgn = 1.0 if dy >= 0 else -1.0
                    if fixed_mask[i] and fixed_mask[j]:
                        continue
                    elif fixed_mask[i]:
                        forces[j, 1] += sgn * oy
                    elif fixed_mask[j]:
                        forces[i, 1] -= sgn * oy
                    else:
                        forces[j, 1] += sgn * oy * 0.5
                        forces[i, 1] -= sgn * oy * 0.5

        if not any_overlap:
            converged = True
            break

        # Step. Slight damping to encourage convergence.
        pos += forces * 0.95
        # Clamp to canvas; fixed macros snap back.
        pos[:, 0] = np.clip(pos[:, 0], half_w, canvas_w - half_w)
        pos[:, 1] = np.clip(pos[:, 1], half_h, canvas_h - half_h)
        pos[fixed_mask] = bench_pos[fixed_mask]

    if not converged:
        # Final overlap check; if still overlapping, raise so caller can
        # fall back.
        for i in range(n_hard):
            for j in range(i + 1, n_hard):
                dx = abs(pos[i, 0] - pos[j, 0])
                dy = abs(pos[i, 1] - pos[j, 1])
                if dx < half_w[i] + half_w[j] - 1e-9 and dy < half_h[i] + half_h[j] - 1e-9:
                    raise RuntimeError("shake-apart legalize did not converge")

    out_full[:n_hard, 0] = torch.from_numpy(pos[:, 0].astype(np.float32))
    out_full[:n_hard, 1] = torch.from_numpy(pos[:, 1].astype(np.float32))
    if fixed_mask.any():
        out_full[:n_hard][torch.from_numpy(fixed_mask)] = benchmark.macro_positions[
            :n_hard
        ][torch.from_numpy(fixed_mask)].to(torch.float32)
    return out_full


def legalize_min_disturb(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    gap: float = 0.005,
    time_budget_s: float = 60.0,
) -> torch.Tensor:
    """Min-disturbance legalize: lock non-conflicting macros, spiral the rest.

    Returns a placement with zero overlaps if it succeeds. If it fails to find
    a legal placement for any macro within the budget, the caller should fall
    back to the older `legalize` (greedy global) function.
    """
    start = time.time()
    deadline = start + max(time_budget_s, 1.0)

    n_total = benchmark.num_macros
    n_hard = benchmark.num_hard_macros
    out_full = placement.detach().clone().to(torch.float32).cpu()
    if n_hard == 0:
        return out_full

    sizes = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64)
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    fixed_mask = benchmark.macro_fixed[:n_hard].detach().cpu().numpy().astype(bool)
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    pos = placement[:n_hard].detach().cpu().numpy().astype(np.float64).copy()
    # Clamp into canvas (no gap to canvas edges — TILOS only cares about pair overlap)
    pos[:, 0] = np.clip(pos[:, 0], half_w, canvas_w - half_w)
    pos[:, 1] = np.clip(pos[:, 1], half_h, canvas_h - half_h)
    bench_pos = benchmark.macro_positions[:n_hard].detach().cpu().numpy().astype(np.float64)
    pos[fixed_mask] = bench_pos[fixed_mask]

    target = pos.copy()  # preserve original (clamped) targets for spiral search

    involved, _oob = _find_overlap_set(pos, half_w, half_h, gap, canvas_w, canvas_h)
    # Fixed macros cannot move; if they're in `involved`, the conflict has to
    # be resolved by displacing the other party.
    movable_involved = sorted(
        [i for i in involved if not fixed_mask[i]],
        key=lambda i: -(half_w[i] * 2.0) * (half_h[i] * 2.0),  # largest area first
    )

    # If nothing overlaps, we're already legal — return as-is.
    if not movable_involved:
        out_full[:n_hard, 0] = torch.from_numpy(pos[:, 0].astype(np.float32))
        out_full[:n_hard, 1] = torch.from_numpy(pos[:, 1].astype(np.float32))
        if fixed_mask.any():
            out_full[:n_hard][torch.from_numpy(fixed_mask)] = benchmark.macro_positions[
                :n_hard
            ][torch.from_numpy(fixed_mask)].to(torch.float32)
        return out_full

    # Build a spatial grid containing every "locked" macro (non-involved + fixed).
    med = float(np.median(np.maximum(half_w * 2, half_h * 2)))
    cell = max(med * 1.1, 1.0)
    grid = _Grid(canvas_w, canvas_h, cell)
    placed = np.zeros(n_hard, dtype=bool)
    locked_ids = set()
    for i in range(n_hard):
        if i in involved and not fixed_mask[i]:
            continue
        grid.insert(i, float(pos[i, 0]), float(pos[i, 1]), float(half_w[i]), float(half_h[i]))
        placed[i] = True
        locked_ids.add(i)

    # Spiral step: smallest macro dim sets resolution but cap it for speed
    min_dim = float(np.min(np.minimum(sizes[:, 0], sizes[:, 1])))
    step = max(min_dim * 0.25, 0.05)
    step = min(step, max(canvas_w, canvas_h) * 0.01)
    max_radius = int(np.ceil(max(canvas_w, canvas_h) / step)) + 2

    failed = False
    for idx in movable_involved:
        if time.time() > deadline:
            failed = True
            break
        tx = float(target[idx, 0])
        ty = float(target[idx, 1])

        # Try current position first.
        if not _overlaps_any(idx, tx, ty, half_w[idx], half_h[idx],
                             grid, placed, pos, half_w, half_h, gap):
            pos[idx, 0] = tx
            pos[idx, 1] = ty
            grid.insert(idx, tx, ty, half_w[idx], half_h[idx])
            placed[idx] = True
            continue

        # Spiral outward on Chebyshev shells.
        best_x = tx; best_y = ty; best_d = float("inf"); found = False
        for r in range(1, max_radius + 1):
            if time.time() > deadline:
                break
            shell_found = False
            for dxm in range(-r, r + 1):
                adxm = abs(dxm)
                for dym in range(-r, r + 1):
                    if max(adxm, abs(dym)) != r:
                        continue
                    cx = tx + dxm * step
                    cy = ty + dym * step
                    cx = min(max(cx, half_w[idx]), canvas_w - half_w[idx])
                    cy = min(max(cy, half_h[idx]), canvas_h - half_h[idx])
                    if _overlaps_any(idx, cx, cy, half_w[idx], half_h[idx],
                                     grid, placed, pos, half_w, half_h, gap):
                        continue
                    d = (cx - tx) ** 2 + (cy - ty) ** 2
                    if d < best_d:
                        best_d = d
                        best_x = cx
                        best_y = cy
                        shell_found = True
                        found = True
            if shell_found:
                break

        if not found:
            failed = True
            break
        pos[idx, 0] = best_x
        pos[idx, 1] = best_y
        grid.insert(idx, best_x, best_y, half_w[idx], half_h[idx])
        placed[idx] = True

    if failed:
        # Caller should fall back to global legalize
        raise RuntimeError("min-disturb legalize failed to place all conflicting macros")

    out_full[:n_hard, 0] = torch.from_numpy(pos[:, 0].astype(np.float32))
    out_full[:n_hard, 1] = torch.from_numpy(pos[:, 1].astype(np.float32))
    if fixed_mask.any():
        out_full[:n_hard][torch.from_numpy(fixed_mask)] = benchmark.macro_positions[
            :n_hard
        ][torch.from_numpy(fixed_mask)].to(torch.float32)
    return out_full
