"""
Incremental cost surrogate state for macro placement refinement.

Provides three reusable building blocks consumed by `pproxy_refine.py`:

  * `_build_net_index(benchmark)` — flatten the benchmark's net/pin structure
    into per-net arrays for fast incremental HPWL updates.
  * `HpwlState` — maintains per-net min/max x/y and the weighted-HPWL total.
    Supports O(|nets touched|) delta queries and apply for a single-macro
    move. Matches TILOS `get_wirelength` to within fp tolerance.
  * `DensityState` — maintains per-grid-cell occupied area for hard + soft
    macros and exposes the TILOS top-10% mean density cost.
  * `SpatialHash` — uniform-cell spatial hash over hard macros for fast
    pairwise overlap queries.

These are stored on the CPU; we never spend wall-clock time inside the
surrogate that we could spend inside the true TILOS evaluator running in
worker processes. The surrogate exists to *propose* moves and to keep
bookkeeping after we *commit* one — never to gate acceptance, which is
always done by `compute_proxy_cost`.
"""

from __future__ import annotations

import math
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


# ---------------------------------------------------------------------------
# Net index (HPWL incremental)
# ---------------------------------------------------------------------------


def _build_net_index(benchmark: Benchmark):
    """Pre-build per-net pin metadata for fast HPWL updates.

    Returns dict with:
        net_owner    : list[np.ndarray int32]    owner index per pin per net
        net_off      : list[np.ndarray float64]  pin offset (relative to macro center)
        net_anchor   : list[np.ndarray float64]  fixed absolute anchor (for ports)
        net_is_macro : list[np.ndarray bool]     True if pin belongs to a macro
        net_weight   : np.ndarray float64
        macro_nets   : list[list[int]]           nets touching each macro
    """
    num_hard = benchmark.num_hard_macros
    num_macros = benchmark.num_macros
    port_pos = benchmark.port_positions.cpu().numpy().astype(np.float64) \
        if benchmark.port_positions is not None and benchmark.port_positions.shape[0] > 0 \
        else np.zeros((0, 2), dtype=np.float64)

    pin_offsets_np: List[np.ndarray] = []
    for off in benchmark.macro_pin_offsets:
        if off is None or off.shape[0] == 0:
            pin_offsets_np.append(np.zeros((0, 2), dtype=np.float64))
        else:
            pin_offsets_np.append(off.cpu().numpy().astype(np.float64))

    net_owner: List[np.ndarray] = []
    net_off: List[np.ndarray] = []
    net_anchor: List[np.ndarray] = []
    net_is_macro: List[np.ndarray] = []
    macro_nets: List[List[int]] = [[] for _ in range(num_macros)]
    net_weights = benchmark.net_weights.cpu().numpy().astype(np.float64) \
        if benchmark.net_weights is not None and benchmark.net_weights.numel() > 0 \
        else np.ones(benchmark.num_nets, dtype=np.float64)

    for n_idx, pin_tensor in enumerate(benchmark.net_pin_nodes):
        if pin_tensor is None or pin_tensor.shape[0] == 0:
            net_owner.append(np.zeros(0, dtype=np.int32))
            net_off.append(np.zeros((0, 2), dtype=np.float64))
            net_anchor.append(np.zeros((0, 2), dtype=np.float64))
            net_is_macro.append(np.zeros(0, dtype=bool))
            continue
        pins = pin_tensor.cpu().numpy().astype(np.int64)
        owners = np.zeros(pins.shape[0], dtype=np.int32)
        offsets = np.zeros((pins.shape[0], 2), dtype=np.float64)
        anchors = np.zeros((pins.shape[0], 2), dtype=np.float64)
        is_macro = np.zeros(pins.shape[0], dtype=bool)
        touched = set()
        for k, (owner, slot) in enumerate(pins):
            if owner < num_hard:
                owners[k] = owner
                po = pin_offsets_np[owner]
                if 0 <= slot < po.shape[0]:
                    offsets[k] = po[slot]
                is_macro[k] = True
                touched.add(int(owner))
            elif owner < num_macros:
                owners[k] = owner  # soft macro
                is_macro[k] = True
                touched.add(int(owner))
            else:
                port_idx = int(owner) - num_macros
                if 0 <= port_idx < port_pos.shape[0]:
                    anchors[k] = port_pos[port_idx]
                owners[k] = -1
        net_owner.append(owners)
        net_off.append(offsets)
        net_anchor.append(anchors)
        net_is_macro.append(is_macro)
        for m in touched:
            macro_nets[m].append(n_idx)

    return dict(
        net_owner=net_owner, net_off=net_off, net_anchor=net_anchor,
        net_is_macro=net_is_macro, net_weight=net_weights, macro_nets=macro_nets,
    )


# ---------------------------------------------------------------------------
# Incremental HPWL state (matches TILOS exactly)
# ---------------------------------------------------------------------------


class HpwlState:
    """Maintain per-net min/max x/y and a normalized HPWL total."""

    def __init__(self, idx, positions_np: np.ndarray, canvas_w: float, canvas_h: float):
        self.idx = idx
        self.cw, self.ch = canvas_w, canvas_h
        self.num_nets = len(idx['net_owner'])
        self.minx = np.empty(self.num_nets, dtype=np.float64)
        self.maxx = np.empty(self.num_nets, dtype=np.float64)
        self.miny = np.empty(self.num_nets, dtype=np.float64)
        self.maxy = np.empty(self.num_nets, dtype=np.float64)
        for n in range(self.num_nets):
            xs, ys = self._pin_coords(n, positions_np)
            if xs.size == 0:
                self.minx[n] = self.maxx[n] = self.miny[n] = self.maxy[n] = 0.0
            else:
                self.minx[n] = xs.min(); self.maxx[n] = xs.max()
                self.miny[n] = ys.min(); self.maxy[n] = ys.max()
        # TILOS normalization: WL_norm = sum(weight*hpwl) / (cw+ch) / num_nets
        # See TILOS get_cost(): returns total weighted HPWL normalized.
        self._wl = self._compute_total()

    def _pin_coords(self, n, positions_np):
        owners = self.idx['net_owner'][n]
        offsets = self.idx['net_off'][n]
        anchors = self.idx['net_anchor'][n]
        if owners.size == 0:
            return np.zeros(0), np.zeros(0)
        owners_i = owners.astype(np.int64)
        is_macro = (owners_i >= 0)
        coords = np.zeros((owners.size, 2), dtype=np.float64)
        if is_macro.any():
            mi = owners_i[is_macro]
            coords[is_macro] = positions_np[mi] + offsets[is_macro]
        if (~is_macro).any():
            coords[~is_macro] = anchors[~is_macro]
        return coords[:, 0], coords[:, 1]

    def _compute_total(self):
        w = self.idx['net_weight']
        hpwl = (self.maxx - self.minx) + (self.maxy - self.miny)
        return float((w * hpwl).sum())

    @property
    def wl_total(self):
        return self._wl

    def wl_cost_normalized(self):
        # Match TILOS: get_cost() returns sum(hpwl)/normalizer; normalizer based
        # on canvas perimeter and number of nets. Use ratio only since
        # we want a comparison metric; absolute scale doesn't matter for SA.
        denom = max(1.0, (self.cw + self.ch) * self.num_nets)
        return self._wl / denom

    def delta_for_move(self, macro_idx, new_pos, positions_np, sizes_np):
        """Return (delta_wl, affected_nets, new_minx, new_maxx, new_miny, new_maxy)."""
        nets = self.idx['macro_nets'][macro_idx]
        if not nets:
            return 0.0, [], None, None, None, None
        cur_pos = positions_np[macro_idx]
        positions_np[macro_idx] = new_pos
        new_minx = np.empty(len(nets))
        new_maxx = np.empty(len(nets))
        new_miny = np.empty(len(nets))
        new_maxy = np.empty(len(nets))
        delta = 0.0
        for k, n in enumerate(nets):
            xs, ys = self._pin_coords(n, positions_np)
            if xs.size == 0:
                nmin = nmax = nymin = nymax = 0.0
            else:
                nmin = xs.min(); nmax = xs.max(); nymin = ys.min(); nymax = ys.max()
            new_minx[k] = nmin; new_maxx[k] = nmax
            new_miny[k] = nymin; new_maxy[k] = nymax
            w = self.idx['net_weight'][n]
            old_hp = (self.maxx[n] - self.minx[n]) + (self.maxy[n] - self.miny[n])
            new_hp = (nmax - nmin) + (nymax - nymin)
            delta += w * (new_hp - old_hp)
        positions_np[macro_idx] = cur_pos  # restore
        return delta, nets, new_minx, new_maxx, new_miny, new_maxy

    def apply_move(self, macro_idx, new_pos, positions_np,
                   new_minx, new_maxx, new_miny, new_maxy):
        nets = self.idx['macro_nets'][macro_idx]
        positions_np[macro_idx] = new_pos
        for k, n in enumerate(nets):
            old_hp = (self.maxx[n] - self.minx[n]) + (self.maxy[n] - self.miny[n])
            new_hp = (new_maxx[k] - new_minx[k]) + (new_maxy[k] - new_miny[k])
            w = self.idx['net_weight'][n]
            self._wl += w * (new_hp - old_hp)
            self.minx[n] = new_minx[k]; self.maxx[n] = new_maxx[k]
            self.miny[n] = new_miny[k]; self.maxy[n] = new_maxy[k]


# ---------------------------------------------------------------------------
# Incremental density state (matches TILOS exactly)
# ---------------------------------------------------------------------------


class DensityState:
    """Maintain per-cell occupied area; top-10% mean / 2 matches TILOS density_cost."""

    def __init__(self, benchmark: Benchmark, positions_np: np.ndarray):
        self.grid_cols = int(benchmark.grid_cols)
        self.grid_rows = int(benchmark.grid_rows)
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.cell_w = self.cw / self.grid_cols
        self.cell_h = self.ch / self.grid_rows
        self.cell_area = self.cell_w * self.cell_h
        self.n_cells = self.grid_cols * self.grid_rows
        # top 10% by count
        self.top_k = max(1, int(math.floor(self.n_cells * 0.10)))

        # Sizes for all macros (hard + soft)
        self.sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
        self.num_macros = benchmark.num_macros

        # Occupied area per cell (sum of macro/cell overlap areas)
        self.occ = np.zeros(self.n_cells, dtype=np.float64)
        # Track which cells each macro currently contributes to (sparse).
        # We'll recompute deltas, so keep last positions handy.
        for m in range(self.num_macros):
            self._add_macro(m, positions_np[m, 0], positions_np[m, 1])

    def _cell_bounds_for_rect(self, x, y, w, h):
        """Return inclusive (r0, r1, c0, c1) range of cells overlapped by rect."""
        bl_x = x - w * 0.5; bl_y = y - h * 0.5
        ur_x = x + w * 0.5; ur_y = y + h * 0.5
        # If completely outside canvas, skip
        if ur_x <= 0 or ur_y <= 0 or bl_x >= self.cw or bl_y >= self.ch:
            return None
        c0 = max(0, int(math.floor(bl_x / self.cell_w)))
        c1 = min(self.grid_cols - 1, int(math.floor((ur_x - 1e-12) / self.cell_w)))
        r0 = max(0, int(math.floor(bl_y / self.cell_h)))
        r1 = min(self.grid_rows - 1, int(math.floor((ur_y - 1e-12) / self.cell_h)))
        if c1 < c0 or r1 < r0:
            return None
        return r0, r1, c0, c1

    def _add_macro(self, m, x, y):
        w, h = self.sizes[m]
        bnds = self._cell_bounds_for_rect(x, y, w, h)
        if bnds is None: return
        r0, r1, c0, c1 = bnds
        # rect bounds
        rx0 = x - w * 0.5; rx1 = x + w * 0.5
        ry0 = y - h * 0.5; ry1 = y + h * 0.5
        for r in range(r0, r1 + 1):
            cy0 = r * self.cell_h; cy1 = cy0 + self.cell_h
            oy = max(0.0, min(ry1, cy1) - max(ry0, cy0))
            if oy <= 0: continue
            for c in range(c0, c1 + 1):
                cx0 = c * self.cell_w; cx1 = cx0 + self.cell_w
                ox = max(0.0, min(rx1, cx1) - max(rx0, cx0))
                if ox <= 0: continue
                self.occ[r * self.grid_cols + c] += ox * oy

    def _sub_macro(self, m, x, y):
        w, h = self.sizes[m]
        bnds = self._cell_bounds_for_rect(x, y, w, h)
        if bnds is None: return
        r0, r1, c0, c1 = bnds
        rx0 = x - w * 0.5; rx1 = x + w * 0.5
        ry0 = y - h * 0.5; ry1 = y + h * 0.5
        for r in range(r0, r1 + 1):
            cy0 = r * self.cell_h; cy1 = cy0 + self.cell_h
            oy = max(0.0, min(ry1, cy1) - max(ry0, cy0))
            if oy <= 0: continue
            for c in range(c0, c1 + 1):
                cx0 = c * self.cell_w; cx1 = cx0 + self.cell_w
                ox = max(0.0, min(rx1, cx1) - max(rx0, cx0))
                if ox <= 0: continue
                idx = r * self.grid_cols + c
                self.occ[idx] = max(0.0, self.occ[idx] - ox * oy)

    def move_macro(self, m, old_pos, new_pos):
        # Subtract old contribution
        self._sub_macro(m, old_pos[0], old_pos[1])
        self._add_macro(m, new_pos[0], new_pos[1])

    def density_cost(self):
        """Match TILOS: occupied/cell_area, top 10% by count, mean, * 0.5."""
        gd = self.occ / self.cell_area
        # mask out zero cells (TILOS sorts only nonzero, but top-10% by count of all cells)
        # actually TILOS uses occupied_cells = [gc for gc in self.grid_cells if gc != 0.0]
        # then density_cnt = floor(N_total * 0.10), picks top density_cnt over occupied list
        occ_only = gd[gd > 0]
        if occ_only.size == 0:
            return 0.0
        k = min(self.top_k, occ_only.size)
        top = np.partition(occ_only, -k)[-k:] if k < occ_only.size else occ_only
        return 0.5 * float(top.mean())

    def delta_density_for_move(self, m, old_pos, new_pos):
        """Snapshot current density, simulate move, compute delta, restore."""
        before = self.density_cost()
        self._sub_macro(m, old_pos[0], old_pos[1])
        self._add_macro(m, new_pos[0], new_pos[1])
        after = self.density_cost()
        # Restore
        self._sub_macro(m, new_pos[0], new_pos[1])
        self._add_macro(m, old_pos[0], old_pos[1])
        return after - before, before


# ---------------------------------------------------------------------------
# Spatial hash for overlap check
# ---------------------------------------------------------------------------


class SpatialHash:
    def __init__(self, cell, num_macros, half_w, half_h, positions, fixed):
        self.cell = cell
        self.bins = {}
        self.macro_bins = [set() for _ in range(num_macros)]
        self.half_w = half_w
        self.half_h = half_h
        self.fixed = fixed
        for m in range(num_macros):
            self.add(m, positions[m, 0], positions[m, 1])

    def _range(self, x, y, hw, hh):
        c = self.cell
        c0 = int(math.floor((x - hw) / c))
        c1 = int(math.floor((x + hw) / c))
        r0 = int(math.floor((y - hh) / c))
        r1 = int(math.floor((y + hh) / c))
        return r0, r1, c0, c1

    def add(self, m, x, y):
        r0, r1, c0, c1 = self._range(x, y, self.half_w[m], self.half_h[m])
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                self.bins.setdefault((r, c), set()).add(m)
                self.macro_bins[m].add((r, c))

    def remove(self, m):
        for key in self.macro_bins[m]:
            s = self.bins.get(key)
            if s is not None:
                s.discard(m)
        self.macro_bins[m].clear()

    def candidates(self, x, y, hw, hh):
        r0, r1, c0, c1 = self._range(x, y, hw, hh)
        out = set()
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                s = self.bins.get((r, c))
                if s: out.update(s)
        return out

    def has_overlap(self, m, x, y, hw, hh, positions, gap=0.05):
        cand = self.candidates(x, y, hw, hh)
        cand.discard(m)
        for j in cand:
            jx, jy = positions[j, 0], positions[j, 1]
            jhw, jhh = self.half_w[j], self.half_h[j]
            if abs(x - jx) < hw + jhw + gap and abs(y - jy) < hh + jhh + gap:
                return True
        return False
