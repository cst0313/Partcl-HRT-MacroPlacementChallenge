"""DREAMPlace integration wrapper.

Converts a TILOS Benchmark into bookshelf format, runs DREAMPlace's
global placement, then reads positions back. The wrapper deliberately
keeps DREAMPlace at module import-time scope so we can fall back
cleanly if the build is incomplete.
"""
from __future__ import annotations

import os
import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from macro_place.benchmark import Benchmark


DPL_ROOT = Path("/tmp/dreamplace_src/DREAMPlace-master")
DPL_BUILD = DPL_ROOT / "build"


def _scale(x: float) -> int:
    """Bookshelf wants integer coordinates. Use 1000x scaling to preserve resolution."""
    return int(round(x * 1000))


def _write_bookshelf(bm: Benchmark, outdir: Path, name: str = "design") -> None:
    """Write the benchmark as bookshelf .aux/.nodes/.nets/.pl/.scl."""
    outdir.mkdir(parents=True, exist_ok=True)

    n_macros = bm.num_macros
    n_hard = bm.num_hard_macros
    n_ports = bm.port_positions.shape[0]
    n_total = n_macros + n_ports

    # --- .nodes ---
    nodes_path = outdir / f"{name}.nodes"
    with nodes_path.open("w") as f:
        f.write("UCLA nodes 1.0\n\n")
        f.write(f"NumNodes : {n_total}\n")
        # Terminals = fixed macros + ports. We mark ports as terminal_NI (overlap-able).
        n_fixed_macros = int(bm.macro_fixed.sum().item())
        f.write(f"NumTerminals : {n_fixed_macros + n_ports}\n")

        sizes = bm.macro_sizes.cpu().numpy()
        fixed = bm.macro_fixed.cpu().numpy()
        for i in range(n_macros):
            w, h = _scale(sizes[i, 0]), _scale(sizes[i, 1])
            w = max(1, w)
            h = max(1, h)
            tag = " terminal" if fixed[i] else ""
            f.write(f"\to{i} {w} {h}{tag}\n")
        # Ports: zero-size terminal_NI
        for j in range(n_ports):
            f.write(f"\tp{j} 1 1 terminal_NI\n")

    # --- .pl ---
    pl_path = outdir / f"{name}.pl"
    with pl_path.open("w") as f:
        f.write("UCLA pl 1.0\n\n")
        pos = bm.macro_positions.cpu().numpy()
        for i in range(n_macros):
            # Bookshelf uses lower-left corner, not center.
            x_ll = _scale(pos[i, 0] - sizes[i, 0] / 2)
            y_ll = _scale(pos[i, 1] - sizes[i, 1] / 2)
            fixed_tag = " /FIXED" if fixed[i] else ""
            f.write(f"o{i} {x_ll} {y_ll} : N{fixed_tag}\n")
        ports = bm.port_positions.cpu().numpy()
        for j in range(n_ports):
            x_ll = _scale(ports[j, 0])
            y_ll = _scale(ports[j, 1])
            f.write(f"p{j} {x_ll} {y_ll} : N /FIXED_NI\n")

    # --- .nets ---
    nets_path = outdir / f"{name}.nets"
    pin_offsets_list = bm.macro_pin_offsets
    with nets_path.open("w") as f:
        f.write("UCLA nets 1.0\n\n")
        # Use pin-level nets if available, else fall back to per-node
        if bm.net_pin_nodes:
            num_pins_total = sum(t.shape[0] for t in bm.net_pin_nodes)
            f.write(f"NumNets : {bm.num_nets}\n")
            f.write(f"NumPins : {num_pins_total}\n")
            for nid, pin_t in enumerate(bm.net_pin_nodes):
                if pin_t.shape[0] < 2:
                    continue
                f.write(f"NetDegree : {pin_t.shape[0]} n{nid}\n")
                arr = pin_t.cpu().numpy()
                for row in arr:
                    owner, pidx = int(row[0]), int(row[1])
                    if owner < n_hard:
                        cell = f"o{owner}"
                        offs = pin_offsets_list[owner].cpu().numpy() if owner < len(pin_offsets_list) else None
                        if offs is not None and pidx < offs.shape[0]:
                            ox, oy = _scale(offs[pidx, 0]), _scale(offs[pidx, 1])
                        else:
                            ox, oy = 0, 0
                    elif owner < n_macros:
                        cell = f"o{owner}"
                        ox, oy = 0, 0
                    else:
                        cell = f"p{owner - n_macros}"
                        ox, oy = 0, 0
                    f.write(f"\t{cell} I : {ox} {oy}\n")
        else:
            num_pins_total = sum(t.shape[0] for t in bm.net_nodes)
            f.write(f"NumNets : {bm.num_nets}\n")
            f.write(f"NumPins : {num_pins_total}\n")
            for nid, nodes_t in enumerate(bm.net_nodes):
                if nodes_t.shape[0] < 2:
                    continue
                f.write(f"NetDegree : {nodes_t.shape[0]} n{nid}\n")
                for node in nodes_t.cpu().numpy():
                    node = int(node)
                    if node < n_macros:
                        cell = f"o{node}"
                    else:
                        cell = f"p{node - n_macros}"
                    f.write(f"\t{cell} I : 0 0\n")

    # --- .scl ---
    scl_path = outdir / f"{name}.scl"
    cw = _scale(bm.canvas_width)
    ch = _scale(bm.canvas_height)
    # Use a single row spanning the canvas with site height = canvas_height/grid_rows
    row_h = max(1, ch // max(1, bm.grid_rows))
    num_rows = ch // row_h
    with scl_path.open("w") as f:
        f.write("UCLA scl 1.0\n\n")
        f.write(f"NumRows : {num_rows}\n\n")
        for r in range(num_rows):
            f.write("CoreRow Horizontal\n")
            f.write(f"  Coordinate    :   {r * row_h}\n")
            f.write(f"  Height        :   {row_h}\n")
            f.write(f"  Sitewidth     :   1\n")
            f.write(f"  Sitespacing   :   1\n")
            f.write(f"  Siteorient    :   N\n")
            f.write(f"  Sitesymmetry  :   Y\n")
            f.write(f"  SubrowOrigin  :   0\tNumSites  :  {cw}\n")
            f.write("End\n")

    # --- .wts ---
    wts_path = outdir / f"{name}.wts"
    with wts_path.open("w") as f:
        f.write("UCLA wts 1.0\n\n")

    # --- .aux ---
    aux_path = outdir / f"{name}.aux"
    with aux_path.open("w") as f:
        f.write(f"RowBasedPlacement : {name}.nodes {name}.nets {name}.wts {name}.pl {name}.scl\n")


def _make_params(aux_path: Path, result_dir: Path, iters: int = 1000) -> Path:
    """Generate DREAMPlace params.json."""
    params = {
        "aux_input": str(aux_path),
        "gpu": 0,  # 0 = CPU
        "num_threads": 16,
        "num_bins_x": 512,
        "num_bins_y": 512,
        "global_place_stages": [{
            "num_bins_x": 512,
            "num_bins_y": 512,
            "iteration": iters,
            "learning_rate": 0.01,
            "wirelength": "weighted_average",
            "optimizer": "nesterov",
            "Llambda_density_weight_iteration": 1,
            "Lsub_iteration": 1
        }],
        "target_density": 0.9,
        "density_weight": 8e-5,
        "random_seed": 1000,
        "result_dir": str(result_dir),
        "scale_factor": 0.0,
        "ignore_net_degree": 100,
        "gp_noise_ratio": 0.025,
        "enable_fillers": 1,
        "global_place_flag": 1,
        "legalize_flag": 0,
        "detailed_place_flag": 0,
        "stop_overflow": 0.07,
        "dtype": "float32",
        "plot_flag": 0,
        "RePlAce_skip_energy_flag": 0,
        "deterministic_flag": 1,
        "num_threads": 16,
    }
    params_path = result_dir / "params.json"
    params_path.write_text(json.dumps(params, indent=2))
    return params_path


def _read_pl_output(pl_path: Path, num_macros: int, sizes: np.ndarray) -> np.ndarray:
    """Read DREAMPlace's output .pl file. Returns macro centers as [num_macros, 2]."""
    out = np.zeros((num_macros, 2), dtype=np.float64)
    with pl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("UCLA") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3 or not parts[0].startswith("o"):
                continue
            try:
                i = int(parts[0][1:])
            except ValueError:
                continue
            if i >= num_macros:
                continue
            x_ll = float(parts[1]) / 1000.0
            y_ll = float(parts[2]) / 1000.0
            out[i, 0] = x_ll + sizes[i, 0] / 2
            out[i, 1] = y_ll + sizes[i, 1] / 2
    return out


def dpl_available() -> bool:
    """Quick check if the DREAMPlace build is present."""
    return DPL_ROOT.is_dir() and (DPL_BUILD / "dreamplace").is_dir()


def run_dpl_global_placement(
    bm: Benchmark,
    iters: int = 1000,
    workdir: Optional[Path] = None,
) -> Optional[torch.Tensor]:
    """Run DREAMPlace global placement on benchmark; return new macro positions or None on failure.

    Only positions for movable macros are updated; fixed macros are returned at their original
    positions.
    """
    if not dpl_available():
        return None

    cleanup = False
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="dpl_"))
        cleanup = True
    name = "design"

    try:
        _write_bookshelf(bm, workdir, name)
        result_dir = workdir / "result"
        result_dir.mkdir(exist_ok=True)
        params_path = _make_params(workdir / f"{name}.aux", result_dir, iters=iters)

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "16"
        cmd = [
            "/home/coder/macro-place-challenge-2026/.venv/bin/python",
            "dreamplace/Placer.py",
            str(params_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600, cwd=str(DPL_BUILD))
        if proc.returncode != 0:
            print(f"[dpl] returncode={proc.returncode}")
            print(proc.stderr[-2000:])
            return None

        # Output is in result_dir/<design_name>/<design_name>.gp.pl
        candidates = list(result_dir.rglob("*.gp.pl"))
        if not candidates:
            print(f"[dpl] no .gp.pl found in {result_dir}")
            return None
        sizes = bm.macro_sizes.cpu().numpy()
        new_pos = _read_pl_output(candidates[0], bm.num_macros, sizes)
        # Keep fixed macros at original positions
        fixed = bm.macro_fixed.cpu().numpy()
        orig = bm.macro_positions.cpu().numpy()
        new_pos = np.where(fixed[:, None], orig, new_pos)
        return torch.tensor(new_pos, dtype=torch.float32)
    finally:
        if cleanup:
            shutil.rmtree(workdir, ignore_errors=True)
