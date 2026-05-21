"""
Multi-process parallel proxy cost evaluator.

Spawns N worker processes; each holds a private copy of the PlacementCost
object. Submit candidate placements; receive (proxy, overlap_count) tuples.

Used by true_refine to evaluate K>1 candidates per macro in wall-clock
parallel, getting K-fold speedup on the dominant cost (TILOS evaluator).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch


def _worker_loop(
    benchmark_dir: str,
    ng45_dir: Optional[str],
    in_queue: mp.Queue,
    out_queue: mp.Queue,
):
    """Worker: load benchmark+plc once, then service eval requests."""
    # Suppress TILOS prints in workers
    import sys
    sys.stdout = open(os.devnull, "w")

    # Apply PlacementCost speed patches (no-op if already applied).
    import sys as _sys
    from pathlib import Path as _Path
    _here = _Path(__file__).resolve().parent
    if str(_here) not in _sys.path:
        _sys.path.insert(0, str(_here))
    import _plc_patches  # noqa: F401
    from macro_place.loader import load_benchmark_from_dir, load_benchmark
    from macro_place.objective import compute_proxy_cost

    if ng45_dir:
        netlist_file = f"{ng45_dir}/netlist.pb.txt"
        plc_file = f"{ng45_dir}/initial.plc"
        benchmark, plc = load_benchmark(netlist_file, plc_file)
    else:
        benchmark, plc = load_benchmark_from_dir(benchmark_dir)

    out_queue.put(("READY", os.getpid()))

    while True:
        try:
            msg = in_queue.get()
        except Exception:
            return
        if msg is None:
            return
        request_id, placement_np = msg
        placement_t = torch.from_numpy(placement_np).float()
        try:
            cost = compute_proxy_cost(placement_t, benchmark, plc)
            proxy = float(cost["proxy_cost"])
            ovl = int(cost.get("overlap_count", 0))
            wl = float(cost["wirelength_cost"])
            den = float(cost["density_cost"])
            cong = float(cost["congestion_cost"])
        except Exception as e:
            out_queue.put((request_id, None, str(e)))
            continue
        out_queue.put((request_id, (proxy, ovl, wl, den, cong), None))


class ParallelProxyPool:
    """Worker pool for batch proxy-cost evaluation.

    Usage:
        pool = ParallelProxyPool(benchmark_dir="...", num_workers=4)
        # ...
        results = pool.evaluate_batch([placement1, placement2, ...])
        # results is List[(proxy, overlap_count, wl, den, cong)]
        pool.shutdown()
    """

    def __init__(
        self,
        benchmark_dir: Optional[str] = None,
        ng45_dir: Optional[str] = None,
        num_workers: int = 4,
        spawn_timeout: float = 60.0,
    ):
        if benchmark_dir is None and ng45_dir is None:
            raise ValueError("Must provide benchmark_dir or ng45_dir")
        self.num_workers = num_workers
        self.in_queues: List[mp.Queue] = []
        self.out_queue: mp.Queue = mp.Queue()
        self.procs: List[mp.Process] = []
        for _ in range(num_workers):
            q = mp.Queue()
            p = mp.Process(
                target=_worker_loop,
                args=(benchmark_dir, ng45_dir, q, self.out_queue),
                daemon=True,
            )
            p.start()
            self.in_queues.append(q)
            self.procs.append(p)
        # Wait for all workers to be ready
        ready = 0
        deadline = time.time() + spawn_timeout
        while ready < num_workers and time.time() < deadline:
            try:
                msg = self.out_queue.get(timeout=spawn_timeout)
            except Exception:
                break
            if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "READY":
                ready += 1
        if ready < num_workers:
            self.shutdown()
            raise RuntimeError(f"Only {ready}/{num_workers} workers became ready")

    def evaluate_batch(self, placements: List[np.ndarray]) -> List[Optional[Tuple[float, int, float, float, float]]]:
        """Evaluate a batch of placements. Returns list of (proxy, ovl, wl, den, cong) or None on error.

        placements: list of np.ndarray, each of shape [num_macros, 2].
        """
        n = len(placements)
        if n == 0:
            return []
        if n > self.num_workers:
            # Process in chunks
            results = []
            for i in range(0, n, self.num_workers):
                results.extend(self.evaluate_batch(placements[i : i + self.num_workers]))
            return results

        # Dispatch
        for k in range(n):
            self.in_queues[k].put((k, placements[k]))
        # Collect
        results: List[Optional[Tuple]] = [None] * n
        outstanding = set(range(n))
        deadline = time.time() + 300.0
        while outstanding and time.time() < deadline:
            try:
                msg = self.out_queue.get(timeout=60.0)
            except Exception:
                break
            if isinstance(msg, tuple) and len(msg) >= 2 and isinstance(msg[0], int):
                req_id, payload, _err = msg if len(msg) == 3 else (msg[0], msg[1], None)
                if req_id in outstanding:
                    results[req_id] = payload
                    outstanding.discard(req_id)
        return results

    def shutdown(self):
        for q in self.in_queues:
            try:
                q.put(None, timeout=1.0)
            except Exception:
                pass
        for p in self.procs:
            try:
                p.join(timeout=2.0)
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
