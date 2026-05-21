"""
Monkey-patches for TILOS PlacementCost to make `compute_proxy_cost` fast
without changing its output.

Cost of an evaluation on a big IBM benchmark (e.g. ibm10) is ~44 s, almost
all in the `get_ref_node_id` hot path which does a linear membership test
against `soft_macro_pin_indices` / `hard_macro_pin_indices` for every pin
lookup inside `get_wirelength`.

Replacing those `list in` checks with a pre-built `dict` lookup gives the
exact same result, and reduces ibm10 evaluations from ~44 s to a few
hundred ms.

This module is intentionally importable from both the main process and the
ParallelProxyPool workers (no benchmark-specific state).
"""

from __future__ import annotations

# Use the same import path as macro_place/_plc.py — otherwise the patch
# binds to a different class object than what runtime evaluators use.
from macro_place._plc import PlacementCost

# Save originals so we can guard against double-patch.
if not getattr(PlacementCost, "_HRT_FAST_GET_REF", False):

    _original_get_ref_node_id = PlacementCost.get_ref_node_id

    def _fast_get_ref_node_id(self, node_idx: int = -1) -> int:
        """O(1) replacement for the upstream linear-scan `get_ref_node_id`.

        Builds a `pin_idx -> macro_idx` cache on first call, then re-uses it
        for every subsequent lookup. The cache only depends on netlist
        topology, which is fixed once the benchmark is loaded.
        """
        if node_idx == -1:
            return -1
        cache = getattr(self, "_hrt_pin_to_macro", None)
        if cache is None:
            cache = {}
            mod_name_to_indices = self.mod_name_to_indices
            modules_w_pins = self.modules_w_pins
            for pin_idx in self.hard_macro_pin_indices:
                pin = modules_w_pins[pin_idx]
                cache[pin_idx] = mod_name_to_indices.get(pin.get_macro_name(), -1)
            for pin_idx in self.soft_macro_pin_indices:
                pin = modules_w_pins[pin_idx]
                cache[pin_idx] = mod_name_to_indices.get(pin.get_macro_name(), -1)
            self._hrt_pin_to_macro = cache
            self._hrt_port_set = set(self.port_indices) if hasattr(self, "port_indices") else set()

        ref = cache.get(node_idx)
        if ref is not None:
            return ref
        # Port: returns itself per upstream contract.
        if node_idx in self._hrt_port_set:
            return node_idx
        # Unknown node: fall back to the slow path so behavior is
        # bit-exact in edge cases.
        return _original_get_ref_node_id(self, node_idx)

    PlacementCost.get_ref_node_id = _fast_get_ref_node_id
    PlacementCost._HRT_FAST_GET_REF = True
