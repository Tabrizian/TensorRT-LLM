# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# CUDA-graph-safe NaN/Inf trap for diagnosing where bad values originate inside a model.
#
# Design:
#   - Walk the model's named_modules() once at attach time and assign each module a slot in a
#     persistent device-resident bool tensor (the 'flags').
#   - Register a forward hook per module that does ONLY device-side ops: torch.isnan + torch.isinf
#     OR-ed into the module's flag slot via logical_or_. No .item(), no .cpu(), no print inside
#     the hook -> safe to run inside captured CUDA graphs.
#   - check_and_log() is called from the executor driver loop AFTER model.forward returns
#     (outside the captured region). It copies flags to host and logs the earliest True slot.
#
# Enable with env var TRTLLM_NAN_TRAP=1.

from __future__ import annotations

import os
from typing import List, Optional

import torch

from tensorrt_llm.logger import logger

# Dtypes for which torch.isnan / torch.isinf are implemented in this PyTorch build.
# Sub-byte dtypes like Float4_e2m1fn_x2 (NVFP4) do NOT support these ops, so we
# silently skip them in the hook.
_CHECKABLE_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}
# FP8 dtypes are present in modern torch; guard via getattr because older builds
# may not expose them.
for _name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
    _dt = getattr(torch, _name, None)
    if _dt is not None:
        _CHECKABLE_DTYPES.add(_dt)


def _walk_floating_tensors(obj):
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point() and obj.dtype in _CHECKABLE_DTYPES:
            yield obj
        return
    if isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _walk_floating_tensors(x)
        return
    if isinstance(obj, dict):
        for x in obj.values():
            yield from _walk_floating_tensors(x)
        return


_NAN_TRAP_ENABLED = False


def enable_nan_trap() -> None:
    """Activate the NaN-TRAP — called after warmup completes so that
    warmup-time NaN (masked by the framework) doesn't dominate the trap log.
    """
    global _NAN_TRAP_ENABLED
    _NAN_TRAP_ENABLED = True
    try:
        logger.error("[NaN-TRAP] enabled post-warmup")
    except Exception:
        pass


class NanTrap:
    def __init__(self, names: List[str], device: torch.device):
        self.names = names
        # One bool slot per module. Persistent device buffer -> captureable.
        self.flags = torch.zeros(len(names), dtype=torch.bool, device=device)
        self._step = 0

    @classmethod
    def create_and_attach(cls, model: torch.nn.Module) -> "NanTrap":
        device = next(
            (p.device for p in model.parameters() if p.device.type == "cuda"), torch.device("cuda")
        )
        names: List[str] = []
        modules = []
        # The DSv4 compressor module is called for side-effect only; its return
        # value (kv_comp) has uninitialized regions that look like NaN to the
        # hook but never propagate downstream. Skip it to avoid false positives.
        for name, mod in model.named_modules():
            if name == "":
                continue
            _skip_compressor = name.endswith(".compressor") or ".compressor." in name
            if _skip_compressor:
                continue
            names.append(name)
            modules.append(mod)
        trap = cls(names, device)
        for i, mod in enumerate(modules):

            def make_hook(slot):
                def hook(_m, _inp, out):
                    try:
                        flag = None
                        for t in _walk_floating_tensors(out):
                            bad = torch.isnan(t).any() | torch.isinf(t).any()
                            flag = bad if flag is None else (flag | bad)
                        if flag is not None:
                            # In-place OR into persistent flag tensor (pure device op).
                            trap.flags[slot].logical_or_(flag)
                    except (NotImplementedError, RuntimeError):
                        # Some exotic dtypes (e.g. Float4_e2m1fn_x2) don't support
                        # isnan/isinf. Skip silently rather than killing the worker.
                        pass

                return hook

            mod.register_forward_hook(make_hook(i))
        logger.info(f"[NaN-TRAP] attached {len(names)} module hooks")
        return trap

    def check_and_log(self, rank: int) -> None:
        # Host-side; call OUTSIDE the captured region (once per model.forward).
        if not _NAN_TRAP_ENABLED:
            # Still clear flags so warmup-time triggers don't leak forward.
            try:
                self.flags.zero_()
            except Exception:
                pass
            return
        self._step += 1
        try:
            flags_host = self.flags.to("cpu", non_blocking=False)
        except RuntimeError:
            return
        bad = flags_host.nonzero(as_tuple=False).flatten().tolist()
        if not bad:
            return
        first = self.names[bad[0]]
        all_names = [self.names[i] for i in bad[:16]]
        logger.error(
            f"[NaN-TRAP] rank={rank} step={self._step} first_nan_module={first!r} "
            f"num_flagged={len(bad)} flagged_modules={all_names}"
        )
        self.flags.zero_()


def maybe_create_nan_trap(model: torch.nn.Module) -> Optional[NanTrap]:
    if os.environ.get("TRTLLM_NAN_TRAP", "0") != "1":
        return None
    return NanTrap.create_and_attach(model)
