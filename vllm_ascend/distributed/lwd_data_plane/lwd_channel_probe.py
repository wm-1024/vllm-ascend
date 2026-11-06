# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Identity probe exchanged on the channel warmup round-trip.

Before any business payload flows, each channel's existing warmup
isend/irecv carries one self-describing frame instead of zeros: the sender
derives an 8-element bf16 frame from public facts only (its own global rank,
the DP connection index, the channel direction), and the receiver
independently rebuilds the expected frame and compares bit-exact.  A wrong
peer, a crossed connection, a swapped direction or a corrupted / never-
written buffer therefore fails loudly here, at init time, instead of
surfacing as an ~1836 s HCCL hang on first business traffic.

This module is deliberately a **leaf** next to ``lwd_payload_scrambler``: it
reads no configuration and imports nothing from this project except
``LwdChannelType``.  Both ends derive the frame from the same public facts,
so the probe needs no nonce and no exchange of its own.

Scope note: this probes the handshake only.  Corruption of real in-flight
transfers (buffer reuse, stream races, truncation) needs a runtime checksum,
and large-message / segmented-path acceptance belongs to the offline
hccl_test matrix.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.distributed as dist

from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType

__all__ = [
    "LWD_PROBE_TIMEOUT_S",
    "lwd_probe_aggregate",
    "lwd_probe_frame",
    "lwd_probe_verify",
    "lwd_probe_wait",
]

# Fail-fast budget; public because the caller logs it in failure messages.
LWD_PROBE_TIMEOUT_S = 30.0
_LWD_PROBE_POLL_INTERVAL_S = 0.02

# Frame layout: 8 bf16 elements, self-describing, size-invariant (the warmup
# has always carried exactly this many elements).
_LWD_PROBE_FRAME_NUM_ELEMENTS = 8
# bf16 represents every integer in [0, 256) exactly (8-bit significand), so
# the frame crosses the wire losslessly; wider identities fail fast instead.
_LWD_PROBE_FIELD_LIMIT = 256
_LWD_PROBE_MAGIC = 90
# The three check sequences are reduced mod 251 (a prime inside the exact
# bf16 range), so they stay in-range and mix the identity fields.
_LWD_PROBE_CHECKSUM_MOD = 251

_LWD_PROBE_FIELD_NAMES = {
    0: "not a probe frame / recv not written",
    1: "peer rank mismatch",
    2: "dp_idx mismatch (crossed connection)",
    3: "channel direction mismatch",
    4: "frame length mismatch",
    5: "data corruption",
    6: "data corruption",
    7: "data corruption",
}


def lwd_probe_frame(
    sender_rank: int,
    dp_idx: int,
    channel: LwdChannelType,
    device: str | torch.device = "npu",
) -> torch.Tensor:
    """The identity frame both peers derive from the same public facts."""
    ch = 0 if channel is LwdChannelType.UP else 1
    assert 0 <= sender_rank < _LWD_PROBE_FIELD_LIMIT, f"sender_rank={sender_rank} out of range"
    assert 0 <= dp_idx < _LWD_PROBE_FIELD_LIMIT, f"dp_idx={dp_idx} out of range"
    vals = (
        _LWD_PROBE_MAGIC,
        sender_rank,
        dp_idx,
        ch,
        _LWD_PROBE_FRAME_NUM_ELEMENTS,
        (sender_rank + 5) % _LWD_PROBE_CHECKSUM_MOD,
        (dp_idx * 7 + 11) % _LWD_PROBE_CHECKSUM_MOD,
        (ch * 13 + 17) % _LWD_PROBE_CHECKSUM_MOD,
    )
    return torch.tensor(vals, dtype=torch.bfloat16, device=device)


def lwd_probe_wait(handle: Any, timeout_s: float = LWD_PROBE_TIMEOUT_S) -> bool:
    """True when the op completed; False on TIMEOUT (wedged channel).

    ``is_completed()`` only observes completion, so the final ``wait()`` is
    kept to surface an op that completed with an error.
    """
    deadline = time.monotonic() + timeout_s
    while not handle.is_completed():
        if time.monotonic() >= deadline:
            return False
        time.sleep(_LWD_PROBE_POLL_INTERVAL_S)
    handle.wait()
    return True


def lwd_probe_verify(
    got: torch.Tensor,
    peer: int,
    dp_idx: int,
    channel: LwdChannelType,
) -> str:
    """Bit-exact per-field compare on the receiver; "" = pass, else diagnosis."""
    if got.numel() != _LWD_PROBE_FRAME_NUM_ELEMENTS:
        return f"frame length mismatch: got {got.numel()} elements, expected {_LWD_PROBE_FRAME_NUM_ELEMENTS}"
    expected = lwd_probe_frame(peer, dp_idx, channel, device=str(got.device))
    got_bits = got.view(torch.int16).tolist()
    exp_bits = expected.view(torch.int16).tolist()
    problems = [
        f"[{i}] {_LWD_PROBE_FIELD_NAMES[i]}: got {got_bits[i]}, expected {exp_bits[i]}"
        for i in range(_LWD_PROBE_FRAME_NUM_ELEMENTS)
        if got_bits[i] != exp_bits[i]
    ]
    return "; ".join(problems)


def lwd_probe_aggregate(local_ok: bool, group: dist.ProcessGroup) -> bool:
    """All-reduce (MIN) the local verdict: one failure fails every rank.

    Runs over the CPU/gloo group, never through the HCCL channels under
    test, so the verdict cannot be blocked by the failure it reports.
    """
    verdict = torch.tensor([1 if local_ok else 0], dtype=torch.int32)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=group)
    return bool(verdict.item())
