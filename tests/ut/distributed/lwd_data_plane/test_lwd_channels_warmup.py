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
"""CPU-only tests for the channel warmup orchestration in ``lwd_channels``.

``lwd_warm_up_channels()`` drives one probe per endpoint channel in a fixed
global order and funnels the outcome through the gloo aggregate, so every
rank of the world raises the same verdict instead of hanging on the HCCL
timeout.  The per-channel probe is stubbed out here (it needs NPU ops);
``lwd_channel_probe`` itself is covered by ``test_lwd_channel_probe.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.distributed.lwd import LwdChannelEndpoints
from vllm_ascend.distributed.lwd_data_plane import lwd_channels
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType

_MY_RANK = 0
_PEER_RANK = 1


def _endpoints(dp_idx: int) -> LwdChannelEndpoints:
    return LwdChannelEndpoints(dp_idx=dp_idx, edge_rank=_MY_RANK, cloud_rank=_PEER_RANK)


def _groups(*dp_indices: int) -> dict:
    """(dp_idx, channel) -> (stand-in group, peer), the shape the warmup reads."""
    return {
        (dp_idx, channel): (SimpleNamespace(name=f"g{dp_idx}_{channel.value}"), _PEER_RANK)
        for dp_idx in dp_indices
        for channel in LwdChannelType
    }


class _World:
    """Scripted probe outcomes plus recorders for the world-side calls."""

    def __init__(self) -> None:
        self.real_probe = lwd_channels._lwd_probe_channel
        self.outcomes: dict = {
            (0, LwdChannelType.UP): lwd_channels._ProbeOutcome.OK,
            (0, LwdChannelType.DOWN): lwd_channels._ProbeOutcome.OK,
        }
        self.probes: list = []
        self.aggregate_calls: list = []
        self.verdict = True
        self.barriers = 0
        self.destroys = 0
        self.cpu_group = SimpleNamespace(name="cpu_group")


@pytest.fixture
def world(monkeypatch) -> _World:
    """Fresh endpoint/group state with the probe, aggregate and world stubbed."""
    state = _World()

    def fake_probe(group, peer, dp_idx, channel, is_sender):
        state.probes.append((dp_idx, channel, is_sender))
        outcome = state.outcomes[(dp_idx, channel)]
        if outcome is lwd_channels._ProbeOutcome.OK:
            return lwd_channels._ChannelProbeResult(lwd_channels._ProbeOutcome.OK)
        if outcome is lwd_channels._ProbeOutcome.TIMEOUT:
            detail = lwd_channels._lwd_probe_timeout_detail(dp_idx, channel, peer)
        else:
            detail = f"frame mismatch (peer={peer}): [{channel.value}] wrong frame"
        return lwd_channels._ChannelProbeResult(outcome, detail)

    def fake_aggregate(local_ok, group):
        state.aggregate_calls.append((local_ok, group))
        # MIN semantics: one failure fails every rank.
        return state.verdict and local_ok

    def fake_destroy() -> None:
        state.destroys += 1

    def fake_barrier() -> None:
        state.barriers += 1

    def fake_world_group():
        return SimpleNamespace(cpu_group=state.cpu_group, barrier=fake_barrier)

    monkeypatch.setattr(lwd_channels, "_LWD_ENDPOINTS", (_endpoints(0),))
    monkeypatch.setattr(lwd_channels, "_LWD_CHANNEL_GROUPS", _groups(0))
    monkeypatch.setattr(lwd_channels.dist, "get_rank", lambda: _MY_RANK)
    monkeypatch.setattr(lwd_channels, "_lwd_probe_channel", fake_probe)
    monkeypatch.setattr(lwd_channels, "lwd_probe_aggregate", fake_aggregate)
    monkeypatch.setattr(lwd_channels, "lwd_destroy_channels", fake_destroy)
    monkeypatch.setattr(lwd_channels, "get_world_group", fake_world_group)
    return state


def test_all_channels_ok(world) -> None:
    """Happy path: every channel probes in the fixed order, then the barrier."""
    lwd_channels.lwd_warm_up_channels()
    assert [(dp, ch) for dp, ch, _ in world.probes] == [
        (0, LwdChannelType.UP),
        (0, LwdChannelType.DOWN),
    ]
    assert [sender for _, _, sender in world.probes] == [True, False]
    assert world.aggregate_calls == [(True, world.cpu_group)]
    assert world.barriers == 1
    assert world.destroys == 0


def test_frame_mismatch_keeps_probing_and_rolls_back(world) -> None:
    """A wrong frame is recorded but probing continues; clean ops roll back."""
    world.outcomes[(0, LwdChannelType.UP)] = lwd_channels._ProbeOutcome.MISMATCH
    with pytest.raises(RuntimeError, match="frame mismatch"):
        lwd_channels.lwd_warm_up_channels()
    # The op count/order must stay invariant for later pairing.
    assert [(dp, ch) for dp, ch, _ in world.probes] == [
        (0, LwdChannelType.UP),
        (0, LwdChannelType.DOWN),
    ]
    assert world.aggregate_calls[0][0] is False
    assert world.destroys == 1
    assert world.barriers == 0


def test_timeout_stops_both_loops_and_skips_rollback(world, monkeypatch) -> None:
    """A wedged channel stops all probing; groups are left for process exit."""
    monkeypatch.setattr(lwd_channels, "_LWD_ENDPOINTS", (_endpoints(0), _endpoints(1)))
    monkeypatch.setattr(lwd_channels, "_LWD_CHANNEL_GROUPS", _groups(0, 1))
    world.outcomes = {
        (0, LwdChannelType.UP): lwd_channels._ProbeOutcome.TIMEOUT,
        (0, LwdChannelType.DOWN): lwd_channels._ProbeOutcome.OK,
        (1, LwdChannelType.UP): lwd_channels._ProbeOutcome.OK,
        (1, LwdChannelType.DOWN): lwd_channels._ProbeOutcome.OK,
    }
    with pytest.raises(RuntimeError, match="TIMEOUT"):
        lwd_channels.lwd_warm_up_channels()
    assert world.probes == [(0, LwdChannelType.UP, True)]
    assert world.aggregate_calls[0][0] is False
    assert world.destroys == 0
    assert world.barriers == 0


def test_non_endpoint_rank_only_aggregates_and_barriers(world, monkeypatch) -> None:
    """Ranks without channels still join the verdict and the closing barrier."""
    monkeypatch.setattr(lwd_channels, "_LWD_ENDPOINTS", ())
    lwd_channels.lwd_warm_up_channels()
    assert world.probes == []
    assert world.aggregate_calls == [(True, world.cpu_group)]
    assert world.barriers == 1
    assert world.destroys == 0


def test_remote_failure_without_local_detail_uses_fallback(world) -> None:
    """A remote failure with clean local probes names the fallback detail."""
    world.verdict = False
    with pytest.raises(RuntimeError, match=r"remote rank\(s\) reported probe failure"):
        lwd_channels.lwd_warm_up_channels()
    assert world.aggregate_calls[0][0] is True
    assert world.destroys == 1


def test_non_hccl_backend_fails_before_posting_an_op(world, monkeypatch) -> None:
    """A degraded (gloo) group fails before any op, via the shared verdict."""
    monkeypatch.setattr(lwd_channels, "_lwd_probe_channel", world.real_probe)
    posted: list = []
    monkeypatch.setattr(lwd_channels.dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(
        lwd_channels.dist, "isend", lambda *args, **kwargs: posted.append("isend")
    )
    monkeypatch.setattr(
        lwd_channels.dist, "irecv", lambda *args, **kwargs: posted.append("irecv")
    )
    with pytest.raises(RuntimeError, match="backend must be hccl"):
        lwd_channels.lwd_warm_up_channels()
    assert posted == []
    # Ledgered like a TIMEOUT: the rank still reaches the shared verdict and
    # skips the rollback, so a symmetric failure cannot hang the other ranks.
    assert world.aggregate_calls == [(False, world.cpu_group)]
    assert world.destroys == 0
    assert world.barriers == 0


def test_unexpected_exception_is_ledgered_and_aborts(world, monkeypatch) -> None:
    """A raising probe (OOM, broken context) is recorded like a TIMEOUT.

    The rank still reaches the aggregate, so one failing rank cannot hang
    the others at the verdict; the groups are left for process exit.
    """
    monkeypatch.setattr(lwd_channels, "_LWD_ENDPOINTS", (_endpoints(0), _endpoints(1)))
    monkeypatch.setattr(lwd_channels, "_LWD_CHANNEL_GROUPS", _groups(0, 1))

    def raising_probe(group, peer, dp_idx, channel, is_sender):
        world.probes.append((dp_idx, channel, is_sender))
        raise MemoryError("OOM")

    monkeypatch.setattr(lwd_channels, "_lwd_probe_channel", raising_probe)
    with pytest.raises(RuntimeError, match="probe raised"):
        lwd_channels.lwd_warm_up_channels()
    assert world.probes == [(0, LwdChannelType.UP, True)]
    assert world.aggregate_calls == [(False, world.cpu_group)]
    assert world.destroys == 0
    assert world.barriers == 0


def test_base_exception_propagates(world, monkeypatch) -> None:
    """KeyboardInterrupt/SystemExit are never swallowed into the ledger."""
    monkeypatch.setattr(
        lwd_channels,
        "_lwd_probe_channel",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    with pytest.raises(KeyboardInterrupt):
        lwd_channels.lwd_warm_up_channels()
    assert world.aggregate_calls == []
    assert world.destroys == 0
    assert world.barriers == 0
