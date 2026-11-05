# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-only tests for the warmup orchestration in ``lwd_channels``.

Covers the warmup outcomes: every channel OK, a frame mismatch (rollback +
keep probing), a poll TIMEOUT (stop, no rollback), a non-HCCL backend
(fail closed), a remote-only failure (fallback text) and a non-endpoint
rank (verdict + barrier only).  ``_lwd_probe_channel`` is scripted, so no
NPU or HCCL link is needed; the real sender/receiver branches stay on the
real-machine matrix.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.distributed.lwd import LwdChannelEndpoints

from vllm_ascend.distributed.lwd_data_plane import lwd_channels
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType


class _WorldGroup:
    """World-group stand-in: the CPU verdict group and the closing barrier."""

    def __init__(self) -> None:
        self.cpu_group = object()
        self.barrier_calls = 0

    def barrier(self) -> None:
        self.barrier_calls += 1


def _ok() -> lwd_channels._ChannelProbeResult:
    return lwd_channels._ChannelProbeResult(lwd_channels._ProbeOutcome.OK)


def _fail(outcome: lwd_channels._ProbeOutcome, detail: str = "") -> lwd_channels._ChannelProbeResult:
    return lwd_channels._ChannelProbeResult(outcome, detail)


def _install(
    monkeypatch: pytest.MonkeyPatch,
    results: list[lwd_channels._ChannelProbeResult],
    *,
    aggregate_ok: bool = True,
) -> SimpleNamespace:
    """Patch the warmup inputs: two DP connections served by edge rank 0."""
    endpoints = (
        LwdChannelEndpoints(dp_idx=0, edge_rank=0, cloud_rank=1),
        LwdChannelEndpoints(dp_idx=1, edge_rank=0, cloud_rank=3),
    )
    groups = {(ep.dp_idx, ch): (object(), ep.cloud_rank) for ep in endpoints for ch in LwdChannelType}
    world = _WorldGroup()
    rec = SimpleNamespace(calls=[], logs=[], verdicts=[], destroyed=0, world=world)
    pending = list(results)

    def fake_probe(
        group: object, peer: int, dp_idx: int, channel: LwdChannelType, is_sender: bool
    ) -> lwd_channels._ChannelProbeResult:
        rec.calls.append((dp_idx, channel, is_sender))
        return pending.pop(0)

    def fake_aggregate(local_ok: bool, group: object) -> bool:
        rec.verdicts.append(local_ok)
        return aggregate_ok

    def fake_destroy() -> None:
        rec.destroyed += 1

    def fake_log(fmt: str, *args: object) -> None:
        rec.logs.append(fmt % args)

    monkeypatch.setattr(lwd_channels, "_LWD_ENDPOINTS", endpoints)
    monkeypatch.setattr(lwd_channels, "_LWD_CHANNEL_GROUPS", groups)
    monkeypatch.setattr(lwd_channels, "get_world_group", lambda: world)
    monkeypatch.setattr(lwd_channels, "logger", SimpleNamespace(info=fake_log, error=fake_log))
    monkeypatch.setattr(lwd_channels.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(lwd_channels, "_lwd_probe_channel", fake_probe)
    monkeypatch.setattr(lwd_channels, "lwd_probe_aggregate", fake_aggregate)
    monkeypatch.setattr(lwd_channels, "lwd_destroy_channels", fake_destroy)
    return rec


def test_all_channels_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every channel passes: one done line each, one verdict, one barrier."""
    rec = _install(monkeypatch, [_ok()] * 4)

    lwd_channels.lwd_warm_up_channels()

    assert [(dp, ch) for dp, ch, _ in rec.calls] == [
        (0, LwdChannelType.UP),
        (0, LwdChannelType.DOWN),
        (1, LwdChannelType.UP),
        (1, LwdChannelType.DOWN),
    ]
    assert len(rec.logs) == 4
    assert rec.verdicts == [True]
    assert rec.world.barrier_calls == 1
    assert rec.destroyed == 0


def test_frame_mismatch_keeps_probing_and_rolls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad frame is recorded, the remaining channels are probed, then rollback."""
    mismatch = _fail(lwd_channels._ProbeOutcome.MISMATCH, "frame mismatch: [1] peer rank mismatch")
    rec = _install(monkeypatch, [_ok(), mismatch, _ok(), _ok()], aggregate_ok=False)

    with pytest.raises(RuntimeError, match="frame mismatch"):
        lwd_channels.lwd_warm_up_channels()

    assert len(rec.calls) == 4
    assert rec.verdicts == [False]
    assert rec.destroyed == 1


def test_timeout_stops_both_loops_and_skips_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wedged channel breaks both loops and the destroy rollback is skipped."""
    timeout = _fail(lwd_channels._ProbeOutcome.TIMEOUT, "probe TIMEOUT after 30s")
    rec = _install(monkeypatch, [_ok(), timeout], aggregate_ok=False)

    with pytest.raises(RuntimeError, match="TIMEOUT"):
        lwd_channels.lwd_warm_up_channels()

    assert len(rec.calls) == 2
    assert rec.verdicts == [False]
    assert rec.destroyed == 0


def test_remote_failure_without_local_detail_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A locally clean rank still raises with the fallback text on a remote failure."""
    rec = _install(monkeypatch, [_ok()] * 4, aggregate_ok=False)

    with pytest.raises(RuntimeError, match=r"remote rank\(s\) reported probe failure"):
        lwd_channels.lwd_warm_up_channels()

    assert rec.verdicts == [False]
    assert rec.destroyed == 1


def test_non_endpoint_rank_only_aggregates_and_barriers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rank holding no endpoint skips probing but still joins verdict and barrier."""
    rec = _install(monkeypatch, [])
    monkeypatch.setattr(lwd_channels, "_LWD_ENDPOINTS", ())

    lwd_channels.lwd_warm_up_channels()

    assert rec.calls == []
    assert rec.verdicts == [True]
    assert rec.world.barrier_calls == 1
    assert rec.destroyed == 0


def test_non_hccl_backend_fails_before_posting_an_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """A degraded (gloo) channel group is rejected instead of silently probed."""
    monkeypatch.setattr(lwd_channels.dist, "get_backend", lambda group: "gloo")

    with pytest.raises(RuntimeError, match="must be hccl, got gloo"):
        lwd_channels._lwd_probe_channel(object(), peer=1, dp_idx=0, channel=LwdChannelType.UP, is_sender=True)
