# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Functional test of the LWD channel warmup probe on real NPU P2P links.

Run inside the vllm-ascend repo on a box with 2 NPUs (real HCCL P2P,
nothing stubbed):

    torchrun --nproc_per_node=2 tests/ut/distributed/lwd_data_plane/npu_functional_test.py

Ranks model a production edge-cloud pair: rank 0 = edge, rank 1 = cloud.
The warmup probes the two channels in the fixed global order (UP then DOWN):
UP edge->cloud (rank 0 sends), DOWN cloud->edge (rank 1 sends).

  [S1] healthy round-trip          -> OK on both ranks, no exception
  [S2] corrupted payload (dp_idx)  -> MISMATCH with a per-field diagnosis
  [S3] frame of the wrong length   -> MISMATCH (length checked first)
  [S4] peer never posts            -> wait() returns False on the timeout

  [O1] all channels healthy        -> one INFO "warmup done" per channel,
                                      UP then DOWN, barrier, no rollback
  [O2] corrupted sender frames     -> receiver diagnoses MISMATCH, probing
                                      continues, rollback, shared raise
  [O3] wedged channel              -> TIMEOUT aborts probing, WARNING lists
                                      the unprobed channel, no rollback, raise
  [O4] unexpected raise            -> ledgered like a TIMEOUT, no rollback,
                                      raise; the peer still sees the verdict

Every functional point prints its expected and actual result explicitly
(预期/实际) instead of a bare boolean, so the run log can go straight into
the PR verification report. A scenario counts as 符合预期 only when every
one of its points matches.

S4 leaves an unmatched recv behind -- exactly the state the production code
refuses to roll back -- so the process exits through ``os._exit``: the NPU
is reclaimed by the OS and the wedged group is never reused.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from vllm.distributed.lwd import LwdChannelEndpoints

from vllm_ascend.distributed.lwd_data_plane import lwd_channel_probe as probe
from vllm_ascend.distributed.lwd_data_plane import lwd_channels as channels
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType

_DP_IDX = 0
# Short enough to keep the run fast, long enough to separate a timeout from a
# slow-but-healthy rendezvous.
_TIMEOUT_S = 5.0
_ORCH_TIMEOUT_S = 3.0
_UP = LwdChannelType.UP
_DOWN = LwdChannelType.DOWN

_world: _StubWorldGroup | None = None
_destroyed: list[int] = []
_scenarios: list[_Scenario] = []


@dataclass
class _Check:
    """One functional point: what was expected and what actually happened."""

    point: str
    expected: str
    actual: str
    ok: bool


@dataclass
class _Scenario:
    title: str
    checks: list[_Check]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)


def _scenario(title: str, checks: list[_Check]) -> None:
    _scenarios.append(_Scenario(title, checks))


class _LogCapture(logging.Handler):
    """Collect the ``[LWD]`` lines the orchestration emits."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "[LWD]" in message:
            self.lines.append(message)

    def count(self, token: str) -> int:
        return sum(token in line for line in self.lines)


class _StubWorldGroup:
    """The two members ``lwd_warm_up_channels`` reads off the world group."""

    def __init__(self, cpu_group: dist.ProcessGroup) -> None:
        self.cpu_group = cpu_group
        self.barriers = 0

    def barrier(self) -> None:
        self.barriers += 1
        dist.barrier(group=self.cpu_group)


class _NeverDone:
    """A handle that never completes, for the wedged-channel scenario."""

    def is_completed(self) -> bool:
        return False

    def wait(self) -> None:
        raise AssertionError("wait() must not be called on a timed-out op")


def _print_report(rank: int) -> int:
    """Print the per-point 预期/实际 report; 0 exit code iff all scenarios match."""
    passed = sum(scenario.ok for scenario in _scenarios)
    print()
    print(f"==================== rank={rank} 功能测试结果 ====================")
    for scenario in _scenarios:
        status = "符合预期" if scenario.ok else "不符合预期"
        print(f"\n■ {scenario.title} —— {status}")
        for check in scenario.checks:
            mark = "✓" if check.ok else "✗"
            print(f"  {mark} {check.point}")
            print(f"      预期: {check.expected}")
            print(f"      实际: {check.actual}")
    print()
    print(f"==================== 汇总 (rank={rank}): {passed}/{len(_scenarios)} 个场景符合预期 ====================")
    return 0 if passed == len(_scenarios) else 1


def _run_orchestration(
    title: str,
    capture: _LogCapture,
    *,
    logs: dict[str, int] | None = None,
    order: tuple[str, str] | None = None,
    destroy: int = 0,
    barriers: int = 0,
    raises: str = "",
) -> None:
    """Run the real ``lwd_warm_up_channels`` and report every policy point."""
    assert _world is not None
    before_destroy = len(_destroyed)
    before_barriers = _world.barriers
    capture.lines.clear()
    raised: str | None = None
    try:
        channels.lwd_warm_up_channels()
    except Exception as exc:  # noqa: BLE001 - a raise may be the expectation
        raised = str(exc)
    checks = [
        _Check(
            "异常行为",
            f"抛出 RuntimeError 且信息含「{raises}」" if raises else "无异常抛出，正常返回",
            f"抛出: {raised}" if raised is not None else "未抛出异常",
            bool(raises and raises in raised) if raises else raised is None,
        )
    ]
    rolled_back = len(_destroyed) - before_destroy
    checks.append(_Check("回滚 destroy", f"调用 {destroy} 次", f"调用 {rolled_back} 次", rolled_back == destroy))
    barrier_count = _world.barriers - before_barriers
    checks.append(_Check("world barrier", f"调用 {barriers} 次", f"调用 {barrier_count} 次", barrier_count == barriers))
    for token, expected_count in (logs or {}).items():
        actual_count = capture.count(token)
        checks.append(
            _Check(
                f"日志「{token}」",
                f"出现 {expected_count} 次",
                f"出现 {actual_count} 次",
                actual_count == expected_count,
            )
        )
    if order is not None:
        done = [
            line.split("channel=")[1].split(",")[0].removeprefix("LwdChannelType.")
            for line in capture.lines
            if "data-plane warmup done" in line
        ]
        checks.append(
            _Check("探测顺序", " → ".join(order), " → ".join(done) or "（无 done 日志）", done == list(order))
        )
    _scenario(title, checks)


def main() -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))
    # Bind the NPU before any HCCL group is created, or init fails with
    # ERR99999 UNKNOWN on CANN.
    torch.npu.set_device(local_rank)
    # Explicit init_method instead of torchrun's env://, so a direct
    # `python npu_functional_test.py` works too.
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world,
        init_method=f"tcp://{master_addr}:{master_port}",
    )
    if world != 2:
        raise SystemExit(f"this functional test needs exactly 2 ranks, got {world}")
    peer = 1 - rank
    # A dedicated hccl group per rank pair, the same shape the data plane
    # builds for one (dp_idx, channel).
    transport = dist.new_group([0, 1], backend="hccl")

    print(f"rank={rank} local_rank={local_rank} device=npu:{local_rank} peer={peer}")
    print()

    # ------------------------------------------------------------------ S1
    # The real probe path, both directions of the pair exercised at once.
    s1_checks: list[_Check] = []
    try:
        up = channels._lwd_probe_channel(transport, peer, _DP_IDX, _UP, is_sender=rank == 0)
        s1_checks.append(
            _Check(
                "UP 通道探测",
                "返回 OK（收发配对完成，接收端帧校验通过）",
                f"outcome={up.outcome.value}",
                up.outcome is channels._ProbeOutcome.OK,
            )
        )
        down = channels._lwd_probe_channel(transport, peer, _DP_IDX, _DOWN, is_sender=rank == 1)
        s1_checks.append(
            _Check(
                "DOWN 通道探测",
                "返回 OK（收发配对完成，接收端帧校验通过）",
                f"outcome={down.outcome.value}",
                down.outcome is channels._ProbeOutcome.OK,
            )
        )
    except Exception as exc:  # noqa: BLE001 - a raise is a failing scenario
        s1_checks.append(_Check("探测执行", "正常完成，无异常", f"抛出 {exc!r}", False))
    _scenario("[S1] 通道探测 · 正常收发（UP/DOWN 双向）", s1_checks)

    # ------------------------------------------------------------------ S2
    # The sender puts the wrong dp_idx on the wire, so the receiver must
    # diagnose a crossed connection instead of accepting the frame.
    s2_checks: list[_Check] = []
    if rank == 0:
        bad = probe.lwd_probe_frame(0, _DP_IDX + 1, _UP)
        handle = dist.isend(bad, dst=peer, group=transport)
        sent = probe.lwd_probe_wait(handle, timeout_s=_TIMEOUT_S)
        s2_checks.append(_Check("毒帧发送", "isend 完成，wait 返回 True（对端已收）", f"wait 返回 {sent}", sent))
    else:
        result = channels._lwd_probe_channel(transport, peer, _DP_IDX, _UP, is_sender=False)
        diagnosed = (
            result.outcome is channels._ProbeOutcome.MISMATCH and "[2]" in result.detail and "dp_idx" in result.detail
        )
        s2_checks.append(
            _Check(
                "接收端校验",
                "MISMATCH，且诊断定位到 [2] dp_idx 字段（识别为串接连接）",
                f"outcome={result.outcome.value}；detail={result.detail}",
                diagnosed,
            )
        )
    _scenario("[S2] 通道探测 · 内容校验失败（dp_idx 串接诊断）", s2_checks)

    # ------------------------------------------------------------------ S3
    # Half a frame on the wire fails the length check before any field
    # compare, on the real link.
    s3_checks: list[_Check] = []
    if rank == 0:
        short = probe.lwd_probe_frame(0, _DP_IDX, _UP)[:4]
        handle = dist.isend(short, dst=peer, group=transport)
        sent = probe.lwd_probe_wait(handle, timeout_s=_TIMEOUT_S)
        s3_checks.append(_Check("短帧发送", "4 元素半帧发送完成，wait 返回 True", f"wait 返回 {sent}", sent))
    else:
        payload = torch.zeros(4, dtype=torch.bfloat16, device="npu")
        handle = dist.irecv(payload, src=peer, group=transport)
        if not probe.lwd_probe_wait(handle, timeout_s=_TIMEOUT_S):
            s3_checks.append(_Check("短帧接收", f"irecv 在 {_TIMEOUT_S:g}s 内完成", "irecv 超时未完成", False))
        else:
            torch.npu.synchronize()
            detail = probe.lwd_probe_verify(payload, peer, _DP_IDX, _UP)
            s3_checks.append(
                _Check(
                    "帧长度校验",
                    "报 frame length mismatch，先于任何逐字段比较",
                    detail or "（verify 未报告问题）",
                    "frame length mismatch" in detail,
                )
            )
    _scenario("[S3] 通道探测 · 帧长度错误（长度先于逐字段校验）", s3_checks)

    # --- warmup orchestration, on real channels ---------------------------
    try:
        cpu_group = dist.new_group([0, 1], backend="gloo")
    except Exception as exc:  # noqa: BLE001 - no gloo, no verdict path
        _scenario("[O*] 编排准备", [_Check("gloo 判决组建组", "成功创建（判决路径可用）", f"失败: {exc!r}", False)])
        cpu_group = None

    if cpu_group is not None:
        global _world
        _world = _StubWorldGroup(cpu_group)
        capture = _LogCapture()
        channels.logger.addHandler(capture)
        # The log assertions rely on INFO records existing; force the level
        # in case the embedding env configured vllm's logger above INFO.
        saved_log_level = channels.logger.level
        channels.logger.setLevel(logging.INFO)

        real_probe = channels._lwd_probe_channel
        real_destroy = channels.lwd_destroy_channels
        real_world_group = channels.get_world_group

        # Rank 0 = edge, rank 1 = cloud, matching a production pair.
        endpoints = LwdChannelEndpoints(dp_idx=_DP_IDX, edge_rank=0, cloud_rank=1)
        channel_groups = {
            (_DP_IDX, _UP): (transport, peer),
            (_DP_IDX, _DOWN): (transport, peer),
        }

        def install_probe(kind: str) -> None:
            """Probe behaviour override: corrupt poisons every sender frame,
            wedge makes the UP op never complete; anything else stays real."""

            def behaving(
                group: dist.ProcessGroup,
                peer_r: int,
                dp_idx: int,
                channel: LwdChannelType,
                is_sender: bool,
            ) -> channels._ChannelProbeResult:
                if kind == "corrupt" and is_sender:
                    bad = probe.lwd_probe_frame(dist.get_rank(), dp_idx + 1, channel)
                    handle = dist.isend(bad, dst=peer_r, group=group)
                    if not probe.lwd_probe_wait(handle, timeout_s=_ORCH_TIMEOUT_S):
                        return channels._ChannelProbeResult(
                            channels._ProbeOutcome.TIMEOUT,
                            channels._lwd_probe_timeout_detail(dp_idx, channel, peer_r),
                        )
                    return channels._ChannelProbeResult(channels._ProbeOutcome.OK)
                if kind == "wedge" and channel is _UP:
                    handle = _NeverDone()
                    if not probe.lwd_probe_wait(handle, timeout_s=_ORCH_TIMEOUT_S):
                        return channels._ChannelProbeResult(
                            channels._ProbeOutcome.TIMEOUT,
                            channels._lwd_probe_timeout_detail(dp_idx, channel, peer_r),
                        )
                    return channels._ChannelProbeResult(channels._ProbeOutcome.OK)
                return real_probe(group, peer_r, dp_idx, channel, is_sender)

            channels._lwd_probe_channel = behaving

        def counting_destroy() -> None:
            _destroyed.append(1)

        def stub_world_group() -> _StubWorldGroup:
            assert _world is not None
            return _world

        channels.lwd_destroy_channels = counting_destroy
        channels.get_world_group = stub_world_group
        channels._LWD_ENDPOINTS = (endpoints,)
        channels._LWD_CHANNEL_GROUPS = channel_groups

        try:
            # O1: every channel healthy on the real link, in the fixed order.
            install_probe("ok")
            _run_orchestration(
                "[O1] 预热编排 · 正常场景（全通道健康，顺序 UP→DOWN）",
                capture,
                logs={"data-plane warmup done": 2},
                order=("UP", "DOWN"),
                destroy=0,
                barriers=1,
            )

            # O2: each rank poisons the frame it sends; each receiver diagnoses
            # the mismatch, probing still walks every channel, the healthy
            # groups are rolled back, and both ranks raise the same verdict.
            install_probe("corrupt")
            _run_orchestration(
                "[O2] 预热编排 · 内容校验失败（继续探测 + 回滚 + 全员统一失败）",
                capture,
                raises="channel probe failed",
                logs={"data-plane warmup done": 1, "channel probe failed": 1},
                destroy=1,
                barriers=0,
            )

            # O3: the UP op never completes on either side -> probing stops at
            # once, the skipped channel is listed, no rollback is attempted.
            install_probe("wedge")
            _run_orchestration(
                "[O3] 预热编排 · 超时（立即停止 + 列出未探测通道 + 不回滚）",
                capture,
                raises="TIMEOUT",
                logs={"left unprobed": 1, "channel probe failed": 1},
                destroy=0,
                barriers=0,
            )

            # O4: the probe raises (OOM, broken NPU context): the exception is
            # ledgered like a TIMEOUT and the rank still reaches the aggregate,
            # so one failing rank cannot hang the other at the verdict.
            def raising_probe(*args: object) -> channels._ChannelProbeResult:
                raise MemoryError("simulated OOM in the channel probe")

            channels._lwd_probe_channel = raising_probe
            _run_orchestration(
                "[O4] 预热编排 · 意外异常（记账 + 聚合统一失败，单点不挂死对端）",
                capture,
                raises="probe raised",
                logs={"left unprobed": 1, "channel probe failed": 1},
                destroy=0,
                barriers=0,
            )
        finally:
            channels._lwd_probe_channel = real_probe
            channels.lwd_destroy_channels = real_destroy
            channels.get_world_group = real_world_group
            channels.logger.setLevel(saved_log_level)
            channels.logger.removeHandler(capture)

    # ------------------------------------------------------------------ S4
    # The peer never posts: the wait must time out instead of hanging.
    s4_checks: list[_Check] = []
    if rank == 1:
        payload = torch.zeros(8, dtype=torch.bfloat16, device="npu")
        handle = dist.irecv(payload, src=0, group=transport)
        started = time.monotonic()
        completed = probe.lwd_probe_wait(handle, timeout_s=_TIMEOUT_S)
        elapsed = time.monotonic() - started
        s4_checks.append(
            _Check(
                "等待超时行为",
                f"wait 返回 False，耗时 ≥ {_TIMEOUT_S - 0.5:g}s（预算 {_TIMEOUT_S:g}s）",
                f"返回 {completed}，耗时 {elapsed:.1f}s",
                completed is False and elapsed >= _TIMEOUT_S - 0.5,
            )
        )
    else:
        # Rank 0 stays silent so rank 1's irecv observes the full timeout.
        time.sleep(_TIMEOUT_S + 1.0)
        s4_checks.append(
            _Check("对端静默", "全程不发送任何 op（让对端 irecv 完整超时）", "已保持静默，未发送 op", True)
        )
    _scenario("[S4] 超时场景 · 对端静默（wait 返回 False）", s4_checks)

    return _print_report(rank)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip interpreter teardown: S4 leaves an unmatched recv behind, and a
    # communicator destroy on that state can hang.
    os._exit(code)
