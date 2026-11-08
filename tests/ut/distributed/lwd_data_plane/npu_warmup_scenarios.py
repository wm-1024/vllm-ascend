"""Two-rank NPU functional test: channel probe + warmup orchestration.

Run inside the repo on a box with >= 2 NPUs (real HCCL P2P, nothing stubbed):

    torchrun --nproc_per_node=2 tests/ut/distributed/lwd_data_plane/npu_warmup_scenarios.py

Transport layer, rank 0 as edge sender / rank 1 as cloud receiver:

  [S1] healthy round-trip          -> OK on both ranks, no exception
  [S2] corrupted payload (dp_idx)  -> MISMATCH with a per-field diagnosis
  [S3] peer never posts            -> wait() returns False on the timeout

Warmup orchestration (``lwd_warm_up_channels``) with real HCCL channels:

  [O1] all channels healthy        -> one INFO "warmup done" per channel,
                                      barrier, no rollback, no raise
  [O2] one frame corrupted         -> ERROR "channel probe failed" with the
                                      mismatch detail, rollback 1, raise;
                                      the remaining channel is still probed
  [O3] wedged channel              -> WARNING listing the unprobed channel,
                                      ERROR with TIMEOUT, no rollback, raise

S3 and O3 leave unmatched recvs behind -- exactly the state the production
code refuses to roll back -- so the process exits through ``os._exit``: the
NPU is reclaimed by the OS and the wedged groups are never reused.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import time

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

_results: list[tuple[str, bool, str]] = []
_world: _StubWorldGroup | None = None
_destroyed: list[int] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


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


def _run_orchestration(name: str, capture: _LogCapture, **expect) -> None:
    """Call the real ``lwd_warm_up_channels`` and check logs + policy."""
    capture.lines.clear()
    before = (len(_destroyed), _world.barriers)
    try:
        channels.lwd_warm_up_channels()
        raised = ""
    except RuntimeError as exc:
        raised = str(exc)
    after = (len(_destroyed), _world.barriers)
    problems: list[str] = []

    wanted_raise = expect.get("raise_token", "")
    if wanted_raise and wanted_raise not in raised:
        problems.append(f"expected RuntimeError with {wanted_raise!r}, got {raised or None!r}")
    if not wanted_raise and raised:
        problems.append(f"unexpected RuntimeError: {raised}")
    for token, count in expect.get("logs", {}).items():
        got = capture.count(token)
        if got != count:
            problems.append(f"log {token!r}: expected {count}, got {got}")
    if expect.get("order"):
        first, second = expect["order"]
        ordered = [line for line in capture.lines if "data-plane warmup done" in line]
        if len(ordered) != 2 or first not in ordered[0] or second not in ordered[1]:
            problems.append(f"warmup log order wrong: {ordered}")

    destroys = after[0] - before[0]
    if destroys != expect.get("destroys", 0):
        problems.append(f"destroy calls: expected {expect.get('destroys', 0)}, got {destroys}")
    barriers = after[1] - before[1]
    if barriers != expect.get("barriers", 0):
        problems.append(f"barriers: expected {expect.get('barriers', 0)}, got {barriers}")

    record(name, not problems, "; ".join(problems) or f"raised={raised or None!r}")


def main() -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 2:
        raise SystemExit(f"this scenario needs exactly 2 ranks, got {world}")
    torch.npu.set_device(local_rank)
    peer = 1 - rank
    # A dedicated hccl group per rank pair, the same shape the data plane
    # builds for one (dp_idx, channel).
    transport = dist.new_group([0, 1], backend="hccl")

    print(f"rank={rank} local_rank={local_rank} device=npu:{local_rank} peer={peer}")
    print()

    # S1: the real probe path, both directions of the pair exercised at once.
    try:
        result = channels._lwd_probe_channel(transport, peer, _DP_IDX, _UP, is_sender=rank == 0)
        record("[S1] healthy round-trip", result.outcome is channels._ProbeOutcome.OK, result.detail)
    except Exception as exc:  # noqa: BLE001 - a raise is a failing scenario
        record("[S1] healthy round-trip", False, repr(exc))

    # S2: the sender puts the wrong dp_idx on the wire, so the receiver must
    # diagnose a crossed connection instead of accepting the frame.
    if rank == 0:
        bad = probe.lwd_probe_frame(0, _DP_IDX + 1, _UP)
        handle = dist.isend(bad, dst=peer, group=transport)
        record("[S2] corrupted payload / sender", probe.lwd_probe_wait(handle), "sent dp_idx+1")
    else:
        result = channels._lwd_probe_channel(transport, peer, _DP_IDX, _UP, is_sender=False)
        diagnosed = result.outcome is channels._ProbeOutcome.MISMATCH and "frame mismatch" in result.detail
        record("[S2] corrupted payload / receiver", diagnosed, result.detail)

    # --- warmup orchestration, on real channels ---------------------------
    try:
        cpu_group = dist.new_group([0, 1], backend="gloo")
    except Exception as exc:  # noqa: BLE001 - no gloo, no verdict path
        record("[O*] orchestration setup", False, f"cannot create the gloo cpu group: {exc!r}")
        cpu_group = None

    group_up = dist.new_group([0, 1], backend="hccl")
    group_down = dist.new_group([0, 1], backend="hccl")

    if cpu_group is not None:
        global _world, _destroyed
        _world = _StubWorldGroup(cpu_group)
        _destroyed = []
        capture = _LogCapture()
        channels.logger.addHandler(capture)
        channels._LWD_ENDPOINTS = (LwdChannelEndpoints(dp_idx=_DP_IDX, edge_rank=0, cloud_rank=1),)
        channels._LWD_CHANNEL_GROUPS = {
            (_DP_IDX, _UP): (group_up, peer),
            (_DP_IDX, _DOWN): (group_down, peer),
        }
        channels.get_world_group = lambda: _world
        channels.lwd_destroy_channels = lambda: _destroyed.append(1)

        # O1: healthy -- one INFO per channel, in the fixed UP-then-DOWN order.
        _run_orchestration(
            "[O1] orchestration / all healthy",
            capture,
            logs={"data-plane warmup done": 2},
            order=("LwdChannelType.UP", "LwdChannelType.DOWN"),
            barriers=1,
            destroys=0,
        )

        # O2: the UP sender (rank 0) puts the wrong dp_idx on the wire. The
        # receiver records MISMATCH, probing continues to DOWN so the op count
        # stays invariant, the verdict fails everywhere, and the groups roll
        # back for a clean restart.
        if rank == 0:
            healthy_frame = channels.lwd_probe_frame

            def corrupted_frame(sender_rank: int, dp_idx: int, channel: LwdChannelType, device: str = "npu") -> torch.Tensor:
                return healthy_frame(sender_rank, dp_idx + 1, channel, device=device)

            channels.lwd_probe_frame = corrupted_frame
        done_logs = 2 if rank == 0 else 1  # rank 0's own probes are all OK
        _run_orchestration(
            "[O2] orchestration / frame mismatch",
            capture,
            raise_token="frame mismatch",
            logs={"data-plane warmup done": done_logs, "channel probe failed": 1},
            destroys=1,
            barriers=0,
        )
        channels.lwd_probe_frame = probe.lwd_probe_frame if rank == 0 else channels.lwd_probe_frame

        # O3: rank 0 never posts (a handle that never completes), rank 1's
        # recv times out; both abort at once, skip the rollback and name the
        # channel that was left unprobed.
        channels.lwd_probe_wait = functools.partial(probe.lwd_probe_wait, timeout_s=_ORCH_TIMEOUT_S)
        channels.dist.isend = lambda *args, **kwargs: _NeverDone()
        _run_orchestration(
            "[O3] orchestration / wedged channel",
            capture,
            raise_token="TIMEOUT",
            logs={"channel probe failed": 1, "left unprobed": 1},
            destroys=0,
            barriers=0,
        )

    # S3 last: it leaves an unmatched recv on the transport communicator.
    if rank == 1:
        payload = torch.zeros_like(probe.lwd_probe_frame(peer, _DP_IDX, _UP))
        handle = dist.irecv(payload, src=peer, group=transport)
        started = time.monotonic()
        completed = probe.lwd_probe_wait(handle, timeout_s=_TIMEOUT_S)
        elapsed = time.monotonic() - started
        record(
            "[S3] peer never posts / timeout",
            completed is False and elapsed >= _TIMEOUT_S - 0.5,
            f"wait returned {completed} after {elapsed:.1f}s (expected False at {_TIMEOUT_S:g}s)",
        )

    failed = [name for name, ok, _ in _results if not ok]
    print()
    print(f"rank={rank} total={len(_results)} passed={len(_results) - len(failed)} failed={len(failed)}")
    for name in failed:
        print(f"FAILED: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip interpreter teardown: S3/O3 leave unmatched recvs behind, and a
    # communicator destroy on that state can hang.
    os._exit(code)
