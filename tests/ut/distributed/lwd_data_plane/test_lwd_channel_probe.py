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
"""CPU-only tests for the LWD channel identity probe.

Covers the three probe outcomes, runnable without NPU:
  * normal pass: both peers derive the identical frame and the receiver's
    bit-exact verify accepts it;
  * content mismatch: a flipped field, a never-written all-zero buffer or a
    wrong frame length is detected and named by the verify;
  * timeout: a never-completing op makes the poll return False within its
    budget instead of blocking forever.

Only ``dist.all_reduce`` needs mocking; real gloo MIN semantics belong to
the real-machine matrix.
"""

import pytest
import torch

from vllm_ascend.distributed.lwd_data_plane import lwd_channel_probe as probe
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType


class _FakeHandle:
    """Collective-op handle stand-in: never blocks, records whether wait() ran."""

    def __init__(self, completed: bool) -> None:
        self._completed = completed
        self.waited = False

    def is_completed(self) -> bool:
        return self._completed

    def wait(self) -> None:
        self.waited = True


# --------------------------------------------------------------------------
# 1. 正常情况通过: the frame is exact and the verify/wait accept it.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sender_rank", "dp_idx", "channel", "expected"),
    [
        (2, 0, LwdChannelType.UP, [90, 2, 0, 0, 8, 7, 11, 17]),
        (5, 3, LwdChannelType.DOWN, [90, 5, 3, 1, 8, 10, 32, 30]),
    ],
)
def test_frame_field_values(sender_rank, dp_idx, channel, expected) -> None:
    """The frame layout, including all three check sequences, is exact."""
    frame = probe.lwd_probe_frame(sender_rank, dp_idx, channel, device="cpu")
    assert frame.dtype is torch.bfloat16
    assert frame.numel() == probe._LWD_PROBE_FRAME_NUM_ELEMENTS
    assert frame.view(torch.int16).tolist() == expected


def test_bf16_representations_0_to_255_are_exact() -> None:
    """The probe's correctness basis: bf16 carries every identity value losslessly."""
    raw = torch.tensor(list(range(256)), dtype=torch.bfloat16, device="cpu")
    assert raw.view(torch.int16).tolist() == list(range(256))


def test_verify_accepts_a_correct_frame() -> None:
    """Normal case: a bit-exact frame passes with an empty diagnosis."""
    frame = probe.lwd_probe_frame(5, 3, LwdChannelType.DOWN, device="cpu")
    assert probe.lwd_probe_verify(frame, peer=5, dp_idx=3, channel=LwdChannelType.DOWN) == ""


def test_wait_returns_true_for_a_completed_op() -> None:
    """Normal case: a completed op reports True and surfaces errors via wait()."""
    handle = _FakeHandle(completed=True)
    assert probe.lwd_probe_wait(handle) is True
    assert handle.waited


# --------------------------------------------------------------------------
# 2. 内容校验失败: any wrong content is detected and named.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("index", range(8))
def test_verify_reports_every_single_field_mismatch(index) -> None:
    """Flipping any single field is detected and named."""
    frame = probe.lwd_probe_frame(5, 3, LwdChannelType.DOWN, device="cpu")
    frame.view(torch.int16)[index] += 1
    problem = probe.lwd_probe_verify(frame, peer=5, dp_idx=3, channel=LwdChannelType.DOWN)
    assert problem
    assert f"[{index}]" in problem
    assert probe._LWD_PROBE_FIELD_NAMES[index] in problem


def test_verify_rejects_a_never_written_zero_buffer() -> None:
    """A no-op irecv leaves the zero-filled buffer: fails at the magic field."""
    buffer = torch.zeros(probe._LWD_PROBE_FRAME_NUM_ELEMENTS, dtype=torch.bfloat16, device="cpu")
    problem = probe.lwd_probe_verify(buffer, peer=5, dp_idx=3, channel=LwdChannelType.DOWN)
    assert "[0]" in problem


def test_verify_rejects_the_wrong_frame_length() -> None:
    """Frames of the wrong length fail before field comparison."""
    short = torch.zeros(4, dtype=torch.bfloat16, device="cpu")
    problem = probe.lwd_probe_verify(short, peer=5, dp_idx=3, channel=LwdChannelType.UP)
    assert "length mismatch" in problem


# --------------------------------------------------------------------------
# 3. 超时失败: a wedged channel is reported within its budget.
# --------------------------------------------------------------------------


def test_wait_returns_false_on_timeout() -> None:
    """A never-completing op returns False without blocking forever."""
    handle = _FakeHandle(completed=False)
    assert probe.lwd_probe_wait(handle, timeout_s=0.05) is False
    assert not handle.waited


# --------------------------------------------------------------------------
# 4. 值域越界: identities wider than the exact bf16 range are refused up
#    front, before any probe frame is posted.
# --------------------------------------------------------------------------


def test_frame_rejects_out_of_range_identities() -> None:
    """The exact bf16 limit is a hard assert, not a silent wraparound."""
    with pytest.raises(AssertionError):
        probe.lwd_probe_frame(256, 0, LwdChannelType.UP, device="cpu")
    with pytest.raises(AssertionError):
        probe.lwd_probe_frame(0, 256, LwdChannelType.UP, device="cpu")
    # The boundary itself still works.
    probe.lwd_probe_frame(255, 255, LwdChannelType.DOWN, device="cpu")


# --------------------------------------------------------------------------
# 5. 失败聚合: one failure fails every rank (mocked all_reduce MIN).
# --------------------------------------------------------------------------


def test_aggregate_every_rank_ok() -> None:
    """All-local-ok makes the world verdict True."""
    group = object()

    def all_reduce(tensor, op=None, group=None):
        assert group is not None
        tensor.copy_(torch.ones_like(tensor))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(probe.dist, "all_reduce", all_reduce)
        assert probe.lwd_probe_aggregate(True, group) is True


def test_aggregate_a_remote_failure_fails_everywhere() -> None:
    """A rank that was locally fine still fails on a remote rank's failure."""
    group = object()

    def all_reduce(tensor, op=None, group=None):
        assert group is not None
        tensor.copy_(torch.zeros_like(tensor))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(probe.dist, "all_reduce", all_reduce)
        assert probe.lwd_probe_aggregate(True, group) is False
