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

The probe module is a pure leaf: frame/poison construction, bit-exact
verify and the timeout poll run on ``device="cpu"`` without NPU.  Only
``dist.all_reduce`` is mocked; the real gloo MIN semantics belong to the
real-machine matrix.
"""

import ast
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from vllm_ascend.distributed.lwd_data_plane import lwd_channel_probe as probe
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType

_MODULE_PATH = Path(probe.__file__)


@pytest.mark.parametrize(
    ("sender_rank", "dp_idx", "channel", "expected"),
    [
        (2, 0, LwdChannelType.UP, [90, 2, 0, 0, 8, 7, 11, 17]),
        (5, 3, LwdChannelType.DOWN, [90, 5, 3, 1, 8, 10, 32, 30]),
        (0, 0, LwdChannelType.DOWN, [90, 0, 0, 1, 8, 5, 11, 30]),
    ],
)
def test_frame_field_values(sender_rank, dp_idx, channel, expected) -> None:
    """The frame layout, including all three check sequences, is exact."""
    frame = probe.lwd_probe_frame(sender_rank, dp_idx, channel, device="cpu")
    assert frame.dtype is torch.bfloat16
    assert frame.numel() == probe._LWD_PROBE_FRAME_NUM_ELEMENTS
    assert frame.view(torch.int16).tolist() == expected


def test_frame_is_bit_exact_and_end_independent() -> None:
    """Both ends build the identical frame from the same public facts."""
    for channel in LwdChannelType:
        local = probe.lwd_probe_frame(7, 2, channel, device="cpu")
        remote = probe.lwd_probe_frame(7, 2, channel, device="cpu")
        assert local.view(torch.int16).tolist() == remote.view(torch.int16).tolist()


def test_frame_rejects_values_beyond_exact_bf16_range() -> None:
    """Fields wider than the exact bf16 integer range fail fast."""
    with pytest.raises(AssertionError, match="sender_rank=256"):
        probe.lwd_probe_frame(256, 0, LwdChannelType.UP, device="cpu")
    with pytest.raises(AssertionError, match="dp_idx=256"):
        probe.lwd_probe_frame(0, 256, LwdChannelType.UP, device="cpu")
    with pytest.raises(AssertionError):
        probe.lwd_probe_frame(-1, 0, LwdChannelType.UP, device="cpu")


def test_poison_is_a_fixed_bit_pattern_no_legal_frame_matches() -> None:
    """Poison bit pattern is 0x5A5A and never equals a valid frame."""
    buffer = probe.lwd_probe_poison(device="cpu")
    assert buffer.dtype is torch.bfloat16
    bits = buffer.view(torch.int16).tolist()
    assert bits == [probe._LWD_PROBE_POISON_BITS] * probe._LWD_PROBE_FRAME_NUM_ELEMENTS
    frame = probe.lwd_probe_frame(0, 0, LwdChannelType.UP, device="cpu")
    assert not torch.equal(buffer, frame)


def test_verify_accepts_a_correct_frame() -> None:
    """Receiver independently derives the same frame and accepts it."""
    frame = probe.lwd_probe_frame(5, 3, LwdChannelType.DOWN, device="cpu")
    assert probe.lwd_probe_verify(frame, peer=5, dp_idx=3, channel=LwdChannelType.DOWN) == ""


@pytest.mark.parametrize("index", range(8))
def test_verify_reports_every_single_field_mismatch(index) -> None:
    """Flipping any single field is detected and named."""
    frame = probe.lwd_probe_frame(5, 3, LwdChannelType.DOWN, device="cpu")
    frame.view(torch.int16)[index] += 1
    problem = probe.lwd_probe_verify(frame, peer=5, dp_idx=3, channel=LwdChannelType.DOWN)
    assert problem
    assert f"[{index}]" in problem
    assert probe._LWD_PROBE_FIELD_NAMES[index] in problem


def test_verify_rejects_a_poisoned_buffer() -> None:
    """A no-op recv leaves poison and fails at the magic field."""
    buffer = probe.lwd_probe_poison(device="cpu")
    problem = probe.lwd_probe_verify(buffer, peer=5, dp_idx=3, channel=LwdChannelType.DOWN)
    assert "[0]" in problem


def test_verify_rejects_the_wrong_frame_length() -> None:
    """Frames of the wrong length fail before field comparison."""
    short = torch.zeros(4, dtype=torch.bfloat16, device="cpu")
    problem = probe.lwd_probe_verify(short, peer=5, dp_idx=3, channel=LwdChannelType.UP)
    assert "length mismatch" in problem


class _FakeHandle:
    def __init__(self, completed: bool) -> None:
        self._completed = completed
        self.waited = False

    def is_completed(self) -> bool:
        return self._completed

    def wait(self) -> None:
        self.waited = True


def test_wait_returns_true_for_a_completed_op() -> None:
    """A completed op returns True and surfaces errors via wait()."""
    handle = _FakeHandle(completed=True)
    assert probe.lwd_probe_wait(handle) is True
    assert handle.waited


def test_wait_returns_false_on_timeout() -> None:
    """A never-completing op returns False without blocking forever."""
    handle = _FakeHandle(completed=False)
    assert probe.lwd_probe_wait(handle, timeout_s=0.05) is False
    assert not handle.waited


def test_aggregate_all_ranks_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """MIN over all 1s stays 1."""

    def fake_all_reduce(tensor, op, group):  # noqa: ANN001
        assert op is dist.ReduceOp.MIN
        assert group is sentinel_group

    monkeypatch.setattr(probe.dist, "all_reduce", fake_all_reduce)
    sentinel_group = object()
    assert probe.lwd_probe_aggregate(True, group=sentinel_group) is True


def test_aggregate_a_remote_failure_fails_everywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    """One remote 0 drops the MIN to 0 everywhere."""

    def fake_all_reduce(tensor, op, group):  # noqa: ANN001
        tensor.fill_(0)

    monkeypatch.setattr(probe.dist, "all_reduce", fake_all_reduce)
    assert probe.lwd_probe_aggregate(True, group=None) is False


def test_public_surface_is_the_five_probe_functions() -> None:
    """__all__ exposes exactly the intended public API."""
    assert probe.__all__ == [
        "LWD_PROBE_TIMEOUT_S",
        "lwd_probe_aggregate",
        "lwd_probe_frame",
        "lwd_probe_poison",
        "lwd_probe_verify",
        "lwd_probe_wait",
    ]


def test_module_is_a_leaf() -> None:
    """Only stdlib + torch (+ torch.distributed) + lwd_types at module scope."""
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "time", "typing", "torch", "vllm_ascend"}, imported
