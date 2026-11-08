"""NPU smoke test for lwd_channel_probe.

Run on the NPU box (inside the repo so the package is importable):

    python npu_smoke_probe.py

It auto-detects the device: NPU if torch_npu is available, otherwise CPU.
Running it on CPU here only proves the script itself works; the NPU path is
what matters, because the previous real-machine failure was device-specific.

Why this script exists instead of pytest:
  * the fork UT asserts through ``frame.view(torch.int16)``, which is exactly
    the call suspected of breaking on torch_npu; a CPU-only test stays green
    while the device path fails.
  * this script therefore tests the NPU path directly AND reports whether the
    legacy bit-view is trustworthy on this device.

Expected output on a healthy NPU (values are the whole point, check them):

    device: npu:0
    [1] frame(rank=0, dp=0, UP)   -> [90.0, 0.0, 0.0, 0.0, 8.0, 5.0, 11.0, 17.0]
    [2] frame(rank=0, dp=0, DOWN) -> [90.0, 0.0, 0.0, 1.0, 8.0, 5.0, 11.0, 30.0]
    [3] frame(rank=7, dp=3, UP)   -> [90.0, 7.0, 3.0, 0.0, 8.0, 12.0, 32.0, 17.0]
    [4] npu vs cpu bit-identical  -> PASS
    [5] legacy view(torch.int16)  -> <verdict>
    [6] verify ok                 -> PASS ("")
    [7] verify wrong peer         -> PASS ([1] peer rank mismatch: got 7, expected 8)
    [8] verify swapped direction  -> PASS ([3] channel direction mismatch: got 0, expected 1)
    [9] verify zero buffer        -> PASS ([0] not a probe frame ...: got 0, expected 90)
    [10] verify NaN field         -> PASS ([6] data corruption: got -1, expected 32)
    [11] 200x verify() on device  -> avg <x> ms

    total=12 passed=12 failed=0

Anything that prints FAIL points at a real device problem; section [5] is the
forensic one, it explains the historical "always mismatch" report.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import types

PKG = "vllm_ascend.distributed.lwd_data_plane"


def find_repo() -> str:
    """Locate the repo root that contains the package, without hard-coding it."""
    env = os.environ.get("VLLM_ASCEND_ROOT")
    here = os.path.dirname(os.path.abspath(__file__))
    # tools/ lives directly under the repo root; also try the cwd.
    candidates = [env, os.path.dirname(here), os.getcwd(), here]
    for cand in candidates:
        if cand and os.path.isdir(os.path.join(cand, "vllm_ascend", "distributed")):
            return cand
    raise SystemExit(
        "cannot locate the vllm-ascend repo; set VLLM_ASCEND_ROOT to its root directory"
    )


REPO = find_repo()
BASE = os.path.join(REPO, "vllm_ascend", "distributed", "lwd_data_plane")


def import_probe():
    """Normal import first (installed package), file loader as fallback."""
    try:
        from vllm_ascend.distributed.lwd_data_plane import lwd_channel_probe as probe
        from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType

        return probe, LwdChannelType
    except Exception:  # noqa: BLE001 - vllm not installed on the CPU box
        for name in ("vllm_ascend", "vllm_ascend.distributed", PKG):
            mod = types.ModuleType(name)
            mod.__path__ = [BASE]
            sys.modules[name] = mod

        def load(name, filename):
            spec = importlib.util.spec_from_file_location(name, BASE + "\\" + filename)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod

        load_types = load(PKG + ".lwd_types", "lwd_types.py")
        load_probe = load(PKG + ".lwd_channel_probe", "lwd_channel_probe.py")
        return load_probe, load_types.LwdChannelType


import torch  # noqa: E402

probe, LwdChannelType = import_probe()
UP, DOWN = LwdChannelType.UP, LwdChannelType.DOWN

DEVICE = "cpu"
try:
    import torch_npu  # noqa: F401

    if torch.npu.is_available():
        torch.npu.set_device(0)
        DEVICE = "npu:0"
except Exception:  # noqa: BLE001
    DEVICE = "cpu"

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))


def read_ints(t):
    return [int(v) for v in t.to(torch.float32).tolist()]


print(f"device: {DEVICE}")
print()

# [1]-[3] frame values on the target device.
# Expected details are printed in the docstring above.
cases = [
    ("[1] frame(rank=0, dp=0, UP)", 0, 0, UP, [90, 0, 0, 0, 8, 5, 11, 17]),
    ("[2] frame(rank=0, dp=0, DOWN)", 0, 0, DOWN, [90, 0, 0, 1, 8, 5, 11, 30]),
    ("[3] frame(rank=7, dp=3, UP)", 7, 3, UP, [90, 7, 3, 0, 8, 12, 32, 17]),
]
for name, rank, dp, ch, expected in cases:
    frame = probe.lwd_probe_frame(rank, dp, ch, device=DEVICE)
    got = read_ints(frame)
    print(f"{name:<28} -> {got}")
    check(name, got == expected, str(got))

# [4] the device-built frame must equal the CPU-built one, element by element.
dev = probe.lwd_probe_frame(7, 3, UP, device=DEVICE).to(torch.float32)
cpu = probe.lwd_probe_frame(7, 3, UP, device="cpu").to(torch.float32)
check("[4] npu vs cpu bit-identical", torch.equal(dev.cpu(), cpu), f"{read_ints(dev)} vs {read_ints(cpu)}")

# [5] forensic: what does the legacy Tensor.view(dtype) actually return here?
#     view(int16) reinterprets the bf16 bits, so 90.0 comes back as 0x42B4 =
#     17076 -- a "number nobody recognises", which is exactly what the old
#     code compared against plain integers.  Expected detail: the bit pattern
#     of the bf16 encoding must match, i.e. agrees_with_bf16_bits=True.
legacy = probe.lwd_probe_frame(7, 3, UP, device=DEVICE)
expected_vals = [90, 7, 3, 0, 8, 12, 32, 17]
expected_bits = torch.tensor(expected_vals, dtype=torch.bfloat16).view(torch.int16).tolist()
try:
    bits = legacy.view(torch.int16).tolist()
    agree = bits == expected_bits
    verdict = f"bits={bits} agrees_with_bf16_bits={agree}"
    check("[5] legacy view(torch.int16)", agree, verdict)
except Exception as exc:  # noqa: BLE001
    # An error here is a *finding*, not a failure of the probe module.
    verdict = f"raised {type(exc).__name__}: {exc}"
    results.append(("[5] legacy view(torch.int16)", False, verdict))
print(f"{'[5] legacy view(torch.int16)':<28} -> {verdict}")
print(f"{'':<28}    (bf16 bits of {expected_vals} = {expected_bits})")

# [6]-[10] verify() on the device, every outcome that matters at init time.
frame = probe.lwd_probe_frame(7, 3, UP, device=DEVICE)
ok = probe.lwd_probe_verify(frame, 7, 3, UP)
check("[6] verify ok", ok == "", repr(ok))

wrong_peer = probe.lwd_probe_verify(frame, 8, 3, UP)
check("[7] verify wrong peer", "[1]" in wrong_peer and "peer rank mismatch" in wrong_peer, wrong_peer)

swapped = probe.lwd_probe_verify(frame, 7, 3, DOWN)
check("[8] verify swapped direction", "[3]" in swapped and "direction" in swapped, swapped)

zeros = torch.zeros(8, dtype=torch.bfloat16, device=DEVICE)
zero_msg = probe.lwd_probe_verify(zeros, 7, 3, UP)
check("[9] verify zero buffer", "[0]" in zero_msg, zero_msg)

nan_frame = frame.clone()
nan_frame[6] = float("nan")
try:
    nan_msg = probe.lwd_probe_verify(nan_frame, 7, 3, UP)
    check("[10] verify NaN field", "[6]" in nan_msg and "-1" in nan_msg, nan_msg)
except Exception as exc:  # noqa: BLE001
    check("[10] verify NaN field", False, repr(exc))

# [11] cost of one verify() on the device (init-time budget sanity check).
iterations = 200
start = time.monotonic()
for _ in range(iterations):
    probe.lwd_probe_verify(frame, 7, 3, UP)
avg_ms = (time.monotonic() - start) * 1000 / iterations
print(f"{'[11] 200x verify() on device':<28} -> avg {avg_ms:.3f} ms")
check("[11] verify cost", avg_ms < 50.0, f"{avg_ms:.3f} ms")

print()
failed = [r for r in results if r[1] is not True]
for name, ok_flag, detail in results:
    if ok_flag is not True:
        print(f"FAIL  {name}  -> {detail}")
print(f"total={len(results)} passed={len(results) - len(failed)} failed={len(failed)}")
sys.exit(1 if failed else 0)
