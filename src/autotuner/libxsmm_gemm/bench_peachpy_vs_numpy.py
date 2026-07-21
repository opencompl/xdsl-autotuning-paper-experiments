"""
Benchmark the PeachPy-JIT'd f32 64x64x64 GEMM microkernel against NumPy matmul.

Kernel generation / encoding / loading is done once up front and excluded from
timing. Timed regions include Python call overhead into the ctypes-loaded
kernel (or into ``np.matmul``), but assume correctly shaped Fortran-order
float32 arrays -- no shape checks or copies inside the hot path.

Semantics note: the PeachPy kernel is beta=1 (``C += A @ B``); NumPy is timed
as ``np.matmul(a, b, out=c)`` (``C = A @ B``) because NumPy has no
zero-allocation accumulating GEMM without SciPy. The extra beta term is
``O(M*N)`` vs ``O(2*M*N*K)`` for the matmul itself (<1% of FLOPs at 64^3), so
it does not materially affect the comparison.

Run on an x86-64 host (the kernel is JIT-loaded and called directly)::

    python -m autotuner.libxsmm_gemm.bench_peachpy_vs_numpy
    python -m autotuner.libxsmm_gemm.bench_peachpy_vs_numpy --aligned
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import sys
import timeit

import numpy as np
import peachpy.x86_64.abi

from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_macros import gemm_flags
from autotuner.libxsmm_gemm.libxsmm_main import (
    DescDatatype,
    GEMMDescriptor,
    GEMMFlag,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype
from autotuner.libxsmm_gemm.peachpy_backend import build_callable

_M = _N = _K = 64
_FLOPS = 2 * _M * _N * _K  # multiply-adds


def _aligned_f32(shape: tuple[int, ...], rng: np.random.Generator, alignment: int = 64):
    """A 64-byte-aligned column-major (Fortran-order) float32 array of random data.

    Over-allocate a byte buffer, slice at the first aligned offset, and reshape as a
    *view* (Fortran order) -- never ``asfortranarray``, which would copy into a fresh,
    unaligned buffer.
    """
    count = int(np.prod(shape))
    itemsize = np.dtype(np.float32).itemsize
    buf = np.empty(count * itemsize + alignment, dtype=np.uint8)
    offset = (-buf.ctypes.data) % alignment
    arr = (
        buf[offset : offset + count * itemsize]
        .view(np.float32)
        .reshape(shape, order="F")
    )
    assert arr.ctypes.data % alignment == 0
    arr[...] = rng.random(count, dtype=np.float32).reshape(shape, order="F")
    return arr


def _plain_f32(shape: tuple[int, ...], rng: np.random.Generator):
    """An unaligned Fortran-order float32 array of random data."""
    return np.asfortranarray(rng.random(shape, dtype=np.float32))


def _format_time(seconds: float) -> str:
    if seconds >= 1e-3:
        return f"{seconds * 1e6:.2f} us"
    return f"{seconds * 1e9:.2f} ns"


def _gflops(seconds_per_call: float) -> float:
    return _FLOPS / seconds_per_call / 1e9


def _report(name: str, times_per_call: list[float]) -> dict[str, float]:
    t_min = min(times_per_call)
    t_med = statistics.median(times_per_call)
    t_mean = statistics.fmean(times_per_call)
    print(f"{name}:")
    print(f"  min    {_format_time(t_min):>12}  ({_gflops(t_min):.2f} GFLOP/s)")
    print(f"  median {_format_time(t_med):>12}  ({_gflops(t_med):.2f} GFLOP/s)")
    print(f"  mean   {_format_time(t_mean):>12}  ({_gflops(t_mean):.2f} GFLOP/s)")
    return {"min": t_min, "median": t_med, "mean": t_mean}


def _bench(
    timer: timeit.Timer, *, warmup: int, number: int, repeat: int
) -> list[float]:
    timer.timeit(number=warmup)  # discarded
    totals = timer.repeat(repeat=repeat, number=number)
    return [t / number for t in totals]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark PeachPy GEMM vs NumPy for f32 64x64x64."
    )
    parser.add_argument(
        "--aligned",
        action="store_true",
        help="Use 64-byte-aligned buffers and ALIGN_A/ALIGN_C kernel flags.",
    )
    parser.add_argument(
        "--number",
        type=int,
        default=2000,
        help="Calls per timing sample (default: 2000).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=11,
        help="Number of timing samples (default: 11).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Warmup calls before timing (default: 100).",
    )
    args = parser.parse_args(argv)

    if peachpy.x86_64.abi.detect() is None:
        print(
            "error: this benchmark JIT-loads and calls the kernel directly, "
            "which requires an x86-64 host; run it on CI's ubuntu-latest or an "
            "x86-64 workstation.",
            file=sys.stderr,
        )
        return 1

    flags = gemm_flags("N", "N")
    if args.aligned:
        flags |= GEMMFlag.ALIGN_A | GEMMFlag.ALIGN_C

    dt = Datatype.F32
    desc = GEMMDescriptor(
        m=_M,
        n=_N,
        k=_K,
        lda=_M,
        ldb=_K,
        ldc=_M,
        datatype=DescDatatype(dt, dt, dt, dt),
        flags=flags,
        prefetch=GEMMPrefetchType.NONE,
    )

    # Generation / encoding / load -- excluded from timing.
    kernel = build_callable(
        "bench_matmul_f32_64x64x64", desc, Arch.LIBXSMM_X86_AVX512_SKX
    )

    rng = np.random.default_rng(0)
    alloc = (
        (lambda shape: _aligned_f32(shape, rng))
        if args.aligned
        else (lambda shape: _plain_f32(shape, rng))
    )
    a = alloc((_M, _K))
    b = alloc((_K, _N))
    c = alloc((_M, _N))

    # PeachPy declares ptr(double_) args (ABI address width only); reinterpret
    # float32 buffer addresses through POINTER(c_double) once, outside timing.
    ptr_double = ctypes.POINTER(ctypes.c_double)
    pa = a.ctypes.data_as(ptr_double)
    pb = b.ctypes.data_as(ptr_double)
    pc = c.ctypes.data_as(ptr_double)

    # One-time correctness check (not timed): C += A @ B.
    c_before = c.copy(order="F")
    kernel(pa, pb, pc)
    expected = c_before + a @ b
    if not np.allclose(c, expected, rtol=1e-4, atol=1e-5):
        print(
            "error: PeachPy kernel failed correctness check vs NumPy.", file=sys.stderr
        )
        max_abs = float(np.max(np.abs(c - expected)))
        print(f"  max abs error: {max_abs}", file=sys.stderr)
        return 1
    print(f"correctness ok (aligned={args.aligned}, M=N=K={_M}, dtype=float32, beta=1)")

    kernel_timer = timeit.Timer(
        stmt="kernel(pa, pb, pc)",
        globals={"kernel": kernel, "pa": pa, "pb": pb, "pc": pc},
    )
    numpy_timer = timeit.Timer(
        stmt="np.matmul(a, b, out=c)",
        globals={"np": np, "a": a, "b": b, "c": c},
    )

    print(f"timing: warmup={args.warmup}, number={args.number}, repeat={args.repeat}")
    peachpy_times = _bench(
        kernel_timer, warmup=args.warmup, number=args.number, repeat=args.repeat
    )
    numpy_times = _bench(
        numpy_timer, warmup=args.warmup, number=args.number, repeat=args.repeat
    )

    peachpy_stats = _report("peachpy", peachpy_times)
    numpy_stats = _report("numpy", numpy_times)

    for label in ("min", "median", "mean"):
        speedup = numpy_stats[label] / peachpy_stats[label]
        print(f"speedup (numpy/peachpy, {label}): {speedup:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
