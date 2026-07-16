"""
User-facing JIT for small double-precision matmuls on Skylake (AVX-512).

This wraps the LIBXSMM GEMM generator + PeachPy backend
(:mod:`autotuner.libxsmm_gemm.peachpy_backend`) behind a numpy-friendly API so a small
``C = A @ B`` can be compiled and called in one line, e.g.::

    from autotuner.libxsmm_gemm.jit import matmul
    C = matmul(A, B)                 # A: (m, k), B: (k, n), all float64, row-major

Public entry points:

* :func:`jit_matmul` -- compile (and cache) a reusable kernel for a given ``(m, n, k)`` shape.
* :func:`matmul` -- convenience that infers the shape and returns ``A @ B``.
* :func:`is_available` -- whether JIT execution is possible on this interpreter (x86-64 only).

Conventions / scope (double precision, ``Arch.LIBXSMM_X86_AVX512_SKX``):

* **Row-major** ordinary C-order numpy arrays. The kernel is built with swapped dims
  ``(n, m, k)`` and called with swapped A/B pointers (``kernel(B, A, C)``) -- the repo's
  validated row-major convention (Snakefile ``libxsmm_rowmaj_c`` / ``ref_matmul``).
* Any ``m, n, k >= 1`` (``n`` not a multiple of 8 uses a k1-masked remainder). Unaligned
  loads/stores, so inputs need no special alignment. Result is ``alpha = 1`` (fixed).
* ``matmul(A, B)`` / ``out=None`` computes a fresh ``C = A @ B``; passing ``out=`` accumulates
  in place (``out += A @ B``, i.e. ``beta = 1`` -- the only reliable generator path).

Execution requires an x86-64 interpreter with AVX-512; :func:`jit_matmul` raises
:class:`RuntimeError` otherwise (via :func:`~autotuner.libxsmm_gemm.peachpy_backend.build_callable`).
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from functools import lru_cache

import numpy as np
import peachpy.x86_64.abi

from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_macros import gemm_flags
from autotuner.libxsmm_gemm.libxsmm_main import (
    DescDatatype,
    GEMMDescriptor,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype
from autotuner.libxsmm_gemm.peachpy_backend import build_callable

_ARCH = Arch.LIBXSMM_X86_AVX512_SKX
_PTR_DOUBLE = ctypes.POINTER(ctypes.c_double)


def _has_avx512f() -> bool:
    """Whether the host CPU supports AVX-512F.

    The generated kernels use AVX-512 (``zmm``/EVEX); executing them on a CPU without AVX-512
    faults with SIGILL (uncatchable), so this must be checked before *calling* a kernel -- being
    x86-64 is not enough (many x86-64 CPUs, including common CI runners, lack AVX-512)."""
    try:
        import numpy as np

        features = getattr(np._core._multiarray_umath, "__cpu_features__", {})
        if "AVX512F" in features:
            return bool(features["AVX512F"])
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as cpuinfo:  # Linux fallback
            return "avx512f" in cpuinfo.read()
    except OSError:
        return False


def is_available() -> bool:
    """Whether compiled kernels can be *executed* here: an x86-64 interpreter on a CPU with
    AVX-512F. Building/encoding a kernel works on any x86-64 host, but calling one requires
    AVX-512 (else SIGILL)."""
    return peachpy.x86_64.abi.detect() is not None and _has_avx512f()


def _as_kernel_arg(arr: np.ndarray) -> ctypes._Pointer:
    return arr.ctypes.data_as(_PTR_DOUBLE)


@lru_cache(maxsize=None)
def jit_matmul(m: int, n: int, k: int) -> Callable[..., np.ndarray]:
    """
    Compile a reusable row-major double-precision matmul kernel for a fixed ``(m, n, k)``.

    Returns a callable ``kernel(a, b, out=None) -> np.ndarray`` where ``a`` is ``(m, k)``, ``b``
    is ``(k, n)`` and the result is ``(m, n)``. With ``out=None`` a fresh ``C = A @ B`` is
    returned; passing ``out`` accumulates in place (``out += A @ B``).

    Results are cached per shape. Raises :class:`RuntimeError` unless the host is an x86-64 CPU
    with AVX-512 (the generated kernels would otherwise fault with SIGILL when called).
    """
    if not is_available():
        raise RuntimeError(
            "JIT matmul requires an x86-64 CPU with AVX-512 support "
            "(the generated kernels use AVX-512 and would fault with SIGILL otherwise)."
        )
    if not (m >= 1 and n >= 1 and k >= 1):
        raise ValueError(f"matmul dims must be >= 1, got (m={m}, n={n}, k={k})")

    dt = Datatype.F64
    # Row-major trick: build the column-major kernel with swapped (m, n) dims and leading
    # dims, then call it with the A/B pointers swapped.
    desc = GEMMDescriptor(
        m=n,
        n=m,
        k=k,
        lda=n,
        ldb=k,
        ldc=n,
        datatype=DescDatatype(dt, dt, dt, dt),
        flags=gemm_flags("N", "N"),  # no transpose, beta=1, unaligned
        prefetch=GEMMPrefetchType.NONE,
    )
    kernel_fn = build_callable(f"matmul_{m}_{n}_{k}", desc, _ARCH)

    def kernel(
        a: np.ndarray, b: np.ndarray, out: np.ndarray | None = None
    ) -> np.ndarray:
        # Read-only inputs: coerce to C-contiguous float64 (may copy; harmless).
        a = np.ascontiguousarray(a, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        if a.shape != (m, k):
            raise ValueError(f"a must have shape {(m, k)}, got {a.shape}")
        if b.shape != (k, n):
            raise ValueError(f"b must have shape {(k, n)}, got {b.shape}")

        if out is None:
            # Fresh result: kernel accumulates (beta=1), so start from zero.
            out = np.zeros((m, n), dtype=np.float64)
        else:
            # In-place accumulate: must write into the caller's buffer, so require it be
            # directly usable (a copy would silently drop the accumulation).
            if out.shape != (m, n):
                raise ValueError(f"out must have shape {(m, n)}, got {out.shape}")
            if out.dtype != np.float64:
                raise ValueError(f"out must be float64, got {out.dtype}")
            if not (out.flags["C_CONTIGUOUS"] and out.flags["WRITEABLE"]):
                raise ValueError("out must be a writable C-contiguous array")

        # System V args rdi, rsi, rdx = (A', B', C'); with the row-major swap A' = b, B' = a.
        kernel_fn(_as_kernel_arg(b), _as_kernel_arg(a), _as_kernel_arg(out))
        return out

    return kernel


def matmul(a: np.ndarray, b: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """
    Compute a row-major double-precision ``a @ b`` (shapes ``(m, k)`` and ``(k, n)``) using a
    JIT-compiled kernel. With ``out`` given, accumulates in place (``out += a @ b``).

    Convenience wrapper around :func:`jit_matmul`; the compiled kernel is cached per shape.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"a and b must be 2-D, got {a.ndim}-D and {b.ndim}-D")
    m, ka = a.shape
    kb, n = b.shape
    if ka != kb:
        raise ValueError(f"inner dimensions must match, got {a.shape} @ {b.shape}")
    return jit_matmul(m, n, ka)(a, b, out)
