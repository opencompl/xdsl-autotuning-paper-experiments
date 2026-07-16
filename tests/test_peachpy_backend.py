"""
Tests for the PeachPy backend that assembles generated x86 GEMM microkernels.

Two levels of verification:

* :func:`test_encode_matches_reference` -- the primary gate, runs on any host (including this
  arm64 Mac). It encodes the ``16x3x5`` DP kernel to machine code, disassembles it, and
  asserts the instruction stream matches the reference Intel-syntax assembly committed in
  ``tests/filecheck/libxsmm/16x3x5_skx_dp.mlir`` (the same text produced by
  ``xdsl-opt … -t x86-asm``).
* :func:`test_execution` -- x86-64 only (``xfail`` elsewhere). It loads the kernel and checks
  the numerical result against a NumPy reference.
"""

from __future__ import annotations

import re
from pathlib import Path

import capstone
import peachpy.x86_64.abi
import pytest
from peachpy.x86_64 import abi

from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_macros import gemm_flags
from autotuner.libxsmm_gemm.libxsmm_main import DescDatatype, GEMMDescriptor, GEMMFlag
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype
from autotuner.libxsmm_gemm.peachpy_backend import build_callable, build_function

# The worked example: matmul_bac 16 3 5 16 5 16 1 1 1 1 skx nopf DP
# (M, N, K, lda, ldb, ldc, alpha, beta, align_a, align_c).
_M, _N, _K = 16, 3, 5
_LDA, _LDB, _LDC = 16, 5, 16
_ROUTINE = "matmul_bac"
_REFERENCE_MLIR = Path(__file__).parent / "filecheck" / "libxsmm" / "16x3x5_skx_dp.mlir"


def _desc() -> GEMMDescriptor:
    dt = Datatype.F64
    flags = gemm_flags("N", "N") | GEMMFlag.ALIGN_A | GEMMFlag.ALIGN_C
    return GEMMDescriptor(
        m=_M,
        n=_N,
        k=_K,
        lda=_LDA,
        ldb=_LDB,
        ldc=_LDC,
        datatype=DescDatatype(dt, dt, dt, dt),
        flags=flags,
        prefetch="nopf",
    )


# region: assembly normalization


def _canon_int(token: str) -> int:
    """Parse a decimal/hex token as a signed 64-bit integer (capstone prints two's
    complement, e.g. ``0xffffffffffffffc0`` for ``-64``)."""
    value = int(token, 0)
    if value >= 2**63:
        value -= 2**64
    return value


def _canon_operand(operand: str) -> str:
    """Canonicalize a single operand so that capstone's and the reference's spelling agree
    (strip ``zmmword ptr`` size hints, hex-vs-decimal, and internal spacing)."""
    operand = operand.strip()
    # Drop capstone's operand-size hint prefix, e.g. "zmmword ptr [rdx]" -> "[rdx]".
    operand = re.sub(r"^\w+ ptr ", "", operand)

    if operand.startswith("["):
        inner = operand[1:-1].strip()
        match = re.fullmatch(r"(\w+)\s*([+-]\s*(?:0x[0-9a-f]+|\d+))?", inner)
        assert match is not None, f"unparsed memory operand: {operand!r}"
        base = match.group(1)
        offset = match.group(2)
        if offset:
            value = _canon_int(offset.replace(" ", ""))
            if value:
                return f"[{base}{value:+d}]"
        return f"[{base}]"

    if re.fullmatch(r"[a-z][a-z0-9]*", operand):
        # A register name.
        return operand

    # An immediate.
    return str(_canon_int(operand))


def _canon_instruction(mnemonic: str, operands: str) -> str:
    mnemonic = mnemonic.strip().lower()
    operands = operands.strip().lower()
    if mnemonic.startswith("j"):
        # Jump target is a label in the reference but an absolute address in capstone's
        # output; canonicalize so the comparison checks mnemonic + presence, not the target
        # spelling (loop structure is guaranteed by the IR verifiers).
        return f"{mnemonic} <target>"
    if not operands:
        return mnemonic
    canon = ",".join(_canon_operand(part) for part in operands.split(","))
    return f"{mnemonic} {canon}"


def _reference_instructions() -> list[str]:
    """The normalized instruction stream from the first RUN block's ``// CHECK`` lines."""
    instructions: list[str] = []
    for line in _REFERENCE_MLIR.read_text().splitlines():
        # Only the first RUN block's plain `CHECK`/`CHECK-NEXT` lines match; the
        # `CHECK-IR-LIBXSMM` IR-check block uses a different prefix and is ignored.
        match = re.match(r"^//\s*CHECK(?:-NEXT)?:\s*(.*)$", line)
        if match is None:
            continue
        text = match.group(1).strip()
        if not text or text.startswith(".") or text.endswith(":"):
            # Skip directives (.text, .globl, …) and label definitions (matmul_bac:, l33:).
            continue
        mnemonic, _, operands = text.partition(" ")
        instructions.append(_canon_instruction(mnemonic, operands))
    return instructions


def _encoded_instructions() -> list[str]:
    """The normalized instruction stream of the PeachPy-encoded kernel.

    ``build_function`` returns an un-finalized ``Function`` on any host; we finalize with the
    explicit System V ABI object to obtain the byte stream (``abi.detect()`` is ``None`` on a
    non-x86 interpreter). PeachPy's epilogue emits a ``vzeroupper`` (because the body uses
    AVX-512 registers) that the xDSL reference does not -- drop it so the streams line up. Both
    sides emit the doubled ``push``/``pop rbp`` (xDSL's prologue-epilogue pass and PeachPy's
    callee-save handling)."""
    function = build_function(_ROUTINE, _desc(), Arch.LIBXSMM_X86_AVX512_SKX)
    encoded = function.finalize(abi.system_v_x86_64_abi).encode()
    code = bytes(encoded.code_section.content)

    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.syntax = capstone.CS_OPT_SYNTAX_INTEL
    instructions: list[str] = []
    for insn in md.disasm(code, 0):
        if insn.mnemonic == "vzeroupper":
            continue
        instructions.append(_canon_instruction(insn.mnemonic, insn.op_str))
    return instructions


# endregion


def test_encode_matches_reference():
    assert _encoded_instructions() == _reference_instructions()


@pytest.mark.xfail(
    not peachpy.x86_64.abi.detect(),
    reason="direct calling requires an x86-64 interpreter",
    strict=True,
)
def test_execution():
    import ctypes

    import numpy as np

    kernel = build_callable(_ROUTINE, _desc(), Arch.LIBXSMM_X86_AVX512_SKX)
    rng = np.random.default_rng(0)

    def aligned(shape, alignment=64):
        """A 64-byte-aligned column-major (Fortran-order) float64 array of random data.

        The aligned ``vmovapd`` loads/stores of A and C require 64-byte alignment; B is only
        read via ``vbroadcastsd`` and has no alignment requirement, but we align it too.

        We over-allocate a byte buffer, slice at the first aligned offset, and reshape as a
        *view* (Fortran order) -- never ``asfortranarray``, which would copy into a fresh,
        unaligned buffer. The assertion guards against any accidental copy."""
        count = int(np.prod(shape))
        buf = np.empty(count * 8 + alignment, dtype=np.uint8)
        offset = (-buf.ctypes.data) % alignment
        arr = (
            buf[offset : offset + count * 8].view(np.float64).reshape(shape, order="F")
        )
        assert arr.ctypes.data % alignment == 0
        arr[...] = rng.random(count).reshape(shape, order="F")
        return arr

    # Roles read off the IR: rdi (arg0) feeds the vmovapd A-loads, rsi (arg1) the
    # vbroadcastsd B-scalars, rdx (arg2) the C accumulators. All column-major:
    # C[M,N] += A[M,K] @ B[K,N] with lda=M, ldb=K, ldc=M (see ref_matmul_colmaj).
    a = aligned((_M, _K))
    b = aligned((_K, _N))
    c = aligned((_M, _N))

    expected = c + a @ b

    # PeachPy loads the kernel with ctypes argtypes POINTER(c_double) (from the ptr(double_)
    # arguments), which rejects c_void_p / plain ints -- pass typed pointers.
    ptr_double = ctypes.POINTER(ctypes.c_double)
    kernel(
        a.ctypes.data_as(ptr_double),
        b.ctypes.data_as(ptr_double),
        c.ctypes.data_as(ptr_double),
    )

    assert np.allclose(c, expected)
