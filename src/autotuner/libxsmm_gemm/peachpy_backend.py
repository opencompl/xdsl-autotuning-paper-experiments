"""
Assemble a generated x86 GEMM microkernel with PeachPy.

The generator in :mod:`autotuner.libxsmm_gemm` builds an in-memory xDSL ``x86``/
``x86_func`` module whose operations already carry **fully-assigned physical registers**
(``zmm26``, ``rdi``, ``r11``, …). The downstream xDSL passes only verify liveness and add
callee-save handling; they never allocate. That means we can translate each instruction op
straight into a PeachPy instruction by reading the physical register name off every operand
-- no SSA/value tracking required.

Public entry points:

* :func:`gemm_func_to_peachpy` -- translate an already-built ``x86_func.FuncOp`` into an
  un-finalized :class:`peachpy.x86_64.Function`.
* :func:`build_function` -- generate the IR and translate it to an un-finalized ``Function``.
  Works on any host; the caller can ``fn.finalize(system_v_x86_64_abi).encode()`` to obtain
  the machine-code bytes even on a non-x86 interpreter (e.g. this arm64 Mac).
* :func:`build_callable` -- finalize/encode/load the kernel into a ctypes-callable. Requires
  an x86-64 interpreter (``peachpy.x86_64.abi.detect()`` is not ``None``) and raises
  :class:`RuntimeError` otherwise.

Scope: the simple aligned DP/SP path (M a multiple of the vector length, ``beta=1``, no
masking). The family-based dispatch below extends cleanly to masked-remainder tiles later --
add the corresponding family handlers; the register/label/control-flow machinery already
supports them.
"""

from __future__ import annotations

import peachpy.x86_64
import peachpy.x86_64.registers as preg
from peachpy import Argument, const_double_, double_, ptr
from peachpy.x86_64 import abi, uarch
from peachpy.x86_64.operand import qword, zword

from xdsl.dialects.x86.ops import (
    ConditionalJumpOperation,
    DI_Operation,
    DM_Operation,
    DMK_Operation,
    DS_Operation,
    DSS_Operation,
    D_PopOp,
    FallthroughOp,
    GetAnyRegisterOperation,
    KS_Operation,
    LabelOp,
    MS_Operation,
    MSK_Operation,
    RI_Operation,
    RS_Operation,
    RSMB_Operation,
    RSS_Operation,
    SI_CmpOp,
    S_PushOp,
)
from xdsl.dialects.x86.registers import Reg32Type
from xdsl.dialects.x86_func import FuncOp, RetOp
from xdsl.ir import SSAValue

from autotuner.libxsmm_gemm.generator_gemm import build_gemm_module
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_main import GEMMDescriptor

# An AVX-512-capable target so that ``zmm*``/EVEX/``k*`` operands are accepted by PeachPy's
# per-instruction target checks.
_TARGET = uarch.skylake_xeon


def PREG(value: SSAValue):
    """The PeachPy physical register singleton named by ``value``'s register type."""
    return getattr(preg, value.type.register_name.data)


def MEM(op):
    """A PeachPy memory operand ``[base]`` or ``[base + offset]`` from a memory op."""
    base = PREG(op.memory)
    offset = op.memory_offset.value.data
    return [base + offset] if offset else [base]


def IMM(attr):
    """The Python integer value of an ``IntegerAttr`` immediate/offset."""
    return attr.value.data


def _mnemonic(op):
    """The PeachPy instruction callable for an ``x86`` instruction op."""
    return getattr(peachpy.x86_64, op.assembly_instruction_name().upper())


def PREG32(value: SSAValue):
    """The 32-bit PeachPy register for a 64-bit register value.

    ``kmovb`` takes a 32-bit GPR operand even though the IR carries the 64-bit register
    (mirrors ``KS_KMovBOp.assembly_line_args``' ``Reg32Type.from_index`` conversion)."""
    return getattr(preg, Reg32Type.from_index(value.type.index.data).register_name.data)


def _mask(op):
    """The PeachPy mask operand for a masked op: ``k{z}`` when zero-masking, else ``k``."""
    mask_reg = PREG(op.mask_reg)
    return mask_reg.z if op.z is not None else mask_reg


def MASKED_REG(value: SSAValue, op):
    """A masked vector register, e.g. ``zmm26(k1)`` or ``zmm26(k1.z)``."""
    return PREG(value)(_mask(op))


def MASKED_MEM(op):
    """A masked 512-bit memory operand for a masked store, e.g. ``zword[rdx+64](k1)``."""
    base = PREG(op.memory)
    offset = op.memory_offset.value.data
    address = base + offset if offset else base
    return zword[address](_mask(op))


def BCAST_MEM(op):
    """The memory operand of an ``RSMB`` op: ``qword.to8[base(+off)]`` when the EVEX broadcast
    modifier is set (``{1to8}``), else a plain ``[base(+off)]`` memory operand."""
    if op.broadcast is None:
        return MEM(op)
    base = PREG(op.memory)
    offset = op.memory_offset.value.data
    return qword.to8[base + offset] if offset else qword.to8[base]


def _emit_op(op, labels: dict[str, peachpy.x86_64.Label]) -> None:
    """
    Translate a single xDSL ``x86``/``x86_func`` op into PeachPy instruction(s) on the
    currently-active PeachPy function stream.

    Dispatch is family-based: one branch per operand-shape base class. Ordered most specific
    first so that e.g. the standalone push/pop/cmp ops are matched before the generic
    register families.
    """
    # Structural ops that produce no instruction.
    if isinstance(op, (GetAnyRegisterOperation, FallthroughOp)):
        # `get_register`/`get_avx_register` only name a physical register; `fallthrough`
        # is the natural fall-through to the next (physically adjacent) block.
        return

    if isinstance(op, LabelOp):
        peachpy.x86_64.LABEL(labels[op.label.data])
        return

    if isinstance(op, RetOp):
        # PeachPy's RETURN pseudo-op emits the epilogue restore of callee-save registers
        # plus `ret`; a bare RET() would leave PeachPy's auto-pushed rbp unbalanced.
        peachpy.x86_64.RETURN()
        return

    if isinstance(op, ConditionalJumpOperation):
        # The not-taken (else) successor is always the physically next block, so we only
        # emit the taken (then) jump and fall through to the else block.
        then_label = op.then_block.first_op
        assert isinstance(then_label, LabelOp)
        _mnemonic(op)(labels[then_label.label.data])
        return

    # Instruction families.
    if isinstance(op, S_PushOp):
        peachpy.x86_64.PUSH(PREG(op.source))
    elif isinstance(op, D_PopOp):
        peachpy.x86_64.POP(PREG(op.destination))
    elif isinstance(op, DS_Operation):
        _mnemonic(op)(PREG(op.destination), PREG(op.source))
    elif isinstance(op, RS_Operation):
        _mnemonic(op)(PREG(op.register_in), PREG(op.source))
    elif isinstance(op, RI_Operation):
        _mnemonic(op)(PREG(op.register_in), IMM(op.immediate))
    elif isinstance(op, DI_Operation):
        _mnemonic(op)(PREG(op.destination), IMM(op.immediate))
    elif isinstance(op, SI_CmpOp):
        peachpy.x86_64.CMP(PREG(op.source), IMM(op.immediate))
    elif isinstance(op, DM_Operation):
        _mnemonic(op)(PREG(op.destination), MEM(op))
    elif isinstance(op, MS_Operation):
        _mnemonic(op)(MEM(op), PREG(op.source))
    elif isinstance(op, RSS_Operation):
        _mnemonic(op)(PREG(op.register_in), PREG(op.source1), PREG(op.source2))
    elif isinstance(op, RSMB_Operation):
        # Memory-broadcast FMA, e.g. `vfmadd231pd zmm, zmm, [rsi]{1to8}`.
        _mnemonic(op)(PREG(op.register_in), PREG(op.source1), BCAST_MEM(op))
    elif isinstance(op, DSS_Operation):
        _mnemonic(op)(PREG(op.destination), PREG(op.source1), PREG(op.source2))
    elif isinstance(op, KS_Operation):
        # kmov from a (32-bit) general register into a mask register.
        _mnemonic(op)(PREG(op.destination), PREG32(op.source))
    elif isinstance(op, DMK_Operation):
        # Masked load: `vmovupd zmm{k}, [mem]`.
        _mnemonic(op)(MASKED_REG(op.destination, op), MEM(op))
    elif isinstance(op, MSK_Operation):
        # Masked store: `vmovupd [mem]{k}, zmm`.
        _mnemonic(op)(MASKED_MEM(op), PREG(op.source))
    else:
        raise NotImplementedError(f"No PeachPy translation for {op.name}")


def gemm_func_to_peachpy(func_op: FuncOp) -> peachpy.x86_64.Function:
    """
    Translate an ``x86_func.FuncOp`` into an un-finalized :class:`peachpy.x86_64.Function`.

    The three System V argument registers ``rdi``/``rsi``/``rdx`` are established by declaring
    three pointer arguments; the body then references those registers directly (that *is*
    System V argument passing), so no ``LOAD.ARGUMENT`` is emitted.
    """
    name = func_op.sym_name.data

    # Pointer argument types are cosmetic here (only the C signature / arg-register mapping
    # matters -- all pointers are 64-bit and the body addresses rdi/rsi/rdx directly), but we
    # keep them faithful: read-only inputs A and B are const, output C is not.
    args = (
        Argument(ptr(const_double_), name="a"),
        Argument(ptr(const_double_), name="b"),
        Argument(ptr(double_), name="c"),
    )

    # One PeachPy Label per xDSL label string, shared by its definition and every jump.
    labels = {
        label_op.label.data: peachpy.x86_64.Label(label_op.label.data)
        for block in func_op.body.blocks
        for label_op in block.ops
        if isinstance(label_op, LabelOp)
    }

    with peachpy.x86_64.Function(name, args, target=_TARGET) as function:
        # Blocks are already in program order; iterate blocks then ops within each block.
        for block in func_op.body.blocks:
            for op in block.ops:
                _emit_op(op, labels)

    return function


def build_function(
    routine_name: str, desc: GEMMDescriptor, arch: Arch
) -> peachpy.x86_64.Function:
    """
    Generate the GEMM IR and translate it into an un-finalized :class:`peachpy.x86_64.Function`.

    Works on any host. To obtain machine-code bytes (including on a non-x86 interpreter),
    ``build_function(...).finalize(abi.system_v_x86_64_abi).encode()``. For a directly-callable
    kernel on an x86-64 host, use :func:`build_callable`.
    """
    _, func_op = build_gemm_module(routine_name, desc, arch)
    return gemm_func_to_peachpy(func_op)


def build_callable(routine_name: str, desc: GEMMDescriptor, arch: Arch):
    """
    Build the kernel and load it into a ctypes-callable, via
    ``fn.finalize(abi).encode().load()``.

    Requires the running interpreter to be x86-64 (``abi.detect()`` is not ``None``); raises
    :class:`RuntimeError` otherwise. Use :func:`build_function` to obtain the un-finalized
    ``Function`` / machine-code bytes on a non-x86 host.
    """
    detected_abi = abi.detect()
    if detected_abi is None:
        raise RuntimeError(
            "build_callable requires an x86-64 interpreter ",
            "(peachpy.x86_64.abi.detect() returned None); use build_function() to obtain ",
            "the un-finalized Function or machine-code bytes on this host.",
        )
    return (
        build_function(routine_name, desc, arch).finalize(detected_abi).encode().load()
    )
