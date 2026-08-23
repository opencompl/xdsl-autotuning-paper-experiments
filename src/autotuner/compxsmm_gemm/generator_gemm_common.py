from xdsl.dialects.x86.ops import si32
from xdsl.ir import SSAValue
from xdsl.parser import IntegerAttr
from xdsl.rewriter import InsertPoint
from xdsl.dialects import x86_scf
from autotuner.libxsmm_gemm.generator_common import GPRegMapping, MicroKernelConfig
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode
from autotuner.libxsmm_gemm.libxsmm_generator import (
    NLoopVals,
)
from xdsl.dialects.x86.registers import (
    GeneralRegisterType,
)

from xdsl.dialects import x86


def _gpr(value: SSAValue) -> SSAValue[GeneralRegisterType]:
    return SSAValue.get(value, type=GeneralRegisterType)


def _nloop_from_args(args: tuple[SSAValue, ...]) -> NLoopVals:
    a, b, c, rbp, rsp = args
    return NLoopVals(_gpr(a), _gpr(b), _gpr(c), _gpr(rbp), _gpr(rsp))


def compxsmm_generator_gemm_header_nloop(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    *,
    n_init: int,
    n_blocking: int,
    n_done: int,
    vals: NLoopVals,
) -> NLoopVals:
    """
    In original, adds three lines of assembly: set counter to n_init, add label, add n_blocking to loop counter.
    We create the same ops, but also create a block to hold the body of the loop, and set the insertion point at end of the new block.
    """
    n_arg_reg = gp_reg_mapping.gp_reg_nloop

    generated_code.insert(n_init_op := x86.ops.DI_MovOp(n_init, destination=n_arg_reg))

    existing_block = n_init_op.parent
    assert existing_block is not None
    parent_region = existing_block.parent
    assert parent_region is not None

    # n is passed as lb, so no need to include in iter_args
    args = vals.vals

    nloop_op = generated_code.builder.insert(
        x86_scf.ForOp(
            n_init_op.destination,
            IntegerAttr(n_done, si32),
            IntegerAttr(n_blocking, si32),
            args,
        )
    )

    body_block = nloop_op.body.block

    generated_code.builder.insertion_point = InsertPoint.at_start(body_block)
    return _nloop_from_args(tuple(body_block.args[1:]))


def compxsmm_generator_gemm_footer_nloop(
    generated_code: GeneratedCode,
    *,
    vals: NLoopVals,
) -> NLoopVals:
    """Close a structured N loop after its body has materialized N semantics."""
    body_block = generated_code.builder.insertion_point.block
    generated_code.insert(x86_scf.YieldOp(*vals.vals))
    nloop_op = body_block.parent_op()
    assert isinstance(nloop_op, x86_scf.ForOp), nloop_op
    generated_code.builder.insertion_point = InsertPoint.after(nloop_op)
    return _nloop_from_args(tuple(nloop_op.results[1:]))
