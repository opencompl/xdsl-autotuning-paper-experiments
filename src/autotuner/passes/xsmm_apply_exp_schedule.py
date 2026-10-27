"""Lower exp through structured register-block loops, like CompXSMM matmul."""

from dataclasses import dataclass

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import builtin, x86, x86_scf
from xdsl.dialects.x86.registers import R10, UNALLOCATED_REG64, GeneralRegisterType
from xdsl.ir import Block, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from autotuner.compxsmm_gemm.generator_exp import (
    configure_exp_registers,
    generate_exp_block,
)
from autotuner.dialects.xsmm import ExpOp
from autotuner.libxsmm_gemm.generator_mateltwise_unary_binary_avx_avx512 import (
    libxsmm_configure_microkernel_loops,
    libxsmm_generator_configure_M_N_blocking,
    libxsmm_load_2d_reg_block,
    libxsmm_store_2d_reg_block,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode
from autotuner.libxsmm_gemm.libxsmm_main import MeltwDescriptor
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


@dataclass
class ApplyExpSchedulePattern(RewritePattern):
    disable_regalloc: bool

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExpOp, rewriter: PatternRewriter, /) -> None:
        code = GeneratedCode(
            Builder(InsertPoint.before(op)), Arch.LIBXSMM_X86_AVX512_SKX
        )
        length = op.length.value.data
        desc = MeltwDescriptor(length, 1, length, length, Datatype.F32)
        config = configure_exp_registers(
            code, op.degree.value.data, disable_regalloc=self.disable_regalloc
        )
        counter_reg = UNALLOCATED_REG64 if self.disable_regalloc else R10
        blocking, _ = libxsmm_generator_configure_M_N_blocking(desc)
        input = SSAValue.get(op.input, type=GeneralRegisterType)
        output = SSAValue.get(op.output, type=GeneralRegisterType)
        for start in range(0, length, blocking):
            extent = min(blocking, length - start)
            trips, unroll = libxsmm_configure_microkernel_loops(extent, config)
            loop = None
            body_code = code
            if trips > unroll:
                counter = code.insert(x86.ops.DI_MovOp(0, destination=counter_reg))
                body = Block(arg_types=(counter_reg, input.type, output.type))
                loop = code.insert(
                    x86_scf.ForOp(
                        counter.destination,
                        builtin.IntegerAttr(trips, x86.ops.si32),
                        builtin.IntegerAttr(unroll, x86.ops.si32),
                        (input, output),
                        body,
                    )
                )
                input, output = (
                    SSAValue.get(arg, type=GeneralRegisterType) for arg in body.args[1:]
                )
                body_code = GeneratedCode(Builder(InsertPoint.at_end(body)), code.arch)
            values = libxsmm_load_2d_reg_block(body_code, input, unroll, config)
            values = generate_exp_block(body_code, values, config)
            libxsmm_store_2d_reg_block(body_code, output, values)
            if loop is not None:
                input = body_code.insert(
                    x86.ops.RI_AddOp(input, unroll * 64)
                ).register_out
                output = body_code.insert(
                    x86.ops.RI_AddOp(output, unroll * 64)
                ).register_out
                body_code.insert(x86_scf.YieldOp(input, output))
                input, output = (
                    SSAValue.get(arg, type=GeneralRegisterType) for arg in loop.res
                )
                # Each region starts at the current pointers, just as in the
                # literal generator. Preserve its restoration/advance schedule.
                input = code.insert(x86.ops.RI_SubOp(input, extent * 4)).register_out
                output = code.insert(x86.ops.RI_SubOp(output, extent * 4)).register_out
            if start + extent < length:
                input = code.insert(x86.ops.RI_AddOp(input, extent * 4)).register_out
                output = code.insert(x86.ops.RI_AddOp(output, extent * 4)).register_out
        rewriter.erase(op)


@dataclass(frozen=True)
class XsmmApplyExpSchedulePass(ModulePass):
    name = "xsmm-apply-exp-schedule"
    disable_regalloc: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            ApplyExpSchedulePattern(self.disable_regalloc), apply_recursively=False
        ).rewrite_module(op)
