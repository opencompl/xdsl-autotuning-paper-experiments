"""Literal packed FP32 EXP subset of the upstream unary/binary generator.

Register configuration, region blocking, runtime loops, and register-block
load/compute/store retain their upstream responsibilities. Arithmetic for one
vector lives in generator_common_x86, shared with future fused operations.
"""

from dataclasses import dataclass

from xdsl.dialects import x86
from xdsl.dialects.x86.registers import (
    R8,
    R9,
    R10,
    AVX512MaskRegisterType,
    AVX512RegisterType,
    GeneralRegisterType,
)
from xdsl.ir import Block, SSAValue
from xdsl.rewriter import InsertPoint

from autotuner.libxsmm_gemm.generator_common import (
    LoopLabelTracker,
    libxsmm_compute_equalized_blocking,
)
from autotuner.libxsmm_gemm.generator_common_x86 import (
    Exp3Coefficients,
    Exp5Coefficients,
    libxsmm_generator_exp_ps_3dts_avx512,
    libxsmm_generator_exp_ps_5dts_avx512,
    libxsmm_generator_prepare_coeffs_exp_ps_3dts_avx512,
    libxsmm_generator_prepare_coeffs_exp_ps_5dts_avx512,
)
from autotuner.libxsmm_gemm.generator_x86_instructions import (
    libxsmm_x86_instruction_jump_back_to_label,
    libxsmm_x86_instruction_register_jump_back_label,
)
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode
from autotuner.libxsmm_gemm.libxsmm_main import MeltwDescriptor


@dataclass(frozen=True)
class MeltwKernelConfig:
    reserved_zmms: int
    vec_y: AVX512RegisterType
    vec_z: AVX512RegisterType
    coeffs: Exp3Coefficients
    disable_regalloc: bool = False


def libxsmm_configure_unary_kernel_vregs_masks(
    code: GeneratedCode, degree: int, *, disable_regalloc: bool = False
) -> MeltwKernelConfig:
    def zmm(index: int) -> AVX512RegisterType:
        return (
            AVX512RegisterType.unallocated()
            if disable_regalloc
            else AVX512RegisterType.from_index(index)
        )

    if degree == 3:
        coeffs = libxsmm_generator_prepare_coeffs_exp_ps_3dts_avx512(
            code, zmm(6), zmm(5), zmm(4), zmm(3), zmm(7), zmm(2)
        )
        return MeltwKernelConfig(8, zmm(0), zmm(1), coeffs, disable_regalloc)
    coeffs = libxsmm_generator_prepare_coeffs_exp_ps_5dts_avx512(
        code,
        zmm(14),
        zmm(13),
        zmm(12),
        zmm(11),
        zmm(10),
        zmm(9),
        zmm(15),
        zmm(8),
        zmm(7),
        zmm(2),
        zmm(6),
        zmm(5),
    )
    # Upstream leaves zmm0/1 reserved even after moving the auxiliaries to 4/3.
    return MeltwKernelConfig(16, zmm(4), zmm(3), coeffs, disable_regalloc)


def libxsmm_generator_configure_M_N_blocking(desc: MeltwDescriptor) -> tuple[int, int]:
    m_chunks = desc.m // 16
    if m_chunks <= 32:
        return desc.m, desc.n
    blocking = libxsmm_compute_equalized_blocking(m_chunks, (m_chunks + 1) // 2)
    # Upstream uses range_1, NOT block_1 (1024 therefore remains one region).
    return blocking.range_1 * 16, desc.n


def libxsmm_configure_microkernel_loops(
    m: int, config: MeltwKernelConfig
) -> tuple[int, int]:
    m_trips = m // 16
    m_unroll = min(m_trips, 16, 32 - config.reserved_zmms)
    while m_trips % m_unroll:
        m_unroll -= 1
    return m_trips, m_unroll


def libxsmm_load_2d_reg_block(
    code: GeneratedCode,
    input: SSAValue[GeneralRegisterType],
    unroll: int,
    config: MeltwKernelConfig,
) -> tuple[SSAValue[AVX512RegisterType], ...]:
    # N=1: column stride disappears, but vector/register order is unchanged.
    return tuple(
        SSAValue.get(
            code.insert(
                x86.ops.DM_VmovupsOp(
                    input,
                    64 * im,
                    destination=(
                        AVX512RegisterType.unallocated()
                        if config.disable_regalloc
                        else AVX512RegisterType.from_index(config.reserved_zmms + im)
                    ),
                )
            ),
            type=AVX512RegisterType,
        )
        for im in range(unroll)
    )


def libxsmm_compute_unary_2d_reg_block_op(
    code: GeneratedCode,
    values: tuple[SSAValue[AVX512RegisterType], ...],
    config: MeltwKernelConfig,
) -> tuple[SSAValue[AVX512RegisterType], ...]:
    if isinstance(config.coeffs, Exp5Coefficients):
        return tuple(
            libxsmm_generator_exp_ps_5dts_avx512(
                code,
                value,
                config.vec_y,
                config.vec_z,
                (
                    AVX512MaskRegisterType.unallocated()
                    if config.disable_regalloc
                    else AVX512MaskRegisterType.from_index(1)
                ),
                config.coeffs,
            )
            for value in values
        )
    return tuple(
        libxsmm_generator_exp_ps_3dts_avx512(
            code, value, config.vec_y, config.vec_z, config.coeffs
        )
        for value in values
    )


def libxsmm_store_2d_reg_block(
    code: GeneratedCode,
    output: SSAValue[GeneralRegisterType],
    values: tuple[SSAValue[AVX512RegisterType], ...],
) -> None:
    for im, value in enumerate(values):
        code.insert(x86.ops.MS_VmovupsOp(output, value, 64 * im))


def libxsmm_generator_unary_binary_2d_microkernel(
    code: GeneratedCode,
    input: SSAValue[GeneralRegisterType],
    output: SSAValue[GeneralRegisterType],
    m: int,
    config: MeltwKernelConfig,
    labels: LoopLabelTracker,
) -> tuple[SSAValue[GeneralRegisterType], SSAValue[GeneralRegisterType]]:
    m_trips, unroll = libxsmm_configure_microkernel_loops(m, config)
    counter = None
    if m_trips > unroll:
        counter = code.insert(x86.ops.DI_MovOp(0, destination=R10)).destination
        header = Block(arg_types=(R10, R8, R9))
        region = code.current_block.parent
        assert region is not None
        region.add_block(header)
        code.insert(x86.ops.FallthroughOp((counter, input, output), header))
        code.builder.insertion_point = InsertPoint.at_end(header)
        labels.dest_blocks.append(header)
        libxsmm_x86_instruction_register_jump_back_label(code, labels)
        counter, input, output = (
            SSAValue.get(value, type=GeneralRegisterType) for value in header.args
        )
        counter = code.insert(x86.ops.RI_AddOp(counter, unroll)).register_out

    values = libxsmm_load_2d_reg_block(code, input, unroll, config)
    values = libxsmm_compute_unary_2d_reg_block_op(code, values, config)
    libxsmm_store_2d_reg_block(code, output, values)

    if counter is not None:
        input = code.insert(x86.ops.RI_AddOp(input, unroll * 64)).register_out
        output = code.insert(x86.ops.RI_AddOp(output, unroll * 64)).register_out
        code.insert(x86.ops.SI_CmpOp(counter, m_trips))
        _, input_arg, output_arg = libxsmm_x86_instruction_jump_back_to_label(
            code, x86.ops.C_JlOp, labels, (counter, input, output)
        )
        input = code.insert(
            x86.ops.RI_SubOp(SSAValue.get(input_arg, type=GeneralRegisterType), m * 4)
        ).register_out
        output = code.insert(
            x86.ops.RI_SubOp(SSAValue.get(output_arg, type=GeneralRegisterType), m * 4)
        ).register_out
    return input, output


def libxsmm_generator_unary_binary_avx512_kernel(
    code: GeneratedCode, desc: MeltwDescriptor, degree: int
) -> None:
    param = code.current_block.args[0]
    # libxsmm_meltw_unary_param's System V layout, asserted in exp/init.c.
    input = code.insert(x86.ops.DM_MovOp(param, 32, destination=R8)).destination
    output = code.insert(x86.ops.DM_MovOp(param, 80, destination=R9)).destination
    config = libxsmm_configure_unary_kernel_vregs_masks(code, degree)
    m_blocking, _ = libxsmm_generator_configure_M_N_blocking(desc)
    labels = LoopLabelTracker()
    # NM order with N=1: no N loop. Keep the generator-time M regions distinct
    # from the runtime register-block loop within each region.
    for m_start in range(0, desc.m, m_blocking):
        m = min(m_blocking, desc.m - m_start)
        input, output = libxsmm_generator_unary_binary_2d_microkernel(
            code, input, output, m, config, labels
        )
        if m_start + m < desc.m:
            input = code.insert(x86.ops.RI_AddOp(input, m * 4)).register_out
            output = code.insert(x86.ops.RI_AddOp(output, m * 4)).register_out
