from xdsl.builder import Builder
from xdsl.dialects.x86.registers import (
    R10,
    R11,
    R12,
    R14,
    R15,
    R8,
    R9,
    RBX,
    RCX,
    RDI,
    RDX,
    RSI,
    UNALLOCATED_REG64,
)
from xdsl.dialects.x86_func import FuncOp, RetOp
from xdsl.rewriter import InsertPoint
from autotuner.libxsmm_gemm.generator_common import (
    GPRegMapping,
    LoopLabelTracker,
)
from autotuner.libxsmm_gemm.generator_x86_instructions import (
    libxsmm_x86_instruction_open_stream_gemm,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode
from autotuner.libxsmm_gemm.libxsmm_main import GEMMDescriptor, GEMMFlag


def libxsmm_generator_gemm_sse_avx_avx2_avx512_kernel_wrapper(
    func_op: FuncOp, arch: Arch, desc: GEMMDescriptor
) -> None:
    loop_label_tracker = LoopLabelTracker()
    gp_reg_mapping = GPRegMapping()
    is_amxfp4_bfp32_gemm = desc.is_Amxfp4_Bfp32_gemm()
    is_amxfp4_bi8_gemm = desc.is_Amxfp4_Bi8_gemm()
    is_amxfp4_bbf16_gemm = desc.is_Amxfp4_Bbf16_gemm()

    # Define GP register mapping

    gp_reg_mapping.gp_reg_param_struct = RDI

    gp_reg_mapping.gp_reg_a = gp_reg_mapping.gp_reg_param_struct
    gp_reg_mapping.gp_reg_b = RSI
    gp_reg_mapping.gp_reg_c = RDX
    gp_reg_mapping.gp_reg_a_prefetch = RCX
    gp_reg_mapping.gp_reg_b_prefetch = R8
    gp_reg_mapping.gp_reg_zpt = R9

    # Python translation of register assignment logic
    dt = desc.datatype

    if dt.a == dt.b == "I8" and dt.c == "F32" and not is_amxfp4_bi8_gemm:
        gp_reg_mapping.gp_reg_scf = RCX
        gp_reg_mapping.gp_reg_a_prefetch = R8
        gp_reg_mapping.gp_reg_b_prefetch = R9
    elif is_amxfp4_bfp32_gemm or is_amxfp4_bbf16_gemm or is_amxfp4_bi8_gemm:
        gp_reg_mapping.gp_reg_scf = RCX
        if is_amxfp4_bi8_gemm:
            gp_reg_mapping.gp_reg_zpt = RBX
    elif (dt.a == "I8" and dt.b == "F16" and dt.c in ("F16", "F32")) or (
        dt.a == "I8" and dt.b == "BF16" and dt.c in ("BF16", "F32")
    ):
        gp_reg_mapping.gp_reg_scf = RBX
        gp_reg_mapping.gp_reg_zpt = RCX
        gp_reg_mapping.gp_reg_a_prefetch = R8
        gp_reg_mapping.gp_reg_b_prefetch = R9
    else:
        gp_reg_mapping.gp_reg_scf = UNALLOCATED_REG64  # GP_REG_UNDEF
        gp_reg_mapping.gp_reg_a_prefetch = RCX
        gp_reg_mapping.gp_reg_b_prefetch = R8

    # Handle decompress A via bitmask flag
    if GEMMFlag.DECOMPRESS_A_VIA_BITMASK in desc.flags:
        gp_reg_mapping.gp_reg_bitmap_a = RCX
        gp_reg_mapping.gp_reg_decompressed_elts = R8
        gp_reg_mapping.gp_reg_popcnt = R9

    # If we are generating the batchreduce kernel, then we rename the registers
    if (
        GEMMFlag.BATCH_REDUCE_ADDRESS in desc.flags
        or GEMMFlag.BATCH_REDUCE_STRIDE in desc.flags
    ):
        raise NotImplementedError
    elif GEMMFlag.BATCH_REDUCE_OFFSET in desc.flags:
        raise NotImplementedError

    gp_reg_mapping.gp_reg_mloop = R10
    gp_reg_mapping.gp_reg_nloop = R11
    gp_reg_mapping.gp_reg_kloop = R12
    gp_reg_mapping.gp_reg_help_0 = R14
    gp_reg_mapping.gp_reg_help_1 = R15
    gp_reg_mapping.gp_reg_help_2 = RBX

    ret_op = func_op.body.block.last_op
    assert isinstance(ret_op, RetOp)
    builder = Builder(InsertPoint.before(ret_op))
    generated_code = GeneratedCode(
        func_op, builder, arch, {arg.type: arg for arg in func_op.body.block.args}
    )

    libxsmm_x86_instruction_open_stream_gemm(
        generated_code, gp_reg_mapping, False, desc.prefetch
    )
    libxsmm_generator_gemm_sse_avx_avx2_avx512_kernel(
        generated_code, loop_label_tracker, gp_reg_mapping, desc
    )

    # In C, the stream is closed with the inline assembly register string, but we don't
    # Need to do this
    # libxsmm_x86_instruction_close_stream_gemm(
    #     generated_code, gp_reg_mapping, False, desc.prefetch
    # )


def libxsmm_generator_gemm_sse_avx_avx2_avx512_kernel(
    generated_code: GeneratedCode,
    label_tracker: LoopLabelTracker,
    gp_reg_mapping: GPRegMapping,
    desc: GEMMDescriptor,
) -> None: ...
