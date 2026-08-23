from xdsl.dialects.x86.ops import si32
from xdsl.ir import SSAValue
from xdsl.parser import IntegerAttr
from xdsl.rewriter import InsertPoint
from xdsl.dialects import x86_scf
from autotuner.libxsmm_gemm.generator_common import (
    GEMMStackVar,
    GPRegMapping,
    MicroKernelConfig,
)
from autotuner.libxsmm_gemm.generator_gemm_common import (
    libxsmm_generator_gemm_getval_stack_var,
    libxsmm_generator_gemm_setval_stack_var,
)
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode
from autotuner.libxsmm_gemm.libxsmm_generator import (
    NLoopVals,
)
from xdsl.dialects.x86.registers import (
    GeneralRegisterType,
)

from xdsl.dialects import x86

from autotuner.libxsmm_gemm.libxsmm_main import (
    GEMMDescriptor,
    GEMMFlag,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


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
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    *,
    n_blocking: int,
    vals: NLoopVals,
) -> NLoopVals:
    is_Ai4_Bi8_gemm = desc.is_Ai4_Bi8_gemm()
    is_Ai2_Bi8_gemm = desc.is_Ai2_Bi8_gemm()
    is_Ai1_Bi8_gemm = desc.is_Ai1_Bi8_gemm()
    is_Amxfp4_Bfp32_gemm = desc.is_Amxfp4_Bfp32_gemm()
    is_Amxfp4_Bi8_gemm = desc.is_Amxfp4_Bi8_gemm()
    is_Amxfp4_Bbf16_gemm = desc.is_Amxfp4_Bbf16_gemm()
    a_adjust = 4 if is_Ai2_Bi8_gemm else 8 if is_Ai1_Bi8_gemm else 1

    if desc.datatype.c == Datatype.BF16:
        raise NotImplementedError
    elif desc.datatype.c == Datatype.I8:
        raise NotImplementedError
    else:
        vals.c = generated_code.insert(
            x86.ops.RI_AddOp(
                vals.c,
                (n_blocking * (desc.ldc) * (micro_kernel_config.datatype_size_out))
                - ((desc.m) * (micro_kernel_config.datatype_size_out)),
            )
        ).register_out

    if (
        desc.datatype.c in (Datatype.F16, Datatype.F32)
        and desc.datatype.b == Datatype.F16
        and desc.datatype.a == Datatype.I8
        and (
            GEMMFlag.USE_COL_VEC_SCF in desc.flags
            or GEMMFlag.USE_COL_VEC_ZPT in desc.flags
        )
    ):
        raise NotImplementedError

    if (
        (desc.datatype.c == Datatype.BF16 or desc.datatype.c == Datatype.F32)
        and desc.datatype.b == Datatype.BF16
        and desc.datatype.a == Datatype.I8
        and (GEMMFlag.USE_COL_VEC_SCF in desc.flags)
    ):
        raise NotImplementedError

    if (
        micro_kernel_config.fused_relu
        or micro_kernel_config.vnni_cvt_output_ext_buf
        or micro_kernel_config.fused_relu_bwd
        or micro_kernel_config.fused_bcolbias
        or micro_kernel_config.fused_hcolbias
        or micro_kernel_config.fused_b8colbias
        or micro_kernel_config.fused_h8colbias
        or micro_kernel_config.fused_scolbias
        or not micro_kernel_config.overwrite_C
    ):
        raise NotImplementedError

    if micro_kernel_config.fused_relu and micro_kernel_config.overwrite_C:
        raise NotImplementedError
    if micro_kernel_config.vnni_cvt_output_ext_buf:
        raise NotImplementedError
    if micro_kernel_config.fused_relu_bwd:
        raise NotImplementedError

    if not micro_kernel_config.overwrite_C:
        # In this case also advance the output ptr
        output_ptr = libxsmm_generator_gemm_getval_stack_var(
            generated_code,
            micro_kernel_config,
            GEMMStackVar.ELT_OUTPUT_PTR,
            gp_reg_mapping.gp_reg_help_0,
            vals.rbp,
        )
        output_ptr = generated_code.insert(
            x86.ops.RI_AddOp(output_ptr, (n_blocking * (desc.ldc) * 2) - ((desc.m) * 2))
        ).register_out
        libxsmm_generator_gemm_setval_stack_var(
            generated_code,
            micro_kernel_config,
            GEMMStackVar.ELT_OUTPUT_PTR,
            output_ptr,
            vals.rbp,
        )

    if micro_kernel_config.fused_bcolbias or micro_kernel_config.fused_hcolbias:
        raise NotImplementedError

    if micro_kernel_config.fused_b8colbias:
        raise NotImplementedError

    if micro_kernel_config.fused_h8colbias:
        raise NotImplementedError

    if micro_kernel_config.fused_scolbias:
        raise NotImplementedError

    if (
        micro_kernel_config.fused_relu
        or micro_kernel_config.vnni_cvt_output_ext_buf
        or micro_kernel_config.fused_relu_bwd
        or micro_kernel_config.fused_bcolbias
        or micro_kernel_config.fused_hcolbias
        or micro_kernel_config.fused_b8colbias
        or micro_kernel_config.fused_h8colbias
        or micro_kernel_config.fused_scolbias
        or not micro_kernel_config.overwrite_C
    ):
        raise NotImplementedError

    if GEMMFlag.BATCH_REDUCE_ADDRESS in desc.flags:
        raise NotImplementedError
    else:
        # handle trans B
        b_offset = 0
        # k packing factor for VNNI
        k_pack_factor = 1

        if GEMMFlag.VNNI_A in desc.flags:
            # for VNNI we are stepping through to pack ks
            raise NotImplementedError

        if GEMMFlag.TRANS_B in desc.flags:
            b_offset = (
                n_blocking * micro_kernel_config.datatype_size_in2 * k_pack_factor
            )
        else:
            b_offset = n_blocking * desc.ldb * micro_kernel_config.datatype_size_in2

        vals.b = generated_code.insert(x86.ops.RI_AddOp(vals.b, b_offset)).register_out

        if GEMMFlag.DECOMPRESS_A_VIA_BITMASK in desc.flags:
            raise NotImplementedError
        else:
            vals.a = generated_code.insert(
                x86.ops.RI_SubOp(
                    vals.a,
                    desc.m
                    * micro_kernel_config.datatype_size_in
                    * k_pack_factor
                    // a_adjust,
                )
            ).register_out

        if is_Amxfp4_Bfp32_gemm or is_Amxfp4_Bbf16_gemm or is_Amxfp4_Bi8_gemm:
            raise NotImplementedError

        if is_Ai4_Bi8_gemm and GEMMFlag.USE_MxK_ZPT in desc.flags:
            raise NotImplementedError

        if GEMMPrefetchType.AL2 == desc.prefetch:
            raise NotImplementedError

    # Insert yield op for resulting registers
    body_block = generated_code.builder.insertion_point.block
    generated_code.insert(x86_scf.YieldOp(*vals.vals))

    # Set up builder to build after loop
    nloop_op = body_block.parent_op()
    assert isinstance(nloop_op, x86_scf.ForOp), nloop_op

    generated_code.builder.insertion_point = InsertPoint.after(nloop_op)
    return _nloop_from_args(tuple(nloop_op.results[1:]))
