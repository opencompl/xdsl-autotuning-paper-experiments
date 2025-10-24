from xdsl.ir import Block
from xdsl.rewriter import InsertPoint, Rewriter
from autotuner.libxsmm_gemm.generator_common import (
    GPRegMapping,
    LoopLabelTracker,
    MicroKernelConfig,
)
from autotuner.libxsmm_gemm.generator_x86_instructions import (
    libxsmm_x86_instruction_jump_back_to_label,
    libxsmm_x86_instruction_register_jump_back_label,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode

from xdsl.dialects import x86

from autotuner.libxsmm_gemm.libxsmm_main import GEMMDescriptor, GEMMFlag
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


def libxsmm_generator_gemm_init_micro_kernel_config(
    config: MicroKernelConfig, arch: Arch, desc: GEMMDescriptor, use_masking_a_c: bool
) -> MicroKernelConfig:
    libxsmm_generator_gemm_setup_fusion_microkernel_properties(desc, config)
    if arch <= Arch.LIBXSMM_X86_GENERIC or arch > Arch.LIBXSMM_X86_ALLFEAT:
        config.instruction_set = Arch.LIBXSMM_TARGET_ARCH_GENERIC
        config.vector_reg_count = 0
        config.use_masking_a_c = 0
        config.vector_name = "a"
        config.vector_length = 0
        config.datatype_size_in = 0
        config.datatype_size_in2 = 0
        config.datatype_size_out = 0
    elif arch <= Arch.LIBXSMM_X86_SSE42:
        raise NotImplementedError
    elif arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX:
        raise NotImplementedError
    elif arch < Arch.LIBXSMM_X86_AVX512_SKX:
        raise NotImplementedError
    elif arch <= Arch.LIBXSMM_X86_ALLFEAT:
        config.instruction_set = arch
        config.vector_reg_count = 32
        config.use_masking_a_c = use_masking_a_c
        config.vector_name = "z"
        if Datatype.F64 == desc.datatype.ab:
            config.vector_length = 8
            config.datatype_size_in = 8
            config.datatype_size_in2 = 8
            config.datatype_size_out = 8
            if GEMMFlag.ALIGN_A in desc.flags:
                assert not desc.lda % config.vector_length
                config.a_vmove_instruction = x86.ops.DM_VmovapdOp
            else:
                config.a_vmove_instruction = x86.ops.DM_VMovupdOp

            config.b_vmove_instruction = x86.ops.DM_VbroadcastsdOp
            if GEMMFlag.ALIGN_C in desc.flags:
                assert not desc.ldc % config.vector_length
                config.c_vmove_instruction = x86.ops.DM_VmovapdOp
                if not use_masking_a_c:
                    config.c_vmove_nts_instruction = x86.ops.MS_VmovntpdOp

                else:
                    config.c_vmove_nts_instruction = x86.ops.MS_VmovapdOp
            else:
                config.c_vmove_instruction = x86.ops.DM_VmovupdOp
                config.c_vmove_nts_instruction = x86.ops.MS_VmovupdOp

            # config.vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
            config.vmul_instruction = x86.ops.RSS_Vfmadd231pdOp
            config.vadd_instruction = x86.ops.DSS_AddpdOp
        elif Datatype.F32 == desc.datatype.ab:
            config.vector_length = 16
            config.datatype_size_in = 4
            config.datatype_size_in2 = 4
            config.datatype_size_out = 4
            # We overwrite in order to support F32BF8 kernels (currently used internally for BF8 emulation via stack allocated arrays)
            if desc.datatype.c in (Datatype.BF8, Datatype.HF8):
                config.datatype_size_out = 1

            if GEMMFlag.ALIGN_A in desc.flags:
                assert not desc.lda % config.vector_length
                config.a_vmove_instruction = x86.ops.DM_VmovapsOp
            else:
                config.a_vmove_instruction = x86.ops.DM_VmovupsOp
            config.b_vmove_instruction = x86.ops.DM_VbroadcastssOp
            if GEMMFlag.ALIGN_C in desc.flags:
                assert not desc.ldc % config.vector_length
                config.c_vmove_instruction = x86.ops.DM_VmovapsOp
                if not use_masking_a_c:
                    config.c_vmove_nts_instruction = x86.ops.MS_VmovapsOp
                else:
                    config.c_vmove_nts_instruction = x86.ops.MS_VmovntpsOp
            else:
                config.c_vmove_instruction = x86.ops.DM_VmovupsOp
                config.c_vmove_nts_instruction = x86.ops.MS_VmovupsOp

            # config.vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
            config.vmul_instruction = x86.ops.RSS_Vfmadd231psOp
            config.vadd_instruction = x86.ops.DSS_AddpsOp

        elif Datatype.I16 == desc.datatype.ab:
            raise NotImplementedError
        elif Datatype.I8 == desc.datatype.ab:
            raise NotImplementedError
        elif (
            desc.datatype.a == Datatype.I8
            and desc.datatype.b == Datatype.BF16
            and (desc.datatype.c == Datatype.BF16 or desc.datatype.c == Datatype.F32)
        ):
            raise NotImplementedError
        elif (
            (desc.datatype.a == Datatype.BF8 or desc.datatype.a == Datatype.HF8)
            and desc.datatype.b == Datatype.BF16
            and (desc.datatype.c == Datatype.BF16 or desc.datatype.c == Datatype.F32)
        ):
            raise NotImplementedError
        elif (
            (desc.datatype.a == Datatype.I8 or desc.datatype.a == Datatype.BF8)
            and desc.datatype.b == Datatype.F16
            and (desc.datatype.c == Datatype.F16 or desc.datatype.c == Datatype.F32)
            and desc.datatype.comp == Datatype.F16
        ):
            raise NotImplementedError
        elif (
            desc.datatype.a == Datatype.F16
            and (desc.datatype.c == Datatype.F16 or desc.datatype.c == Datatype.F32)
            and desc.datatype.comp == Datatype.F16
        ):
            raise NotImplementedError
        elif (
            desc.datatype.a == Datatype.F16
            and (desc.datatype.c == Datatype.F16 or desc.datatype.c == Datatype.F32)
            and desc.datatype.comp == Datatype.F32
        ):
            raise NotImplementedError
        elif desc.datatype.a == Datatype.BF16 and (desc.flags & GEMMFlag.VNNI_A) > 0:
            raise NotImplementedError
        elif desc.datatype.a == Datatype.BF16 and (desc.flags & GEMMFlag.VNNI_A) == 0:
            raise NotImplementedError
        elif (desc.datatype.a == Datatype.BF8 or desc.datatype.a == Datatype.HF8) and (
            desc.flags & GEMMFlag.VNNI_A
        ) > 0:
            raise NotImplementedError

        else:
            assert False, "Should not happen"
    else:
        assert False, "Should not happen"

    return config


def libxsmm_generator_gemm_setup_fusion_microkernel_properties(
    desc: GEMMDescriptor, config: MicroKernelConfig
):
    config.fused_bcolbias = 0
    config.fused_hcolbias = 0
    config.fused_b8colbias = 0
    config.fused_h8colbias = 0
    config.fused_scolbias = 0
    config.fused_relu = 0
    config.fused_relu_nobitmask = 0
    config.fused_relu_bwd = 0
    config.fused_sigmoid = 0
    config.overwrite_C = 1
    config.vnni_format_C = 0
    config.decompress_A = 0
    config.sparsity_factor_A = 1
    config.vnni_cvt_output_ext_buf = 0
    config.norm_to_normT_B_ext_buf = 0
    config.stride_b_trans = 0
    config.fused_eltwise = 0
    config.has_colbias_act_fused = 0
    # Python translation of the C-style block above

    # Set VNNI format for the output C tensor if the flag is set
    config.vnni_format_C = 1 if (desc.flags & GEMMFlag.VNNI_C) > 0 else 0

    if GEMMFlag.USE_XGEMM_EXT_ABI in desc.flags:
        raise NotImplementedError


def libxsmm_generator_gemm_header_kloop(
    generated_code: GeneratedCode,
    loop_label_tracker: LoopLabelTracker,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    m_blocking: int,
    k_blocking: int,
) -> None:
    """
    In original, adds three lines of assembly: set counter to 0, add label, add k_blocking to loop counter.
    We create the same ops, but also create a block to hold the body of the loop, and set the insertion point at end of the new block.
    """
    k_arg_reg = gp_reg_mapping.gp_reg_kloop

    curr_vals = generated_code.current_val_by_reg
    generated_code.insert(k_init_op := x86.ops.DI_MovOp(0, destination=k_arg_reg))

    existing_block = k_init_op.parent
    assert existing_block is not None
    parent_region = existing_block.parent
    assert parent_region is not None

    if k_init_op.next_op is not None:
        new_block = existing_block.split_before(
            k_init_op.next_op, arg_types=(k_arg_reg,)
        )
    else:
        new_block = Block(arg_types=(k_arg_reg,))
        parent_region.insert_block_after(new_block, existing_block)

    # Jump/fallthrough to the newly created block
    # TODO: make sure that we don't print the jump in xDSL if the destination is the
    # next block
    Rewriter.insert_op(
        x86.ops.C_JmpOp((k_init_op.destination,), new_block),
        InsertPoint.at_end(existing_block),
    )

    generated_code.builder.insertion_point = InsertPoint.at_start(new_block)
    curr_vals.clear()
    curr_vals |= {arg.type: arg for arg in new_block.args}

    libxsmm_x86_instruction_register_jump_back_label(generated_code, loop_label_tracker)
    generated_code.insert(
        x86.ops.RI_AddOp(curr_vals[k_arg_reg], 4, register_out=k_arg_reg)
    )


def libxsmm_generator_gemm_footer_kloop(
    generated_code: GeneratedCode,
    loop_label_tracker: LoopLabelTracker,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    gemm_desc: GEMMDescriptor,
    m_blocking: int,
    max_blocked_k: int,
    k_loop_complete: bool,
) -> None:
    generated_code.insert(
        x86.ops.SI_CmpOp(
            generated_code.current_val_by_reg[gp_reg_mapping.gp_reg_kloop],
            max_blocked_k,
        )
    )

    libxsmm_x86_instruction_jump_back_to_label(
        generated_code, x86.ops.C_JlOp, loop_label_tracker
    )
    if k_loop_complete:
        b_offset = 0
        if GEMMFlag.TRANS_B in gemm_desc.flags:
            b_offset = (
                gemm_desc.ldb * gemm_desc.k * micro_kernel_config.datatype_size_in2
            )
        else:
            b_offset = gemm_desc.ldb * micro_kernel_config.datatype_size_in2

        generated_code.insert(
            x86.ops.RI_SubOp(
                generated_code.current_val_by_reg[gp_reg_mapping.gp_reg_b],
                b_offset,
                register_out=gp_reg_mapping.gp_reg_b,
            )
        )


def libxsmm_generator_gemm_get_blocking_and_mask(
    range: int, max_block: int, nomask_block: int, block: int
) -> tuple[int, bool]:
    """Returns new block and use_mask"""
    use_mask = False
    # TODO: check if there is a better blocking strategy
    if block == max_block:
        block = range % max_block
        if block % nomask_block:
            use_mask = True
    elif block == 0:
        if range >= max_block:
            block = max_block
        else:
            block = range
            # in case we do not have a full vector length, we use masking
            if block % nomask_block:
                use_mask = True
    return block, use_mask
