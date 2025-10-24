from xdsl.dialects.builtin import IntegerAttr, i64
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


def libxsmm_generator_gemm_setup_stack_frame(
    generated_code: GeneratedCode,
    desc: GEMMDescriptor,
    gp_reg_mapping: GPRegMapping,
    config: MicroKernelConfig,
):
    temp_reg = x86.registers.R10
    l_is_Ai4_Bi8_gemm = desc.is_Ai4_Bi8_gemm()
    l_is_Amxfp4_Bbf16_gemm = desc.is_Amxfp4_Bbf16_gemm()
    l_is_Amxfp4_Bfp32_gemm = desc.is_Amxfp4_Bfp32_gemm()
    l_is_Amxfp4_Bi8_gemm = desc.is_Amxfp4_Bi8_gemm()
    l_is_Abf8_Bbf16_gemm = desc.is_Abf8_Bbf16_gemm()
    l_is_Abf8_Bf16_gemm = desc.is_Abf8_Bf16_gemm()
    l_is_Ahf8_Bbf16_gemm = desc.is_Ahf8_Bbf16_gemm()

    rbp = generated_code.insert(x86.ops.GetRegisterOp(x86.registers.RBP)).result
    rsp = generated_code.insert(x86.ops.GetRegisterOp(x86.registers.RSP)).result
    rsp = generated_code.insert(x86.ops.S_PushOp(rsp, rbp)).rsp_out
    rbp = generated_code.insert(
        x86.ops.DS_MovOp(rsp, destination=x86.registers.RBP)
    ).destination
    rsp = generated_code.insert(x86.ops.RI_SubOp(rsp, 192)).register_out

    # The stack now looks like this:
    #      10th param (if applicable)                <-- RBP+80
    #      9th param (if applicable)                 <-- RBP+72
    #      8th param (if applicable)                 <-- RBP+64
    #      7th param (if applicable)                 <-- RBP+56
    #      Return address                            <-- RBP+48
    #      Calle SAVED-regs                          <-- RBP[+8,+16,+24,+32,+40]
    #      Entry/saved RBP                           <-- RBP
    #      prefetch A ptr                            <-- RBP-8
    #      prefetch B ptr                            <-- RBP-16
    #      Offset A array ptr                        <-- RBP-24
    #      Offset B array ptr                        <-- RBP-32
    #      Int8 scaling factor                       <-- RBP-40
    #      GEMM_scratch ptr in stack (to be filled)  <-- RBP-48
    #      Eltwise bias ptr                          <-- RBP-56
    #      Eltwise output_ptr                        <-- RBP-64
    #      Eltwise buf1_ptr                          <-- RBP-72
    #      Eltwise buf2_ptr                          <-- RBP-80
    #      Batch-reduce count                        <-- RBP-88,
    #      Transpose A ptr                           <-- RBP-96,
    #      AVX2 Mask                                 <-- RBP-104,
    #      SSE/AVX2 low precision helper             <-- RBP-112,
    #      FP32 A EMULATION PTR                      <-- RBP-120,
    #      FP32 B EMULATION PTR                      <-- RBP-128,
    #      MELTW STRUCT PTR                          <-- RBP-136,
    #      A SCRATCH PTR                             <-- RBP-144,
    #      C SCRATCH PTR                             <-- RBP-152,
    #      C OUTPUT PTR                              <-- RBP-160,
    #      BIAS SCRATCH PTR                          <-- RBP-168,
    #      Variable LDA PTR                          <-- RBP-176,
    #      Variable LDB PTR                          <-- RBP-184,
    #      Variable LDC PTR                          <-- RBP-192, RSP
    #

    if (
        (GEMMFlag.USE_XGEMM_EXT_ABI in desc.flags)
        or l_is_Ai4_Bi8_gemm
        or l_is_Amxfp4_Bbf16_gemm
        or l_is_Amxfp4_Bfp32_gemm
        or l_is_Amxfp4_Bi8_gemm
        or l_is_Abf8_Bbf16_gemm
        or l_is_Abf8_Bf16_gemm
        or l_is_Ahf8_Bbf16_gemm
        or config.atrans_gemm_stack_alloc_tensors > 0
        or config.avnni_gemm_stack_alloc_tensors > 0
        or config.avnni_btrans_gemm_stack_alloc_tensors > 0
        or config.atvnni_gemm_stack_alloc_tensors > 0
        or config.atvnni_btrans_gemm_stack_alloc_tensors > 0
        or config.bvnni_btrans_gemm_stack_alloc_tensors > 0
        or config.bf8_gemm_via_stack_alloc_tensors > 0
        or config.hf8_gemm_via_stack_alloc_tensors > 0
    ):
        raise NotImplementedError
    else:
        has_scf = desc.datatype.ab == Datatype.I8 and desc.datatype.c == Datatype.F32
        has_a_scf = (
            desc.datatype.a == Datatype.I8
            and desc.datatype.b == Datatype.F16
            and (desc.datatype.c == Datatype.F16 or desc.datatype.c == Datatype.F32)
        ) or (
            desc.datatype.a == Datatype.I8
            and desc.datatype.b == Datatype.BF16
            and (desc.datatype.c == Datatype.BF16 or desc.datatype.c == Datatype.F32)
        )

        if has_scf or has_a_scf:
            raise NotImplementedError

    # Now align RSP to 64 byte boundary
    temp = generated_code.insert(
        x86.ops.DI_MovOp(IntegerAttr(0xFFFFFFFFFFFFFFC0, i64), destination=temp_reg)
    ).destination
    rsp = generated_code.insert(x86.ops.RS_AndOp(rsp, temp))

    # Now allocate in stack required GEMM scratch if necessary

    libxsmm_generator_gemm_setup_stack_frame_allocate_scratch(
        generated_code, desc, config
    )

    # The stack at exit of setup looks like this:
    #
    #      10th param (if applicable)            <-- RBP+80
    #      9th param (if applicable)             <-- RBP+72
    #      8th param (if applicable)             <-- RBP+64
    #      7th param (if applicable)             <-- RBP+56
    #      Return address                        <-- RBP+48
    #      Callee SAVED-regs                     <-- RBP[+8,+16,+24,+32,+40]
    #      Entry/saved RBP                       <-- RBP
    #      prefetch A ptr                        <-- RBP-8
    #      prefetch B ptr                        <-- RBP-16
    #      Offset A array ptr                    <-- RBP-24
    #      Offset B array ptr                    <-- RBP-32
    #      Int8 scaling factor                   <-- RBP-40
    #      GEMM_scratch ptr in stack             <-- RBP-48
    #      Eltwise bias ptr                      <-- RBP-56
    #      Eltwise output_ptr                    <-- RBP-64
    #      Eltwise buf1_ptr                      <-- RBP-72
    #      Eltwise buf2_ptr                      <-- RBP-80
    #      Batch-reduce count                    <-- RBP-88
    #      Transpose A ptr                       <-- RBP-96
    #      AVX2 Mask                             <-- RBP-104
    #      SSE/AVX2 low precision helper         <-- RBP-112
    #      FP32 A EMULATION PTR                  <-- RBP-120
    #      FP32 B EMULATION PTR                  <-- RBP-128
    #      MELTW STRUCT PTR                      <-- RBP-136
    #      A SCRATCH PTR                         <-- RBP-144
    #      C SCRATCH PTR                         <-- RBP-152
    #      C OUTPUT PTR                          <-- RBP-160
    #      BIAS SCRATCH PTR                      <-- RBP-168
    #      Variable LDA PTR                      <-- RBP-176
    #      Variable LDB PTR                      <-- RBP-184
    #      Variable LDC PTR                      <-- RBP-192, RSP
    #
    #      [ Potential pad for 64b align ]
    #      AVX2 mask, 64b aligned                <-- (RBP-104) contains this address
    #      SSE/AVX2 low precision helper, 64b aligned <-- (RBP-112) contains this address
    #      GEMM scratch, 64b aligned             <-- (RBP-48) contains this address


def libxsmm_generator_gemm_destroy_stack_frame(
    generated_code: GeneratedCode,
    desc: GEMMDescriptor,
    gp_reg_mapping: GPRegMapping,
    config: MicroKernelConfig,
) -> None:
    rbp = generated_code.current_val_by_reg[x86.registers.RBP]
    rsp = generated_code.insert(
        x86.ops.DS_MovOp(rbp, destination=x86.registers.RSP)
    ).destination
    rbp = generated_code.insert(x86.ops.D_PopOp(rsp, destination=x86.registers.RBP))


def libxsmm_generator_gemm_setup_stack_frame_allocate_scratch(
    generated_code: GeneratedCode, desc: GEMMDescriptor, config: MicroKernelConfig
): ...


# LIBXSMM_API_INTERN
# void libxsmm_generator_gemm_setup_stack_frame_allocate_scratch( libxsmm_generated_code*            io_generated_code,
#     const libxsmm_gemm_descriptor*      i_xgemm_desc,
#     libxsmm_micro_kernel_config*        i_micro_kernel_config ) {
#   unsigned int gemm_scratch_size      = 0;
#   unsigned int scratch_pad_size       = 0;
#   unsigned int avx2_mask_size         = 64;
#   unsigned int avx2_ones_size         = 64;
#   short sixteen_ones[16] = { 1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1 };
#   unsigned short avx2_bf16_mask[16] = { 0x0, 0xffff, 0x0, 0xffff,   0x0, 0xffff, 0x0, 0xffff,   0x0, 0xffff, 0x0, 0xffff,   0x0, 0xffff, 0x0, 0xffff };
#   unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
#   unsigned int l_is_Ai2_Bi8_gemm = libxsmm_x86_is_Ai2_Bi8_gemm(i_xgemm_desc);
#   unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
#   unsigned int l_is_Abf8_Bbf16_gemm = libxsmm_x86_is_Abf8_Bbf16_gemm(i_xgemm_desc);
#   unsigned int l_is_Abf8_Bf16_gemm = libxsmm_x86_is_Abf8_Bf16_gemm(i_xgemm_desc);
#   unsigned int l_is_Ahf8_Bbf16_gemm = libxsmm_x86_is_Ahf8_Bbf16_gemm(i_xgemm_desc);

#   if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) && (io_generated_code->arch < LIBXSMM_X86_ALLFEAT)) {
#     i_micro_kernel_config->gemm_scratch_ld = 16;
#     gemm_scratch_size = LIBXSMM_MAX(((i_micro_kernel_config->vnni_format_C > 0) ? (32 * 64 + i_xgemm_desc->n * i_xgemm_desc->m * 4) : (32*64)), 64 * i_micro_kernel_config->gemm_scratch_ld * 4 + 8 * 1024/*i_micro_kernel_config->datatype_size*/);
#   } else {
#     /* Allocate scratch for stashing 32 zmms */
#     if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & i_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) ) {
#       gemm_scratch_size = 32 * 64;
#     }
#     if (i_micro_kernel_config->vnni_format_C > 0) {
#       gemm_scratch_size = 32 * 64 + i_xgemm_desc->n * i_xgemm_desc->m * 4;
#     }
#   }

#   scratch_pad_size  = (gemm_scratch_size % 64 == 0) ? 0 : ((gemm_scratch_size + 63)/64) * 64 - gemm_scratch_size;
#   gemm_scratch_size += scratch_pad_size;

#   if ( (io_generated_code->arch <= LIBXSMM_X86_SSE42) && (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) ) {
#     libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, avx2_ones_size );
#     libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_SSE_AVX2_LP_HELPER_PTR, LIBXSMM_X86_GP_REG_RSP );

#     libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)sixteen_ones, "sixteen_short_ones", 'x', 0);
#     libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_MOVUPS, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'x', 0, 0, 0, 1 );
#   }
#   if ( (io_generated_code->arch <= LIBXSMM_X86_SSE42) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0)) {
#     libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, avx2_ones_size );
#     libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_SSE_AVX2_LP_HELPER_PTR, LIBXSMM_X86_GP_REG_RSP );

#     libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)avx2_bf16_mask, "avx2_bf16_mask", 'x', 0);
#     libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_MOVUPS, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'x', 0, 0, 0, 1 );
#   }
#   if ( (io_generated_code->arch >= LIBXSMM_X86_AVX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) ) {
#     libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, avx2_mask_size );
#     libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AVX2_MASK_PTR, LIBXSMM_X86_GP_REG_RSP );

#     libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, avx2_ones_size );
#     libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_SSE_AVX2_LP_HELPER_PTR, LIBXSMM_X86_GP_REG_RSP );
#   }
#   if ( (io_generated_code->arch == LIBXSMM_X86_AVX2) && (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) ) {
#     libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)sixteen_ones, "sixteen_short_ones", 'y', 0);
#     libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VMOVUPS, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'y', 0, 0, 0, 1 );
#   }
#   if ( (io_generated_code->arch >= LIBXSMM_X86_AVX2) && (((io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0)) || (l_is_Amxfp4_Bbf16_gemm > 0)) ) {
#     libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)avx2_bf16_mask, "avx2_bf16_mask", 'y', 0);
#     libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VMOVUPS, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'y', 0, 0, 0, 1 );
#   }

#   if (gemm_scratch_size > 0) {
#     libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, gemm_scratch_size );
#     libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, LIBXSMM_X86_GP_REG_RSP );
#   }

#   if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) && (l_is_Ai4_Bi8_gemm > 0 || l_is_Ai2_Bi8_gemm > 0 || l_is_Abf8_Bbf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || i_micro_kernel_config->avnni_gemm_sw_pipeline > 0)) {
#     unsigned int l_decompress_dtype = (l_is_Ai4_Bi8_gemm > 0 || l_is_Ai2_Bi8_gemm > 0) ? 1 : 2;
#     unsigned int scratch_a_decompress_size = 2 * (LIBXSMM_MIN(i_xgemm_desc->m, 64) * i_xgemm_desc->k) * l_decompress_dtype;
#     unsigned int scratch_a_decompress_pad  = (scratch_a_decompress_size % 64 == 0) ? 0 : ((scratch_a_decompress_size + 63)/64) * 64 - scratch_a_decompress_size;
#     scratch_a_decompress_size += scratch_a_decompress_pad;
#     libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, scratch_a_decompress_size );
#     libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BUF1, LIBXSMM_X86_GP_REG_RSP );
#   }

#   if ((i_micro_kernel_config->bf8_gemm_via_stack_alloc_tensors > 0) || (i_micro_kernel_config->hf8_gemm_via_stack_alloc_tensors > 0) ||
#       (i_micro_kernel_config->atrans_gemm_stack_alloc_tensors > 0 ) || (i_micro_kernel_config->avnni_gemm_stack_alloc_tensors > 0) ||
#       (i_micro_kernel_config->atvnni_gemm_stack_alloc_tensors > 0) || (i_micro_kernel_config->avnni_btrans_gemm_stack_alloc_tensors > 0) ||
#       (i_micro_kernel_config->atvnni_btrans_gemm_stack_alloc_tensors > 0) ||
#       (i_micro_kernel_config->bvnni_btrans_gemm_stack_alloc_tensors > 0) ) {
#     int is_stride_brgemm  = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0) ? 1 : 0;
#     int is_offset_brgemm  = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0) ? 1 : 0;
#     int is_address_brgemm = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0) ? 1 : 0;
#     int is_brgemm         = ((is_stride_brgemm == 1) || (is_offset_brgemm == 1) || (is_address_brgemm == 1)) ? 1 : 0;
#     unsigned int inp_dtype_size = ( (i_micro_kernel_config->atrans_gemm_stack_alloc_tensors > 0) || (i_micro_kernel_config->avnni_gemm_stack_alloc_tensors > 0) || (i_micro_kernel_config->atvnni_gemm_stack_alloc_tensors > 0) || (i_micro_kernel_config->avnni_btrans_gemm_stack_alloc_tensors > 0 ) || (i_micro_kernel_config->atvnni_btrans_gemm_stack_alloc_tensors > 0 ) || (i_micro_kernel_config->bvnni_btrans_gemm_stack_alloc_tensors > 0) ) ?  LIBXSMM_TYPESIZE(LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC(i_xgemm_desc->datatype)) : 4;
#     unsigned int a_size  = (i_xgemm_desc->m * i_xgemm_desc->k) * inp_dtype_size;
#     unsigned int b_size  = (i_xgemm_desc->k * i_xgemm_desc->n) * inp_dtype_size;
#     unsigned int c_size  = (i_xgemm_desc->m * i_xgemm_desc->n) * 4;
#     unsigned int bias_size = i_xgemm_desc->m * 4;
#     unsigned int a_pad  = (a_size % 64 == 0) ? 0 : ((a_size + 63)/64) * 64 - a_size;
#     unsigned int b_pad  = (b_size % 64 == 0) ? 0 : ((b_size + 63)/64) * 64 - b_size;
#     unsigned int c_pad  = (c_size % 64 == 0) ? 0 : ((c_size + 63)/64) * 64 - c_size;
#     unsigned int bias_pad  = (bias_size % 64 == 0) ? 0 : ((bias_size + 63)/64) * 64 - bias_size;
#     a_size += a_pad;
#     b_size += b_pad;
#     c_size += c_pad;
#     /* Extra scratch for relu bitmask  */
#     if ((i_micro_kernel_config->fused_relu > 0) && ((i_micro_kernel_config->bf8_gemm_via_stack_alloc_tensors > 0) || (i_micro_kernel_config->hf8_gemm_via_stack_alloc_tensors > 0))) {
#       c_size  = (i_xgemm_desc->m * i_xgemm_desc->n) * 4 + ((i_xgemm_desc->m+15)/16) * 16 * i_xgemm_desc->n;
#       c_pad  = (c_size % 64 == 0) ? 0 : ((c_size + 63)/64) * 64 - c_size;
#       c_size += c_pad;
#     }
#     bias_size += bias_pad;
#     if ((i_micro_kernel_config->bf8_gemm_via_stack_alloc_tensors > 0) || (i_micro_kernel_config->hf8_gemm_via_stack_alloc_tensors > 0)) {
#       libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, a_size );
#       libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, LIBXSMM_X86_GP_REG_RSP );
#     }
#     if (((io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) && (io_generated_code->arch < LIBXSMM_X86_ALLFEAT)) && ((i_micro_kernel_config->bf8_gemm_via_stack_alloc_tensors > 0) || (i_micro_kernel_config->hf8_gemm_via_stack_alloc_tensors > 0))) {
#       libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, c_size );
#       libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_C_SCRATCH_PTR, LIBXSMM_X86_GP_REG_RSP );
#       if ((i_micro_kernel_config->fused_b8colbias > 0) || (i_micro_kernel_config->fused_h8colbias > 0 )) {
#         libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, bias_size );
#         libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BIAS_SCRATCH_PTR, LIBXSMM_X86_GP_REG_RSP );
#       }
#     }
#     if (is_brgemm == 0) {
#       if ( i_micro_kernel_config->atrans_gemm_stack_alloc_tensors > 0 ) {
#         libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, a_size );
#         libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, LIBXSMM_X86_GP_REG_RSP );
#       } else if ( i_micro_kernel_config->avnni_gemm_stack_alloc_tensors > 0 ) {
#         libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, a_size );
#         libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, LIBXSMM_X86_GP_REG_RSP );
#       } else if ( (i_micro_kernel_config->avnni_btrans_gemm_stack_alloc_tensors > 0) || (i_micro_kernel_config->atvnni_btrans_gemm_stack_alloc_tensors > 0) ) {
#         libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, a_size );
#         libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, LIBXSMM_X86_GP_REG_RSP );
#         libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, b_size );
#         libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
#       } else if ( i_micro_kernel_config->atvnni_gemm_stack_alloc_tensors > 0 ) {
#         libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, a_size );
#         libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, LIBXSMM_X86_GP_REG_RSP );
#       } else if ( i_micro_kernel_config->bvnni_btrans_gemm_stack_alloc_tensors > 0 ) {
#         libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, b_size );
#         libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
#       } else {
#         libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, a_size );
#         libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_A_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
#         libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, b_size );
#         libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
#       }
#     } else {
#       unsigned int temp_reg = LIBXSMM_X86_GP_REG_R10;
#       if ( i_micro_kernel_config->atrans_gemm_stack_alloc_tensors > 0 ) {
#         libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
#         libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, a_size);
#         libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
#         libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, LIBXSMM_X86_GP_REG_RSP );
#         if (is_offset_brgemm > 0 || is_address_brgemm > 0) {
#           libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
#           libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, b_size);
#           libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
#           libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
#         }
#       } else if ( i_micro_kernel_config->avnni_gemm_stack_alloc_tensors > 0 ) {
#         libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
#         libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, a_size);
#         libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
#         libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, LIBXSMM_X86_GP_REG_RSP );
#         if (is_offset_brgemm > 0 || is_address_brgemm > 0) {
#           libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
#           libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, b_size);
#           libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
#           libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
#         }
#       } else if ( (i_micro_kernel_config->avnni_btrans_gemm_stack_alloc_tensors > 0) || (i_micro_kernel_config->atvnni_btrans_gemm_stack_alloc_tensors > 0) ) {
#         libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
#         libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, a_size);
#         libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
#         libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, LIBXSMM_X86_GP_REG_RSP );
#         libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
#         libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, b_size);
#         libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
#         libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
#       } else if ( i_micro_kernel_config->atvnni_gemm_stack_alloc_tensors > 0 ) {
#         libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
#         libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, a_size);
#         libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
#         libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, LIBXSMM_X86_GP_REG_RSP );
#         if (is_offset_brgemm > 0 || is_address_brgemm > 0) {
#           libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
#           libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, b_size);
#           libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
#           libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
#         }
#       } else if ( i_micro_kernel_config->bvnni_btrans_gemm_stack_alloc_tensors > 0 ) {
#         libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
#         libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, b_size);
#         libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
#         libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
#         if (is_offset_brgemm > 0 || is_address_brgemm > 0) {
#           libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
#           libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, a_size);
#           libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
#           libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_A_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
#         }
#       } else {
#         libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
#         libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, a_size);
#         libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
#         libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_A_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
#         libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
#         libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, b_size);
#         libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
#         libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
#       }
#     }
#     libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, 128 );
#     libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MELTW_STRUCT_PTR, LIBXSMM_X86_GP_REG_RSP );
#   }
# }
