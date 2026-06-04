from typing import Sequence
from xdsl.dialects.builtin import IntegerAttr
from xdsl.dialects.x86.ops import si32
from xdsl.dialects.x86.registers import AVX512MaskRegisterType, GeneralRegisterType
from xdsl.ir import SSAValue
from xdsl.rewriter import InsertPoint
from xdsl.dialects import x86_scf, x86
from autotuner.libxsmm_gemm.generator_common import (
    GEMMStackVar,
    GPRegMapping,
    MicroKernelConfig,
)
from autotuner.compxsmm_gemm.generator_x86_instructions import (
    compxsmm_x86_instruction_unified_vec_move_ld,
    compxsmm_x86_instruction_unified_vec_move_st,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.compxsmm_gemm.libxsmm_generator import GeneratedCode

from autotuner.libxsmm_gemm.libxsmm_main import (
    GEMMDescriptor,
    GEMMFlag,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


def compxsmm_generator_gemm_init_micro_kernel_config(
    config: MicroKernelConfig, arch: Arch, desc: GEMMDescriptor, use_masking_a_c: bool
) -> MicroKernelConfig:
    compxsmm_generator_gemm_setup_fusion_microkernel_properties(desc, config)
    if arch <= Arch.LIBXSMM_X86_GENERIC or arch > Arch.LIBXSMM_X86_ALLFEAT:
        config.instruction_set = Arch.LIBXSMM_TARGET_ARCH_GENERIC
        config.vector_reg_count = 0
        config.use_masking_a_c = False
        # config.vector_name = "a"
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
                config.a_vmove_instruction = x86.ops.DM_VmovupdOp

            config.b_vmove_instruction = x86.ops.DM_VbroadcastsdOp
            if GEMMFlag.ALIGN_C in desc.flags:
                assert not desc.ldc % config.vector_length
                config.c_vmove_ld_instruction = x86.ops.DM_VmovapdOp
                config.c_vmove_st_instruction = x86.ops.MS_VmovapdOp
                if not use_masking_a_c:
                    config.c_vmove_nts_instruction = x86.ops.MS_VmovntpdOp

                else:
                    config.c_vmove_nts_instruction = x86.ops.MS_VmovapdOp
            else:
                config.c_vmove_ld_instruction = x86.ops.DM_VmovupdOp
                config.c_vmove_st_instruction = x86.ops.MS_VmovupdOp
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
                config.c_vmove_ld_instruction = x86.ops.DM_VmovapsOp
                config.c_vmove_st_instruction = x86.ops.MS_VmovapsOp
                if not use_masking_a_c:
                    config.c_vmove_nts_instruction = x86.ops.MS_VmovapsOp
                else:
                    config.c_vmove_nts_instruction = x86.ops.MS_VmovntpsOp
            else:
                config.c_vmove_ld_instruction = x86.ops.DM_VmovupsOp
                config.c_vmove_st_instruction = x86.ops.MS_VmovupsOp
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


def compxsmm_generator_gemm_setup_fusion_microkernel_properties(
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


GEMM_STACK_VAR_OFFSETS = {
    GEMMStackVar.NONE: 0,
    GEMMStackVar.PFA_PTR: -8,
    GEMMStackVar.PFB_PTR: -16,
    GEMMStackVar.A_OFFS_BRGEMM_PTR: -24,
    GEMMStackVar.B_OFFS_BRGEMM_PTR: -32,
    GEMMStackVar.INT8_SCF: -40,
    GEMMStackVar.GEMM_SCRATCH_PTR: -48,
    GEMMStackVar.ELT_BIAS_PTR: -56,
    GEMMStackVar.ZPT_PTR: -56,
    GEMMStackVar.MXSCALE_PTR: -56,
    GEMMStackVar.ELT_OUTPUT_PTR: -64,
    GEMMStackVar.ELT_RELU_BITMASK_PTR: -72,
    GEMMStackVar.ELT_BUF1: -72,
    GEMMStackVar.ELT_BUF2: -80,
    GEMMStackVar.AUX_VAR: -80,
    GEMMStackVar.BRCOUNT: -88,
    GEMMStackVar.TRANS_EXT_BUF_B: -72,
    GEMMStackVar.TRANS_EXT_BUF_C: -80,
    GEMMStackVar.ELT_BITMAP_PTR: -72,
    GEMMStackVar.ELT_DECOMPRESS_BUF: -80,
    GEMMStackVar.ARG_7: 56,
    GEMMStackVar.ARG_8: 64,
    GEMMStackVar.ARG_9: 72,
    GEMMStackVar.ARG_10: 80,
    GEMMStackVar.TRANSPOSE_PTR: -96,
    GEMMStackVar.AVX2_MASK_PTR: -104,
    GEMMStackVar.SSE_AVX2_LP_HELPER_PTR: -112,
    GEMMStackVar.SCF_BRGEMM_PTR: -112,
    GEMMStackVar.A_EMU_PTR: -120,
    GEMMStackVar.B_EMU_PTR: -128,
    GEMMStackVar.MELTW_STRUCT_PTR: -136,
    GEMMStackVar.A_SCRATCH_PTR: -144,
    GEMMStackVar.C_SCRATCH_PTR: -152,
    GEMMStackVar.C_OUTPUT_PTR: -160,
    GEMMStackVar.BIAS_SCRATCH_PTR: -168,
    GEMMStackVar.ZPT_BRGEMM_PTR: -168,
    GEMMStackVar.BSCALE_BRGEMM_PTR: -96,
    GEMMStackVar.BSCALE_PTR: -152,
    GEMMStackVar.LDA_PTR: -176,
    GEMMStackVar.LDB_PTR: -184,
    GEMMStackVar.LDC_PTR: -192,
}
"""
The stack at exit of setup looks like this:

    10th param (if applicable)                <-- RBP+40
    9th param (if applicable)                 <-- RBP+32
    8th param (if applicable)                 <-- RBP+24
    7th param (if applicable)                 <-- RBP+16
    Return address                            <-- RBP+8
    Entry/saved RBP                           <-- RBP
    prefetch A ptr                            <-- RBP-8
    prefetch B ptr                            <-- RBP-16
    Offset A array ptr                        <-- RBP-24
    Offset B array ptr                        <-- RBP-32
    Int8 scaling factor                       <-- RBP-40
    GEMM_scratch ptr in stack (to be filled)  <-- RBP-48
    Eltwise bias ptr                          <-- RBP-56
    Eltwise output_ptr                        <-- RBP-64
    Eltwise buf1_ptr                          <-- RBP-72
    Eltwise buf2_ptr                          <-- RBP-80
    Batch-reduce count                        <-- RBP-88
    Transpose A ptr                           <-- RBP-96
    AVX2 Mask PTR                             <-- RBP-104
    SSE/AVX2 Low precision helper PTR         <-- RBP-112
    FP32 A EMULATION PTR                      <-- RBP-120
    FP32 B EMULATION PTR                      <-- RBP-128
    MELTW STRUCT PTR                          <-- RBP-136
    A SCRATCH PTR                             <-- RBP-144
    C SCRATCH PTR                             <-- RBP-152
    C OUTPUT PTR                              <-- RBP-160
    BIAS SCRATCH PTR                          <-- RBP-168
    Variable LDA PTR                          <-- RBP-176
    Variable LDB PTR                          <-- RBP-184
    Variable LDC PTR                          <-- RBP-192
"""


def compxsmm_generator_gemm_getval_stack_var(
    generated_code: GeneratedCode,
    micro_kernel_config: MicroKernelConfig,
    stack_var: GEMMStackVar,
    destination: x86.registers.GeneralRegisterType,
    rbp_val: SSAValue[x86.registers.GeneralRegisterType],
) -> SSAValue[GeneralRegisterType]:
    offset = GEMM_STACK_VAR_OFFSETS.get(stack_var, 0)
    # make sure we requested a legal stack var
    assert offset
    return generated_code.insert(
        x86.ops.DM_MovOp(rbp_val, offset, destination=destination)
    ).destination


def compxsmm_generator_gemm_setval_stack_var(
    generated_code: GeneratedCode,
    micro_kernel_config: MicroKernelConfig,
    stack_var: GEMMStackVar,
    source: SSAValue,
    rbp_val: SSAValue[GeneralRegisterType],
) -> None:
    offset = GEMM_STACK_VAR_OFFSETS.get(stack_var, 0)
    # make sure we requested a legal stack var
    assert offset
    generated_code.insert(x86.ops.MS_MovOp(rbp_val, source, offset))


def compxsmm_generator_gemm_kloop(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    *,
    m_blocking: int,
    k_blocking: int,
    max_blocked_k: int,
) -> tuple[x86_scf.ForOp, tuple[SSAValue, ...]]:
    """
    In original, adds three lines of assembly: set counter to 0, add label, add k_blocking to loop counter.
    We create the same ops, but also create a block to hold the body of the loop, and set the insertion point at end of the new block.
    """
    k_arg_reg = gp_reg_mapping.gp_reg_kloop
    m_arg_reg = gp_reg_mapping.gp_reg_mloop

    curr_vals = generated_code.current_val_by_reg
    generated_code.insert(k_init_op := x86.ops.DI_MovOp(0, destination=k_arg_reg))

    existing_block = k_init_op.parent
    assert existing_block is not None
    parent_region = existing_block.parent
    assert parent_region is not None

    # k is passed as lb, so no need to include in iter_args
    # m loop is currently accidentally included in the args even though it's not used in the loop, exclude it for now
    args = tuple(
        val
        for val in generated_code.current_val_by_reg.values()
        if val.type not in (k_arg_reg, m_arg_reg)
    )

    assert k_init_op.next_op is None, (
        "Not sure how this can happen, adding assert to catch later (this assert adding when refactoring to x86_scf generation)"
    )

    kloop_op = generated_code.builder.insert(
        x86_scf.ForOp(
            k_init_op.destination,
            IntegerAttr(max_blocked_k, si32),
            IntegerAttr(k_blocking, si32),
            args,
        )
    )

    # Set up builder to build inside of loop
    generated_code.builder.insertion_point = InsertPoint.at_start(kloop_op.body.block)
    curr_vals.clear()
    curr_vals |= {arg.type: arg for arg in kloop_op.body.block.args}

    return kloop_op, kloop_op.body.block.args


def compxsmm_generator_gemm_footer_kloop(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    gemm_desc: GEMMDescriptor,
    m_blocking: int,
    max_blocked_k: int,
    k_loop_complete: bool,
    a_val: SSAValue,
    b_val: SSAValue,
    c_val: SSAValue,
    rsp_val: SSAValue,
    rbp_val: SSAValue,
    acc_vals: Sequence[SSAValue],
) -> None:
    generated_code.insert(
        yield_op := x86_scf.YieldOp(a_val, b_val, c_val, rsp_val, rbp_val, *acc_vals)
    )

    # Set up builder to build at end of block containing for loop
    kloop_op = yield_op.parent_op()
    assert isinstance(kloop_op, x86_scf.ForOp)
    assert kloop_op.parent is not None
    generated_code.builder.insertion_point = InsertPoint.at_end(kloop_op.parent)
    curr_vals = generated_code.current_val_by_reg
    curr_vals.clear()
    curr_vals |= {arg.type: arg for arg in kloop_op.results}

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


def compxsmm_generator_gemm_get_blocking_and_mask(
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


def compxsmm_generator_gemm_setup_stack_frame(
    generated_code: GeneratedCode,
    desc: GEMMDescriptor,
    gp_reg_mapping: GPRegMapping,
    config: MicroKernelConfig,
) -> tuple[SSAValue[GeneralRegisterType], SSAValue[GeneralRegisterType]]:
    """
    Sets up stack frame with reserved offsets.

    Returns RBP and RSP.
    """
    temp_reg = x86.registers.R10
    l_is_Ai4_Bi8_gemm = desc.is_Ai4_Bi8_gemm()
    l_is_Amxfp4_Bbf16_gemm = desc.is_Amxfp4_Bbf16_gemm()
    l_is_Amxfp4_Bfp32_gemm = desc.is_Amxfp4_Bfp32_gemm()
    l_is_Amxfp4_Bi8_gemm = desc.is_Amxfp4_Bi8_gemm()
    l_is_Abf8_Bbf16_gemm = desc.is_Abf8_Bbf16_gemm()
    l_is_Abf8_Bf16_gemm = desc.is_Abf8_Bf16_gemm()
    l_is_Ahf8_Bbf16_gemm = desc.is_Ahf8_Bbf16_gemm()

    rbp_val: SSAValue[GeneralRegisterType] = generated_code.insert(
        x86.ops.GetRegisterOp(x86.registers.RBP)
    ).result
    rsp_val: SSAValue[GeneralRegisterType] = generated_code.insert(
        x86.ops.GetRegisterOp(x86.registers.RSP)
    ).result
    rsp_val = generated_code.insert(x86.ops.S_PushOp(rsp_val, rbp_val)).rsp_out
    rbp_val = generated_code.insert(
        x86.ops.DS_MovOp(rsp_val, destination=x86.registers.RBP)
    ).destination
    rsp_val = generated_code.insert(x86.ops.RI_SubOp(rsp_val, 192)).register_out

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
    # Use builder directly to avoid inserting temp_reg into current_registers
    temp = generated_code.builder.insert(
        x86.ops.DI_MovOp(IntegerAttr(-64, si32), destination=temp_reg)
    ).destination
    rsp_val = generated_code.insert(x86.ops.RS_AndOp(rsp_val, temp)).register_out

    # Now allocate in stack required GEMM scratch if necessary

    compxsmm_generator_gemm_setup_stack_frame_allocate_scratch(
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

    return (rbp_val, rsp_val)


def compxsmm_generator_gemm_destroy_stack_frame(
    generated_code: GeneratedCode,
    desc: GEMMDescriptor,
    gp_reg_mapping: GPRegMapping,
    config: MicroKernelConfig,
    rbp_val: SSAValue[GeneralRegisterType],
) -> SSAValue[GeneralRegisterType]:
    rsp = generated_code.insert(
        x86.ops.DS_MovOp(rbp_val, destination=x86.registers.RSP)
    ).destination
    rbp_val = generated_code.insert(
        x86.ops.D_PopOp(rsp, destination=x86.registers.RBP)
    ).destination
    return rbp_val


def compxsmm_generator_gemm_setup_stack_frame_allocate_scratch(
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


def compxsmm_generator_gemm_header_nloop(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    *,
    n_init: int,
    n_blocking: int,
    n_done: int,
) -> x86_scf.ForOp:
    """
    In original, adds three lines of assembly: set counter to n_init, add label, add n_blocking to loop counter.
    We create the same ops, but also create a block to hold the body of the loop, and set the insertion point at end of the new block.
    """
    n_arg_reg = gp_reg_mapping.gp_reg_nloop

    curr_vals = generated_code.current_val_by_reg
    generated_code.insert(n_init_op := x86.ops.DI_MovOp(n_init, destination=n_arg_reg))

    existing_block = n_init_op.parent
    assert existing_block is not None
    parent_region = existing_block.parent
    assert parent_region is not None

    # n is passed as lb, so no need to include in iter_args
    args = tuple(
        val
        for val in generated_code.current_val_by_reg.values()
        if val.type != n_arg_reg
    )

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
    curr_vals.clear()
    curr_vals |= {arg.type: arg for arg in body_block.args}

    return nloop_op


def compxsmm_generator_gemm_footer_nloop(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    *,
    n_blocking: int,
) -> None:
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
        c = generated_code.current_val_by_reg[gp_reg_mapping.gp_reg_c]
        generated_code.insert(
            x86.ops.RI_AddOp(
                c,
                (n_blocking * (desc.ldc) * (micro_kernel_config.datatype_size_out))
                - ((desc.m) * (micro_kernel_config.datatype_size_out)),
            )
        )

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

        rbp_val = generated_code.get_val(x86.registers.RBP)
        output_ptr = compxsmm_generator_gemm_getval_stack_var(
            generated_code,
            micro_kernel_config,
            GEMMStackVar.ELT_OUTPUT_PTR,
            gp_reg_mapping.gp_reg_help_0,
            rbp_val,
        )
        output_ptr = generated_code.insert(
            x86.ops.RI_AddOp(output_ptr, (n_blocking * (desc.ldc) * 2) - ((desc.m) * 2))
        ).register_out
        compxsmm_generator_gemm_setval_stack_var(
            generated_code,
            micro_kernel_config,
            GEMMStackVar.ELT_OUTPUT_PTR,
            output_ptr,
            rbp_val,
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

        a = generated_code.current_val_by_reg[gp_reg_mapping.gp_reg_a]
        b = generated_code.current_val_by_reg[gp_reg_mapping.gp_reg_b]
        b = generated_code.insert(x86.ops.RI_AddOp(b, b_offset)).register_out

        if GEMMFlag.DECOMPRESS_A_VIA_BITMASK in desc.flags:
            raise NotImplementedError
        else:
            a = generated_code.insert(
                x86.ops.RI_SubOp(
                    a,
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
    yielded_arg_types = body_block.arg_types[1:]
    yielded_args = tuple(
        generated_code.current_val_by_reg[val_type] for val_type in yielded_arg_types
    )

    generated_code.insert(x86_scf.YieldOp(*yielded_args))

    # Set up builder to build after loop
    nloop_op = body_block.parent_op()
    assert isinstance(nloop_op, x86_scf.ForOp), nloop_op

    curr_vals = generated_code.current_val_by_reg
    curr_vals.clear()
    curr_vals |= {arg.type: arg for arg in nloop_op.results}

    generated_code.builder.insertion_point = InsertPoint.after(nloop_op)


def compxsmm_generator_gemm_header_mloop(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    *,
    m_init: int,
    m_blocking: int,
    m_done: int,
) -> x86_scf.ForOp:
    """
    In original, adds three lines of assembly: set counter to m_init, add label, add m_blocking to loop counter.
    We create the same ops, but also create a block to hold the body of the loop, and set the insertion point at end of the new block.
    """
    m_arg_reg = gp_reg_mapping.gp_reg_mloop
    n_arg_reg = gp_reg_mapping.gp_reg_nloop

    curr_vals = generated_code.current_val_by_reg
    generated_code.insert(m_init_op := x86.ops.DI_MovOp(m_init, destination=m_arg_reg))

    existing_block = m_init_op.parent
    assert existing_block is not None
    parent_region = existing_block.parent
    assert parent_region is not None

    # m is passed as lb, so no need to include in iter_args
    # n loop is currently accidentally included in the args even though it's not used in the loop, exclude it for now
    args = tuple(
        val
        for val in generated_code.current_val_by_reg.values()
        if val.type not in (m_arg_reg, n_arg_reg)
    )

    mloop_op = generated_code.builder.insert(
        x86_scf.ForOp(
            m_init_op.destination,
            IntegerAttr(m_done, si32),
            IntegerAttr(m_blocking, si32),
            args,
        )
    )

    body_block = mloop_op.body.block

    generated_code.builder.insertion_point = InsertPoint.at_start(body_block)
    curr_vals.clear()
    curr_vals |= {arg.type: arg for arg in body_block.args}

    return mloop_op


def compxsmm_generator_gemm_footer_mloop(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    *,
    m_blocking: int,
) -> None:
    # k packing factor for VNNI
    k_pack_factor = 1
    is_Ai4_Bf16_gemm = (
        GEMMFlag.INTERPRETE_A_AS_INT4_VNNI2 in desc.flags
        and Datatype.I8 == desc.datatype.a
        and Datatype.F16 == desc.datatype.b
        and desc.datatype.c in (Datatype.F16, Datatype.F32)
    )

    is_Ai4_Bi8_gemm = desc.is_Ai4_Bi8_gemm()
    is_Ai2_Bi8_gemm = desc.is_Ai2_Bi8_gemm()
    is_Ai1_Bi8_gemm = desc.is_Ai1_Bi8_gemm()
    is_Amxfp4_Bfp32_gemm = desc.is_Amxfp4_Bfp32_gemm()
    is_Amxfp4_Bi8_gemm = desc.is_Amxfp4_Bi8_gemm()
    is_Amxfp4_Bbf16_gemm = desc.is_Amxfp4_Bbf16_gemm()
    k_scale = (
        2
        if (
            is_Ai4_Bf16_gemm
            or is_Ai4_Bi8_gemm
            or is_Amxfp4_Bfp32_gemm
            or is_Amxfp4_Bbf16_gemm
            or is_Amxfp4_Bi8_gemm
        )
        else 4
        if is_Ai2_Bi8_gemm
        else 8
        if is_Ai1_Bi8_gemm
        else 1
    )
    a_adjust = 4 if is_Ai2_Bi8_gemm else 8 if is_Ai1_Bi8_gemm else 1

    # for VNNI we are stepping through to pack ks
    if GEMMFlag.VNNI_A in desc.flags:
        raise NotImplementedError

    # Advance C pointer
    c = generated_code.current_val_by_reg[gp_reg_mapping.gp_reg_c]
    c = generated_code.insert(
        x86.ops.RI_AddOp(c, m_blocking * micro_kernel_config.datatype_size_out)
    ).register_out

    if (
        (desc.datatype.c in (Datatype.F16, Datatype.F32))
        and desc.datatype.b == Datatype.F16
        and desc.datatype.a == Datatype.I8
        and (
            GEMMFlag.USE_COL_VEC_SCF in desc.flags
            or GEMMFlag.USE_COL_VEC_ZPT in desc.flags
        )
    ):
        raise NotImplementedError

    if is_Ai4_Bi8_gemm and GEMMFlag.USE_COL_VEC_ZPT in desc.flags:
        raise NotImplementedError

    if (
        desc.datatype.c in (Datatype.BF16, Datatype.F32)
        and desc.datatype.b == Datatype.BF16
        and desc.datatype.a == Datatype.I8
        and GEMMFlag.USE_COL_VEC_SCF in desc.flags
    ):
        raise NotImplementedError

    # Also adjust eltwise pointers
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
        raise NotImplementedError

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

    a_offset = (
        desc.k * micro_kernel_config.datatype_size_in * desc.lda // k_scale
        - m_blocking * micro_kernel_config.datatype_size_in * k_pack_factor // a_adjust
    )
    a = generated_code.current_val_by_reg[gp_reg_mapping.gp_reg_a]

    # A prefetch
    if desc.prefetch == GEMMPrefetchType.AL2:
        raise NotImplementedError

    # Adcance A pointer
    if GEMMFlag.BATCH_REDUCE_ADDRESS in desc.flags:
        raise NotImplementedError
    else:
        if is_Amxfp4_Bfp32_gemm or is_Amxfp4_Bbf16_gemm or is_Amxfp4_Bi8_gemm:
            raise NotImplementedError
        if is_Ai4_Bi8_gemm and GEMMFlag.USE_MxK_ZPT in desc.flags:
            raise NotImplementedError

        if GEMMFlag.DECOMPRESS_A_VIA_BITMASK in desc.flags:
            raise NotImplementedError
        else:
            a = generated_code.insert(x86.ops.RI_SubOp(a, a_offset)).register_out

    # Insert yield op for resulting registers
    body_block = generated_code.builder.insertion_point.block
    yielded_arg_types = body_block.arg_types[1:]
    yielded_args = tuple(
        generated_code.current_val_by_reg[val_type] for val_type in yielded_arg_types
    )
    generated_code.insert(x86_scf.YieldOp(*yielded_args))


def compxsmm_generator_gemm_load_C(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    m_blocking: int,
    n_blocking: int,
) -> None:
    # register blocking counter in n
    n = 0
    # register blocking counter in m
    m = 0
    _is_Ai4_Bf16_gemm = (
        (GEMMFlag.INTERPRETE_A_AS_INT4_VNNI2 in desc.flags)
        and (desc.datatype.a == Datatype.I8)
        and (desc.datatype.b == Datatype.F16)
        and (desc.datatype.c == Datatype.F16 or desc.datatype.c == Datatype.F32)
    )

    is_Ai8_Bf16_gemm = (
        (desc.datatype.a == Datatype.I8)
        and (desc.datatype.b == Datatype.F16)
        and (desc.datatype.c == Datatype.F16 or desc.datatype.c == Datatype.F32)
    )

    is_Amxfp4_Bbf16_gemm = desc.is_Amxfp4_Bbf16_gemm()
    is_Amxfp4_Bfp32_gemm = desc.is_Amxfp4_Bfp32_gemm()
    is_Amxfp4_Bi8_gemm = desc.is_Amxfp4_Bi8_gemm()

    is_Ai8_Bbf16_gemm = (
        (desc.datatype.a == Datatype.I8 and not is_Amxfp4_Bbf16_gemm)
        and (desc.datatype.b == Datatype.BF16)
        and (desc.datatype.c == Datatype.BF16 or desc.datatype.c == Datatype.F32)
    )

    is_Ai4_Bi8_gemm = desc.is_Ai4_Bi8_gemm()

    load_scf_vector = (GEMMFlag.USE_COL_VEC_SCF in desc.flags) and (
        is_Ai8_Bf16_gemm or is_Ai8_Bbf16_gemm
    )

    load_zpt_vector = (GEMMFlag.USE_COL_VEC_ZPT in desc.flags) and (
        is_Ai8_Bf16_gemm or is_Ai4_Bi8_gemm
    )

    assert micro_kernel_config.vector_length

    # deriving register blocking from kernel config
    m_blocking = (
        m_blocking // micro_kernel_config.vector_length
        if (m_blocking % micro_kernel_config.vector_length == 0)
        else (m_blocking // micro_kernel_config.vector_length) + 1
    )
    # start register of accumulator
    vec_reg_acc_start = micro_kernel_config.vector_reg_count - n_blocking * m_blocking

    if load_scf_vector:
        raise NotImplementedError

    if load_zpt_vector:
        raise NotImplementedError

    # load C accumulator
    if GEMMFlag.BETA_0 not in desc.flags:
        # Beta=1
        if (
            micro_kernel_config.instruction_set < Arch.LIBXSMM_X86_AVX
            and Datatype.BF16 == desc.datatype.ab
            and Datatype.BF16 == desc.datatype.c
        ):
            raise NotImplementedError
        elif (
            (
                Arch.LIBXSMM_X86_AVX
                <= micro_kernel_config.instruction_set
                < Arch.LIBXSMM_X86_AVX512_VL256_SKX
            )
            and (Datatype.BF16 == desc.datatype.ab and Datatype.BF16 == desc.datatype.c)
        ) or (
            (
                generated_code.arch >= Arch.LIBXSMM_X86_AVX2
                and (is_Amxfp4_Bbf16_gemm or is_Amxfp4_Bi8_gemm)
            )
            and Datatype.BF16 == desc.datatype.c
        ):
            raise NotImplementedError
        elif (
            (
                Arch.LIBXSMM_X86_AVX512_SKX
                <= micro_kernel_config.instruction_set
                <= Arch.LIBXSMM_X86_ALLFEAT
            )
            or (
                Arch.LIBXSMM_X86_AVX512_VL256_SKX
                <= micro_kernel_config.instruction_set
                <= Arch.LIBXSMM_X86_AVX512_SKX
            )
        ) and (
            (Datatype.BF16 == desc.datatype.ab and Datatype.BF16 == desc.datatype.c)
            or (is_Ai8_Bbf16_gemm and Datatype.BF16 == desc.datatype.c)
        ):
            raise NotImplementedError
        elif (
            (
                Arch.LIBXSMM_X86_AVX512_VL256_SKX
                <= generated_code.arch
                <= Arch.LIBXSMM_X86_ALLFEAT
            )
            and (
                (Datatype.BF8 == desc.datatype.ab and Datatype.BF8 == desc.datatype.c)
                or (
                    Datatype.BF16 == desc.datatype.ab
                    and Datatype.BF8 == desc.datatype.c
                )
                or (
                    Datatype.F32 == desc.datatype.ab and Datatype.BF8 == desc.datatype.c
                )
            )
        ) or (
            (
                Arch.LIBXSMM_X86_AVX512_VL256_SKX
                <= generated_code.arch
                <= Arch.LIBXSMM_X86_ALLFEAT
            )
            and (
                (Datatype.HF8 == desc.datatype.ab and Datatype.HF8 == desc.datatype.c)
                or (
                    Datatype.BF16 == desc.datatype.ab
                    and Datatype.HF8 == desc.datatype.c
                )
                or (
                    Datatype.F32 == desc.datatype.ab and Datatype.HF8 == desc.datatype.c
                )
            )
        ):
            raise NotImplementedError
        elif (
            Datatype.I8 == desc.datatype.ab
            and Datatype.F32 == desc.datatype.c
            and not is_Amxfp4_Bi8_gemm
        ):
            raise NotImplementedError
        else:
            if (
                Arch.LIBXSMM_X86_AVX
                <= micro_kernel_config.instruction_set
                < Arch.LIBXSMM_X86_AVX512_VL256_SKX
            ):
                # in case of AVX/AVX2 we need to load the mask into an ymm
                raise NotImplementedError

            if (
                Datatype.F32 == desc.datatype.c
                and Datatype.F16 == desc.datatype.comp
                and generated_code.arch >= Arch.LIBXSMM_X86_AVX512_SPR
            ):
                # In this case we have to split loading C in 2 chunks of vlen/2 each
                raise
            else:
                # adding to C, so let's load C
                for n in range(n_blocking):
                    for m in range(m_blocking):
                        vname_load = micro_kernel_config.vector_name
                        match vname_load:
                            case "x":
                                dest_type = x86.registers.SSERegisterType
                            case "y":
                                dest_type = x86.registers.AVX2RegisterType
                            case "z":
                                dest_type = x86.registers.AVX512RegisterType
                        c_vec_reg = dest_type.from_index(
                            vec_reg_acc_start + m + (m_blocking * n)
                        )
                        use_masking = (
                            micro_kernel_config.use_masking_a_c
                            if (m == (m_blocking - 1))
                            else False
                        )
                        if (
                            micro_kernel_config.instruction_set >= Arch.LIBXSMM_X86_AVX
                            and use_masking
                        ):
                            mask_reg = AVX512MaskRegisterType.from_index(
                                2
                                if (
                                    is_Amxfp4_Bbf16_gemm
                                    or is_Amxfp4_Bfp32_gemm
                                    or is_Amxfp4_Bi8_gemm
                                )
                                else 1
                            )
                            mask_val = generated_code.get_val(mask_reg)
                            _mask_const = None
                        else:
                            mask_reg = None
                            mask_val = None
                            _mask_const = m_blocking % micro_kernel_config.vector_length

                        if (
                            Datatype.F32 == desc.datatype.comp
                            and Datatype.F16 == desc.datatype.c
                        ) or (
                            Datatype.F16 == desc.datatype.comp
                            and Datatype.F16 == desc.datatype.c
                            and generated_code.arch < Arch.LIBXSMM_X86_AVX512_SPR
                        ):
                            raise NotImplementedError

                        # we only mask the last m-blocked load

                        compxsmm_x86_instruction_unified_vec_move_ld(
                            generated_code,
                            micro_kernel_config.c_vmove_ld_instruction,
                            generated_code.get_val(gp_reg_mapping.gp_reg_c),
                            None,
                            0,
                            ((n * desc.ldc) + (m * (micro_kernel_config.vector_length)))
                            * (micro_kernel_config.datatype_size_out),
                            c_vec_reg,
                            use_masking,
                            mask_val,
                            False,
                        )
                        if (
                            Datatype.F16 == desc.datatype.c
                            and Datatype.F32 == desc.datatype.comp
                        ):
                            raise NotImplementedError

                        if (
                            Datatype.F32 == desc.datatype.c
                            and Datatype.F16 == desc.datatype.comp
                        ):
                            raise NotImplementedError

            # Check if we have to add bias
            if micro_kernel_config.fused_scolbias:
                raise NotImplementedError
            if micro_kernel_config.fused_hcolbias:
                raise NotImplementedError
    else:
        if micro_kernel_config.fused_scolbias:
            raise NotImplementedError
        elif micro_kernel_config.fused_bcolbias:
            raise NotImplementedError
        elif micro_kernel_config.fused_hcolbias:
            raise NotImplementedError
        elif micro_kernel_config.fused_b8colbias:
            raise NotImplementedError
        elif micro_kernel_config.fused_h8colbias:
            raise NotImplementedError
        else:
            # overwriting C, so let's xout accumulator
            for n in range(n_blocking):
                for m in range(m_blocking):
                    if generated_code.arch >= Arch.LIBXSMM_X86_AVX:
                        ...
                        # libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                        #     i_micro_kernel_config->vxor_instruction,
                        #     i_micro_kernel_config->vector_name,
                        #     l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
                        #     l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
                        #     l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
                    else:
                        ...
                        # libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                        #     i_micro_kernel_config->vxor_instruction,
                        #     i_micro_kernel_config->vector_name,
                        #     l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
                        #     l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );


def compxsmm_generator_gemm_store_C(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    m_blocking: int,
    n_blocking: int,
) -> None:
    # deriving register blocking from kernel config
    is_Amxfp4_Bbf16_gemm = desc.is_Amxfp4_Bbf16_gemm()
    is_Amxfp4_Bfp32_gemm = desc.is_Amxfp4_Bfp32_gemm()
    is_Amxfp4_Bi8_gemm = desc.is_Amxfp4_Bi8_gemm()

    m_blocking = (
        (m_blocking // micro_kernel_config.vector_length)
        if m_blocking % micro_kernel_config.vector_length == 0
        else (m_blocking // micro_kernel_config.vector_length + 1)
    )
    # start register of accumulator
    vec_reg_acc_start = micro_kernel_config.vector_reg_count - n_blocking * m_blocking
    # select store instruction */
    if GEMMFlag.ALIGN_C_NTS_HINT in desc.flags:
        raise NotImplementedError
    else:
        vstore = micro_kernel_config.c_vmove_st_instruction
        assert vstore is not None
    # register blocking counter in n- and m-direction
    n = 0
    m = 0

    # libxsmm_micro_kernel_config l_micro_kernel_config_mod;
    # libxsmm_micro_kernel_config *const i_micro_kernel_config_mod = (libxsmm_micro_kernel_config*)&l_micro_kernel_config_mod;
    # memcpy(i_micro_kernel_config_mod, i_micro_kernel_config, sizeof(libxsmm_micro_kernel_config));

    if (
        (micro_kernel_config.instruction_set < Arch.LIBXSMM_X86_AVX)
        or (
            Arch.LIBXSMM_X86_AVX2
            <= micro_kernel_config.instruction_set
            < Arch.LIBXSMM_X86_AVX512_VL256_SKX
        )
        or (
            micro_kernel_config.instruction_set
            in (
                Arch.LIBXSMM_X86_AVX512_SKX,
                Arch.LIBXSMM_X86_AVX512_CLX,
                Arch.LIBXSMM_X86_AVX512_VL256_CLX,
                Arch.LIBXSMM_X86_AVX512_VL256_SKX,
            )
        )
    ) and (Datatype.BF16 == desc.datatype.ab and Datatype.BF16 == desc.datatype.c):
        raise NotImplementedError
    elif (
        (
            Arch.LIBXSMM_X86_AVX512_CPX
            <= micro_kernel_config.instruction_set
            <= Arch.LIBXSMM_X86_ALLFEAT
        )
        or (
            Arch.LIBXSMM_X86_AVX512_VL256_CPX
            <= micro_kernel_config.instruction_set
            < Arch.LIBXSMM_X86_AVX512_SKX
        )
    ) and ((Datatype.BF16 == desc.datatype.ab) and (Datatype.BF16 == desc.datatype.ab)):
        raise NotImplementedError
    elif (
        (
            Arch.LIBXSMM_X86_AVX512_VL256_SKX
            <= micro_kernel_config.instruction_set
            <= Arch.LIBXSMM_X86_ALLFEAT
        )
        and (desc.datatype.ab in (Datatype.BF8, Datatype.BF16, Datatype.F32))
        and Datatype.BF8 == desc.datatype.c
    ) or (
        (
            Arch.LIBXSMM_X86_AVX512_VL256_SKX
            <= micro_kernel_config.instruction_set
            <= Arch.LIBXSMM_X86_ALLFEAT
        )
        and (desc.datatype.ab in (Datatype.HF8, Datatype.BF16, Datatype.F32))
        and Datatype.HF8 == desc.datatype.c
    ):
        raise NotImplementedError
    elif (
        Arch.LIBXSMM_X86_AVX2
        <= micro_kernel_config.instruction_set
        <= Arch.LIBXSMM_X86_ALLFEAT
    ) and (
        (Datatype.I8 == desc.datatype.ab)
        and (Datatype.F32 == desc.datatype.c)
        and (is_Amxfp4_Bi8_gemm == 0)
    ):
        raise NotImplementedError
    elif (
        micro_kernel_config.instruction_set < Arch.LIBXSMM_X86_AVX
        and Datatype.I8 == desc.datatype.ab
        and Datatype.F32 == desc.datatype.c
    ):
        raise NotImplementedError
    else:
        # storing C accumulator
        _relu_bitmask_gpr = gp_reg_mapping.gp_reg_help_2
        _scratch_gpr = gp_reg_mapping.gp_reg_help_2
        _aux_gpr = gp_reg_mapping.gp_reg_help_1
        _zero_vreg = 0
        _aux_vreg = (
            2
            if (is_Amxfp4_Bbf16_gemm or is_Amxfp4_Bfp32_gemm or is_Amxfp4_Bi8_gemm)
            else 1
        )
        _mask_gpr = gp_reg_mapping.gp_reg_help_0
        _sse_scratch_gpr = gp_reg_mapping.gp_reg_help_0

        if (
            Arch.LIBXSMM_X86_AVX
            <= micro_kernel_config.instruction_set
            < Arch.LIBXSMM_X86_AVX512_VL256_SKX
            and micro_kernel_config.use_masking_a_c
        ):
            # in case of AVX/AVX2 we need to load the mask into an ymm
            raise NotImplementedError

        # Check out if fusion has to be applied
        if micro_kernel_config.fused_relu_nobitmask or micro_kernel_config.fused_relu:
            raise NotImplementedError
        elif micro_kernel_config.fused_sigmoid:
            raise NotImplementedError

        if (
            Datatype.F32 == desc.datatype.c
            and Datatype.F16 == desc.datatype.comp
            and Arch.LIBXSMM_X86_AVX512_SPR <= generated_code.arch
        ):
            # In this case we have to split store C in 2 chunks of vlen/2 each
            raise NotImplementedError

        else:
            bf16cvt_replacement = 0
            if Datatype.BF16 == desc.datatype.c and Datatype.F32 == desc.datatype.comp:
                raise NotImplementedError

            for n in range(n_blocking):
                for m in range(m_blocking):
                    reg_X = vec_reg_acc_start + m + m_blocking * n

                    use_masking = (
                        micro_kernel_config.use_masking_a_c
                        if (m == (m_blocking - 1))
                        else False
                    )
                    if (
                        micro_kernel_config.instruction_set >= Arch.LIBXSMM_X86_AVX
                        and use_masking
                    ):
                        mask_reg = AVX512MaskRegisterType.from_index(
                            2
                            if (
                                is_Amxfp4_Bbf16_gemm
                                or is_Amxfp4_Bfp32_gemm
                                or is_Amxfp4_Bi8_gemm
                            )
                            else 1
                        )
                        mask_val = generated_code.get_val(mask_reg)
                        _mask_const = None
                    else:
                        mask_reg = None
                        mask_val = None
                        _mask_const = m_blocking % micro_kernel_config.vector_length

                    vname_store = micro_kernel_config.vector_name
                    match vname_store:
                        case "x":
                            dest_type = x86.registers.SSERegisterType
                        case "y":
                            dest_type = x86.registers.AVX2RegisterType
                        case "z":
                            dest_type = x86.registers.AVX512RegisterType

                    c_vec_reg = dest_type.from_index(reg_X)
                    c_vec_val = generated_code.get_val(c_vec_reg)

                    if (
                        micro_kernel_config.fused_relu_nobitmask
                        or micro_kernel_config.fused_relu
                    ):
                        raise NotImplementedError
                    elif micro_kernel_config.fused_sigmoid:
                        raise NotImplementedError

                    if (
                        Arch.LIBXSMM_X86_AVX
                        <= micro_kernel_config.instruction_set
                        < Arch.LIBXSMM_X86_AVX512_VL256_SKX
                        and micro_kernel_config.use_masking_a_c
                        and (m == m_blocking - 1)
                    ):
                        raise NotImplementedError

                    if (
                        Datatype.F32 == desc.datatype.comp
                        and Datatype.F16 == desc.datatype.c
                    ) or (
                        Datatype.F16 == desc.datatype.comp
                        and Datatype.F16 == desc.datatype.c
                        and generated_code.arch < Arch.LIBXSMM_X86_AVX512_SPR
                    ):
                        raise NotImplementedError

                    if (
                        Datatype.F32 == desc.datatype.c
                        and Datatype.F16 == desc.datatype.comp
                    ):
                        raise NotImplementedError

                    if Datatype.BF16 == desc.datatype.c and (
                        desc.datatype.comp in (Datatype.F32, Datatype.I32)
                    ):
                        raise NotImplementedError

                    if (
                        (desc.is_Amxfp4_Bbf16_gemm() or desc.is_Amxfp4_Bi8_gemm())
                        and Datatype.BF16 == desc.datatype.c
                        and micro_kernel_config.use_masking_a_c
                        and (m == m_blocking - 1)
                    ):
                        raise NotImplementedError
                    else:
                        compxsmm_x86_instruction_unified_vec_move_st(
                            generated_code,
                            vstore,
                            generated_code.get_val(gp_reg_mapping.gp_reg_c),
                            None,
                            0,
                            ((n * desc.ldc) + (m * (micro_kernel_config.vector_length)))
                            * (micro_kernel_config.datatype_size_out),
                            c_vec_val,
                            micro_kernel_config.use_masking_a_c
                            if (m == (m_blocking - 1))
                            else False,
                            mask_val,
                            True,
                        )
            if bf16cvt_replacement:
                raise NotImplementedError

        if micro_kernel_config.fused_relu_nobitmask or micro_kernel_config.fused_relu:
            raise NotImplementedError
        elif micro_kernel_config.fused_sigmoid:
            raise NotImplementedError
