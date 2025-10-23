from xdsl.builder import Builder
from xdsl.dialects import x86
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
    UNALLOCATED_GENERAL,
)
from xdsl.dialects.x86_func import FuncOp, RetOp
from xdsl.rewriter import InsertPoint
from autotuner.libxsmm_gemm.generator_common import (
    GPRegMapping,
    LoopLabelTracker,
    MicroKernelConfig,
)
from autotuner.libxsmm_gemm.generator_gemm_avx512_microkernel import (
    libxsmm_generator_gemm_avx512_kloop_kernel,
)
from autotuner.libxsmm_gemm.generator_gemm_common import (
    libxsmm_generator_gemm_footer_kloop,
    libxsmm_generator_gemm_header_kloop,
)
from autotuner.libxsmm_gemm.generator_x86_instructions import (
    libxsmm_x86_instruction_open_stream_gemm,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode
from autotuner.libxsmm_gemm.libxsmm_main import (
    DescDatatype,
    GEMMDescriptor,
    GEMMFlag,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


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
        gp_reg_mapping.gp_reg_scf = UNALLOCATED_GENERAL  # GP_REG_UNDEF
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
) -> None:
    # micro_kernel_config = MicroKernelConfig()
    # These values may be modified below
    m, n, k, lda, ldb, ldc, dt, flags, prefetch = desc

    is_Ai4_Bf16_gemm = (
        ((desc.flags & GEMMFlag.INTERPRETE_A_AS_INT4_VNNI2) > 0)
        and (Datatype.I8 == dt.a)
        and (Datatype.F16 == dt.b)
        and (dt.c in (Datatype.F16, Datatype.F32))
    )

    is_Ai4_Bi8_gemm = desc.is_Ai4_Bi8_gemm
    is_Ai2_Bi8_gemm = desc.is_Ai2_Bi8_gemm
    is_Ai1_Bi8_gemm = desc.is_Ai1_Bi8_gemm
    is_Amxfp4_Bfp32_gemm = desc.is_Amxfp4_Bfp32_gemm
    is_Amxfp4_Bbf16_gemm = desc.is_Amxfp4_Bbf16_gemm
    is_Amxfp4_Bi8_gemm = desc.is_Amxfp4_Bi8_gemm

    avnni_gemm_stack_alloc_tensors = (
        ((desc.flags & GEMMFlag.TRANS_A) == 0)
        and ((desc.flags & GEMMFlag.VNNI_A) == 0)
        and ((desc.flags & GEMMFlag.TRANS_B) == 0)
        and ((desc.flags & GEMMFlag.VNNI_B) == 0)
        and (k % 2 == 0)
        and (Datatype.BF16 == dt.ab)
    )

    atvnni_gemm_stack_alloc_tensors = (
        ((desc.flags & GEMMFlag.TRANS_A) != 0)
        and ((desc.flags & GEMMFlag.VNNI_A) == 0)
        and ((desc.flags & GEMMFlag.TRANS_B) == 0)
        and ((desc.flags & GEMMFlag.VNNI_B) == 0)
        and (k % 2 == 0)
        and (Datatype.BF16 == dt.ab)
    )

    avnni_btrans_gemm_stack_alloc_tensors = (
        ((desc.flags & GEMMFlag.TRANS_A) == 0)
        and ((desc.flags & GEMMFlag.VNNI_A) == 0)
        and ((desc.flags & GEMMFlag.TRANS_B) != 0)
        and ((desc.flags & GEMMFlag.VNNI_B) == 0)
        and (k % 2 == 0)
        and (Datatype.BF16 == dt.ab)
    )

    atvnni_btrans_gemm_stack_alloc_tensors = (
        ((desc.flags & GEMMFlag.TRANS_A) != 0)
        and ((desc.flags & GEMMFlag.VNNI_A) == 0)
        and ((desc.flags & GEMMFlag.TRANS_B) != 0)
        and ((desc.flags & GEMMFlag.VNNI_B) == 0)
        and (k % 2 == 0)
        and (Datatype.BF16 == dt.ab)
    )

    # Initialize n-blocking variables
    l_n_count = 0  # array counter for blocking arrays
    l_n_done = 0  # progress tracker
    l_n_n = [0, 0]  # blocking sizes for blocks
    l_n_N = [0, 0]  # size of blocks

    adjust_A_pf_ptrs = 0
    adjust_B_pf_ptrs = 0
    l_max_n_blocking = 0

    l_is_Ai8_Bbf16_gemm = (
        (Datatype.I8 == dt.a and not is_Amxfp4_Bbf16_gemm)
        and (Datatype.BF16 == dt.b)
        and (dt.c in (Datatype.BF16, Datatype.F32))
    )

    # TODO: we need to implement a consolidate solution for callee save stuff
    # here we need to handle AMX stuff to allow AMX optimized TPPs to run lower platforms
    if not (
        (
            (desc.flags & GEMMFlag.NO_RESET_TILECONFIG) == 0
            and (desc.flags & GEMMFlag.NO_SETUP_TILECONFIG) == 0
        )
        or (
            (desc.flags & GEMMFlag.NO_RESET_TILECONFIG) != 0
            and (desc.flags & GEMMFlag.NO_SETUP_TILECONFIG) != 0
        )
    ):
        return

    # Make sure we properly adjust A,B prefetch pointers in case of batch-reduce gemm kernel
    if desc.flags & GEMMFlag.BATCH_REDUCE_ADDRESS:
        if desc.prefetch == GEMMPrefetchType.AL2:
            adjust_A_pf_ptrs = 1

    # In case of F16 and IMPLICIT compute set proper compute
    if (
        ((dt.a in (Datatype.I8, Datatype.BF8)) and dt.b == Datatype.F16)
        or (dt.a == Datatype.F16 and dt.b == Datatype.F16)
    ) and dt.comp == Datatype.IMPLICIT:
        # if architecture is AMX (Sapphire Rapids or newer), compute in F16. Otherwise F32.
        if generated_code.arch >= Arch.LIBXSMM_X86_AVX512_SPR:
            # Set desc.datatype fields for computation in F16
            dt = DescDatatype(dt.a, dt.b, dt.c, Datatype.F16)
        else:
            # Set desc.datatype fields for computation in F32
            dt = DescDatatype(dt.a, dt.b, dt.c, Datatype.F32)

    # In case of BF8/HF8 we might need to set different precisions
    if (dt.a == Datatype.BF8 and dt.b in (Datatype.BF8, Datatype.HF8)) or (
        dt.a == Datatype.HF8 and dt.b in (Datatype.BF8, Datatype.HF8)
    ):
        raise NotImplementedError

    # TODO: check if we can make this smarter and don't need two times the same if
    if (
        avnni_gemm_stack_alloc_tensors
        or atvnni_gemm_stack_alloc_tensors
        or avnni_btrans_gemm_stack_alloc_tensors
        or atvnni_btrans_gemm_stack_alloc_tensors
    ):
        flags |= GEMMFlag.VNNI_A

    if atvnni_gemm_stack_alloc_tensors or atvnni_btrans_gemm_stack_alloc_tensors:
        flags &= ~GEMMFlag.TRANS_A
        assert lda % 2 == 0, "LIBXSMM_ERR_VNNI_A"

    if avnni_btrans_gemm_stack_alloc_tensors or atvnni_btrans_gemm_stack_alloc_tensors:
        flags &= ~GEMMFlag.TRANS_B


#   /* define the micro kernel code gen properties */
#   libxsmm_generator_gemm_init_micro_kernel_config( &l_micro_kernel_config, io_generated_code->arch, l_xgemm_desc, 0 );

#   /* setup hf8 / bf8 conversion on stack before GEMM, we need to recheck as we now can update the field in ukernel config, need to use the original GEMM descriptor */
#   if (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
#     l_micro_kernel_config.bf8_gemm_via_stack_alloc_tensors = 1;
#   }

#   if (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
#     l_micro_kernel_config.hf8_gemm_via_stack_alloc_tensors = 1;
#   }

#   /* in case when A needs to be transposed, we need to change temporarily the descriptor dimensions for gemm */
#   if (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) {
#     if ((LIBXSMM_DATATYPE_F32 == (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_ABC_COMMON_PREC(l_xgemm_desc->datatype)) || (LIBXSMM_DATATYPE_F64 == (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_ABC_COMMON_PREC(l_xgemm_desc->datatype))) {
#       l_xgemm_desc->lda = l_xgemm_desc->m;
#       l_xgemm_desc->flags = (unsigned int)((unsigned int)(l_xgemm_desc->flags) & (~LIBXSMM_GEMM_FLAG_TRANS_A));
#       l_micro_kernel_config.atrans_gemm_stack_alloc_tensors = 1;
#     } else {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
#       return;
#     }
#   } else if (((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) && ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0)) {
#     if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) {
#       unsigned int aux_flags = (unsigned int)((unsigned int)l_xgemm_desc->flags & (~LIBXSMM_GEMM_FLAG_TRANS_B));
#       l_xgemm_desc->ldb = l_xgemm_desc->k;
#       l_xgemm_desc->flags = (unsigned int)((unsigned int)aux_flags & (~LIBXSMM_GEMM_FLAG_VNNI_B));
#       l_micro_kernel_config.bvnni_btrans_gemm_stack_alloc_tensors = 1;
#     } else {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
#       return;
#     }
#   }

#   /* handle A VNNI on stack */
#   if ( l_avnni_gemm_stack_alloc_tensors != 0 ) {
#     l_xgemm_desc->lda = l_xgemm_desc->m;
#     l_micro_kernel_config.avnni_gemm_stack_alloc_tensors = 1;
#   }
#   if ( l_atvnni_gemm_stack_alloc_tensors != 0 ) {
#     l_xgemm_desc->lda = l_xgemm_desc->m;
#     l_micro_kernel_config.atvnni_gemm_stack_alloc_tensors = 1;
#   }
#   if ( l_avnni_btrans_gemm_stack_alloc_tensors != 0 ) {
#     l_xgemm_desc->lda = l_xgemm_desc->m;
#     l_xgemm_desc->ldb = l_xgemm_desc->k;
#     l_micro_kernel_config.avnni_btrans_gemm_stack_alloc_tensors = 1;
#   }
#   if ( l_atvnni_btrans_gemm_stack_alloc_tensors != 0 ) {
#     l_xgemm_desc->lda = l_xgemm_desc->m;
#     l_xgemm_desc->ldb = l_xgemm_desc->k;
#     l_micro_kernel_config.atvnni_btrans_gemm_stack_alloc_tensors = 1;
#   }

#   /* block according to the number of available registers or given limits */
#   l_max_n_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_max_n_blocking( &l_micro_kernel_config, l_xgemm_desc, io_generated_code->arch );
# #if 1
#   if (3 < l_max_n_blocking)
# #endif
#   {
#     const unsigned int init_m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking( &l_micro_kernel_config, l_xgemm_desc, io_generated_code->arch, 0 );
#     const unsigned int init_m_blocks = LIBXSMM_UPDIV(init_m_blocking, l_micro_kernel_config.vector_length);
#     unsigned int l_is_Ai8_Bf16_gemm = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype )) &&
#                                             (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype )) &&
#                                             (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype )) ) ? 1 : 0;

#     if (l_is_Ai8_Bf16_gemm > 0) {
#       int l_m_scf_vregs = ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) == 0) ? ((l_is_Ai4_Bf16_gemm > 0) ? 3 + init_m_blocks : 1) : (  (l_is_Ai4_Bf16_gemm > 0) ? 2 + 2*init_m_blocks : init_m_blocks);
#       if ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0) {
#         l_m_scf_vregs += init_m_blocks;
#       }
#       /* In this case we need m vec regs for the scaling factors... */
#       if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && (l_is_Ai4_Bf16_gemm == 0) ) {
#         while ((init_m_blocks * l_max_n_blocking + l_max_n_blocking + 1 + l_m_scf_vregs) > l_micro_kernel_config.vector_reg_count) {
#           l_max_n_blocking--;
#         }
#       } else {
#         while ((init_m_blocks * l_max_n_blocking + init_m_blocks + 1 + l_m_scf_vregs) > l_micro_kernel_config.vector_reg_count) {
#           l_max_n_blocking--;
#         }
#       }
#     } else if (l_is_Ai8_Bbf16_gemm > 0) {
#       int l_m_scf_vregs = ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) == 0) ? 1 : init_m_blocks;
#       /* In this case we need m vec regs for the scaling factors... */
#       while ((init_m_blocks * l_max_n_blocking + init_m_blocks + 2 + l_m_scf_vregs) > l_micro_kernel_config.vector_reg_count) {
#         l_max_n_blocking--;
#       }
#     } else if (l_is_Ai4_Bi8_gemm > 0) {
#       int l_m_zpt_vregs = 3 + init_m_blocks;
#       if ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0 || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0) {
#         l_m_zpt_vregs += init_m_blocks;
#       }
#       while ((init_m_blocks * l_max_n_blocking + init_m_blocks + 1 + l_m_zpt_vregs) > l_micro_kernel_config.vector_reg_count) {
#         l_max_n_blocking--;
#       }
#     } else if (l_is_Ai2_Bi8_gemm > 0) {
#       int l_m_reserved_vregs = 3;
#       while ((init_m_blocks * l_max_n_blocking + init_m_blocks + 1 + l_m_reserved_vregs) > l_micro_kernel_config.vector_reg_count) {
#         l_max_n_blocking--;
#       }
#     } else if (l_is_Ai1_Bi8_gemm > 0) {
#       int l_m_reserved_vregs = 2;
#       while ((init_m_blocks * l_max_n_blocking + init_m_blocks + 1 + l_m_reserved_vregs) > l_micro_kernel_config.vector_reg_count) {
#         l_max_n_blocking--;
#       }
#     } else {
#       if ( (io_generated_code->arch >= LIBXSMM_X86_AVX2_SRF) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ) {
#         while ((init_m_blocks * l_max_n_blocking + l_max_n_blocking + 1) > l_micro_kernel_config.vector_reg_count) {
#           l_max_n_blocking--;
#         }
#       } else {
#         while ((init_m_blocks * l_max_n_blocking + init_m_blocks + 1) > l_micro_kernel_config.vector_reg_count) {
#           l_max_n_blocking--;
#         }
#       }
#     }
#   }
#   if ( l_max_n_blocking == 0 ) {
#     LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
#     return;
#   }
#   libxsmm_compute_equalized_blocking( l_xgemm_desc->n, l_max_n_blocking, &(l_n_N[0]), &(l_n_n[0]), &(l_n_N[1]), &(l_n_n[1]) );

#   /* check that l_n_N1 is non-zero */
#   if ( l_n_N[0] == 0 ) {
#     LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
#     return;
#   }

#   /* implementing load from struct */
#   if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI & l_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) ||
#        ((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & l_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) ) {
#     int l_offset_ptr_a = (int)sizeof(libxsmm_matrix_op_arg);
#     int l_offset_ptr_b = (int)(sizeof(libxsmm_matrix_op_arg) + sizeof(libxsmm_matrix_arg));
#     int l_offset_ptr_c = (int)(sizeof(libxsmm_matrix_op_arg) + 2*sizeof(libxsmm_matrix_arg));

#     /* RDI holds the pointer to the struct, so lets first move this one into R15 */
#     libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_help_1 );
#     /* A pointer */
#     libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
#                                      i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, l_offset_ptr_a, i_gp_reg_mapping->gp_reg_a, 0 );
#     if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) {
#       libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
#                                        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, l_offset_ptr_a + 8, i_gp_reg_mapping->gp_reg_bitmap_a, 0 );
#     }
#     if ( l_is_Amxfp4_Bfp32_gemm > 0  || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0) {
#       libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
#                                        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, l_offset_ptr_a + 16, i_gp_reg_mapping->gp_reg_scf, 0 );
#     }
#     /* B pointer */
#     libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
#                                      i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, l_offset_ptr_b, i_gp_reg_mapping->gp_reg_b, 0 );
#     if (l_is_Amxfp4_Bi8_gemm > 0) {
#       libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
#                                        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, l_offset_ptr_b + 16, i_gp_reg_mapping->gp_reg_zpt, 0 );
#     }

#     /* C pointer */
#     libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
#                                      i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, l_offset_ptr_c, i_gp_reg_mapping->gp_reg_c, 0 );
#     if ( l_xgemm_desc->prefetch != LIBXSMM_GEMM_PREFETCH_NONE ) {
#       /* A prefetch pointer */
#       libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
#                                        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, l_offset_ptr_a + LIBXSMM_MATRIX_ARG_OFFSET_PREFETCH, i_gp_reg_mapping->gp_reg_a_prefetch, 0 );
#       /* B prefetch pointer */
#       libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
#                                        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, l_offset_ptr_b + LIBXSMM_MATRIX_ARG_OFFSET_PREFETCH, i_gp_reg_mapping->gp_reg_b_prefetch, 0 );
#     }
#     /* batch reduce count & offset arrays*/
#     if ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET)) {
#       libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
#                                        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 16, i_gp_reg_mapping->gp_reg_reduce_count, 0 );

#       if ( l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) {
#         libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
#                                          i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, l_offset_ptr_a + 8, i_gp_reg_mapping->gp_reg_a_offset, 0 );
#         libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
#                                          i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, l_offset_ptr_b + 8, i_gp_reg_mapping->gp_reg_b_offset, 0 );
#       }
#     }
#     /* loading scaling factor for ternary C */
#     if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype )) && (l_is_Amxfp4_Bi8_gemm == 0) ) {
#       libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
#                                        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, l_offset_ptr_c + 16, i_gp_reg_mapping->gp_reg_scf, 0 );
#     }

#     /* Load scaling factor for A  */
#     if (((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype )) &&
#         (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype )) &&
#         (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ))) ||
#         ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) && (l_is_Amxfp4_Bbf16_gemm == 0) ) &&
#         (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype )) &&
#         (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype )))) {
#       libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
#                                        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, l_offset_ptr_a + 16, i_gp_reg_mapping->gp_reg_scf, 0 );
#       libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
#                                        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, l_offset_ptr_a + 24, i_gp_reg_mapping->gp_reg_zpt, 0 );
#     }

#     if (l_is_Ai4_Bi8_gemm > 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0) {
#       libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
#                                        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, l_offset_ptr_a + 24, i_gp_reg_mapping->gp_reg_zpt, 0 );
#     }
#   }

#   if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & l_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) ||
#        (l_micro_kernel_config.vnni_format_C > 0) ) {
#     /* Illegal ext_abi when precision is not fp32 or bf16 */
#     if (!(LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ||
#           LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ||
#           LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ||
#           LIBXSMM_DATATYPE_HF8  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ||
#           LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) ) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ILLEGAL_ABI );
#       return;
#     }
#   }

#   /* Setting up the stack frame */
#   libxsmm_generator_gemm_setup_stack_frame( io_generated_code, l_xgemm_desc, i_gp_reg_mapping, &l_micro_kernel_config);

#   /* In this case we store C to scratch */
#   if (l_micro_kernel_config.vnni_format_C > 0) {
#     libxsmm_generator_gemm_setval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_C, i_gp_reg_mapping->gp_reg_c );
#     libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, i_gp_reg_mapping->gp_reg_c );
#     libxsmm_x86_instruction_alu_imm_i64( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_help_1, 32LL * 64LL );
#     libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_c);
#     l_xgemm_desc->ldc = l_xgemm_desc->m;
#   }

#   /* Apply potential opA / opB */
#   libxsmm_generator_gemm_apply_opA_opB( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc, i_xgemm_desc);

#   libxsmm_reset_loop_label_tracker( io_loop_label_tracker );

#   /* generate hoisted BF16 emulation mask for AVX512 */
#   if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) &&
#          ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) &&
#          (io_generated_code->arch != LIBXSMM_X86_AVX512_CPX) &&
#          (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) &&
#          (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT) &&
#          (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX)) {
#     libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
#     libxsmm_x86_instruction_alu_imm_i64( io_generated_code,  LIBXSMM_X86_INSTR_MOVQ,
#                                          i_gp_reg_mapping->gp_reg_help_2, 0xaaaaaaaa );
#     libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mapping->gp_reg_help_2, 3 );
#     libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
#   }

#   /* generate hoisted UU SS i8 emulation mask for AVX512 */
#   if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) &&
#          ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) &&
#          ( ( ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) == 0) && ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) == 0) ) ||
#            ( ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) >  0) && ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) >  0) ) ) &&
#          (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) &&
#          (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT)) {
#     libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
#     libxsmm_x86_instruction_alu_imm_i64( io_generated_code,  LIBXSMM_X86_INSTR_MOVQ,
#                                          i_gp_reg_mapping->gp_reg_help_2, 0x55555555 );
#     libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mapping->gp_reg_help_2, 3 );
#     libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
#   }

#   if ( l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 ) {
#     /* Set to 0 lo mask and to 1 hi mask */
#     float lut_mant[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
#     unsigned int mask_sign[8] = { 8, 8, 8, 8, 8, 8, 8, 8 };
#     l_micro_kernel_config.io_loop_label_tracker = io_loop_label_tracker;
#     libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                          (const unsigned char *) lut_mant ,
#                                                          "vperm_mant",
#                                                          'y',
#                                                          0 );
#     libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                          (const unsigned char *) mask_sign ,
#                                                          "vperm_sign",
#                                                          'y',
#                                                          1 );
#   }

#   if ( l_is_Amxfp4_Bi8_gemm > 0 ) {
#     /* Set to 0 lo mask and to 1 hi mask */
#     char lut_mxfp4[32] = { 0, 11, 21, 32, 42, 64, 85, 127, 0, (char)-11, (char)-21, (char)-32, (char)-42, (char)-64, (char)-85, (char)-127,
#                            0, 11, 21, 32, 42, 64, 85, 127, 0, (char)-11, (char)-21, (char)-32, (char)-42, (char)-64, (char)-85, (char)-127 };
#     unsigned int mask_idx[8] = { 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f };
#     l_micro_kernel_config.io_loop_label_tracker = io_loop_label_tracker;
#     libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                          (const unsigned char *) lut_mxfp4 ,
#                                                          "vperm_lut",
#                                                          'y',
#                                                          0 );
#     libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                          (const unsigned char *) mask_idx ,
#                                                          "vmask_idx",
#                                                          'y',
#                                                          1 );
#   }

#   if (l_is_Ai4_Bf16_gemm > 0) {
#     unsigned int l_use_perm_based_cvt = (io_generated_code->arch > LIBXSMM_X86_AVX2 && io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1;
#     if (l_use_perm_based_cvt > 0) {
#       unsigned int lut_f32[16] = {0x00000000, 0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000, 0x40e00000, 0x41000000, 0x41100000, 0x41200000, 0x41300000, 0x41400000, 0x41500000, 0x41600000, 0x41700000};
#       unsigned int signed_lut_f32[16] = {0x00000000, 0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000, 0x40e00000, 0xc1000000, 0xc0e00000, 0xc0c00000, 0xc0a00000, 0xc0800000, 0xc0400000, 0xc0000000, 0xbf800000};
#       unsigned short lut[32] = {0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800, 0x4880, 0x4900, 0x4980, 0x4a00, 0x4a80, 0x4b00, 0x4b80,
#                                 0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800, 0x4880, 0x4900, 0x4980, 0x4a00, 0x4a80, 0x4b00, 0x4b80};
#       unsigned short signed_lut[32] = {0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0xc800, 0xc700, 0xc600, 0xc500, 0xc400, 0xc200, 0xc000, 0xbc00,
#                                        0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0xc800, 0xc700, 0xc600, 0xc500, 0xc400, 0xc200, 0xc000, 0xbc00};
#       if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype) || io_generated_code->arch < LIBXSMM_X86_AVX512_SPR ) {
#         libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,  (const unsigned char *) (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) ? lut_f32 : signed_lut_f32), "my_lut", 'z', 0 );
#       } else {
#         libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,  (const unsigned char *) (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) ? lut : signed_lut), "my_lut", 'z', 0 );
#       }
#     } else {
#       /* Set to 0 lo mask and to 1 hi mask */
#       unsigned int mask_lo_i4[8] = { 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f};
#       unsigned int mask_hi_i4[8] = { 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0};
#       libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                            (const unsigned char *) mask_lo_i4 ,
#                                                            "my_i4_lo",
#                                                            'y',
#                                                            0 );
#       libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                            (const unsigned char *) mask_hi_i4 ,
#                                                            "my_i4_hi",
#                                                            'y',
#                                                            1 );
#     }
#   }

#   if (l_is_Ai4_Bi8_gemm > 0) {
#     /* Set 2 to vperm reg */
#     unsigned char perm_rpt_zpt[64] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
#                                       8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15};
#     /* Set to 0 lo mask and to 1 hi mask */
#     unsigned int mask_lo_i4[16] = { 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f,
#                                     0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f};
#     unsigned int mask_hi_i4[16] = { 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0,
#                                     0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0};
#     libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                          (const unsigned char *) mask_lo_i4 ,
#                                                          "my_i4_lo",
#                                                          'z',
#                                                          0 );
#     libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                          (const unsigned char *) mask_hi_i4 ,
#                                                          "my_i4_hi",
#                                                          'z',
#                                                          1 );
#     if (io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) {
#       libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,  (const unsigned char *) perm_rpt_zpt, "my_vperm_i4", 'z', 2 );
#     }
#   }

#   if (l_is_Ai2_Bi8_gemm > 0) {
#     if (io_generated_code->arch == LIBXSMM_X86_AVX2_SRF) {
#       char perm_bits_01[32];
#       char perm_bits_23[32];
#       char mask_bits[32];
#       unsigned int __i;

#       for (__i = 0; __i < 32; __i++) {
#         if ((__i & 0x3) == 0) {
#           perm_bits_01[__i] = 0;
#         }
#         if ((__i & 0x3) == 1) {
#           perm_bits_01[__i] = 1;
#         }
#         if ((__i & 0x3) == 2) {
#           perm_bits_01[__i] = -1;
#         }
#         if ((__i & 0xc) == 0) {
#           perm_bits_23[__i] = 0;
#         }
#         if ((__i & 0xc) == 4) {
#           perm_bits_23[__i] = 1;
#         }
#         if ((__i & 0xc) == 8) {
#           perm_bits_23[__i] = -1;
#         }
#         mask_bits[__i] = 15;
#       }

#       libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                            (const unsigned char *) perm_bits_01 ,
#                                                            "my_perm_01",
#                                                            'y',
#                                                            0 );
#       libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                            (const unsigned char *) perm_bits_23 ,
#                                                            "my_perm_23",
#                                                            'y',
#                                                            1 );

#       libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                            (const unsigned char *) mask_bits ,
#                                                            "my_mask_bits",
#                                                            'y',
#                                                            2 );

#     } else {
#       char perm_bits_01[64];
#       char perm_bits_23[64];
#       char perm_bits_45[64];
#       unsigned int __i;

#       for (__i = 0; __i < 64; __i++) {
#         if ((__i & 0x3) == 0) {
#           perm_bits_01[__i] = 0;
#         }
#         if ((__i & 0x3) == 1) {
#           perm_bits_01[__i] = 1;
#         }
#         if ((__i & 0x3) == 2) {
#           perm_bits_01[__i] = -1;
#         }
#         if ((__i & 0xc) == 0) {
#           perm_bits_23[__i] = 0;
#         }
#         if ((__i & 0xc) == 4) {
#           perm_bits_23[__i] = 1;
#         }
#         if ((__i & 0xc) == 8) {
#           perm_bits_23[__i] = -1;
#         }
#         if ((__i & 0x30) == 0) {
#           perm_bits_45[__i] = 0;
#         }
#         if ((__i & 0x30) == 16) {
#           perm_bits_45[__i] = 1;
#         }
#         if ((__i & 0x30) == 32) {
#           perm_bits_45[__i] = -1;
#         }
#       }

#       libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                            (const unsigned char *) perm_bits_01 ,
#                                                            "my_perm_01",
#                                                            'z',
#                                                            0 );
#       libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                            (const unsigned char *) perm_bits_23 ,
#                                                            "my_perm_23",
#                                                            'z',
#                                                            1 );
#       libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                            (const unsigned char *) perm_bits_45 ,
#                                                            "my_perm_45",
#                                                            'z',
#                                                            2 );
#     }
#   }

#   if (l_is_Ai1_Bi8_gemm > 0) {
#     if (io_generated_code->arch == LIBXSMM_X86_AVX2_SRF) {
#       unsigned int l_lut_expand = 0;
#       unsigned int l_mask_bits = 1;
#       unsigned int l_vreg_one = 2;
#       unsigned char l_expand_array[32] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
#                                            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
#                                            0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
#                                            0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03 };
#       unsigned char l_mask_array[32] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
#                                          0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
#                                          0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
#                                          0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 };
#       unsigned char l_one_array[32] = {    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
#                                            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
#                                            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
#                                            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01 };
#       libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                            (const unsigned char *) l_expand_array ,
#                                                            "my_lut_expand",
#                                                            'y',
#                                                            l_lut_expand );
#       libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                            (const unsigned char *) l_mask_array ,
#                                                            "my_mask_bits",
#                                                            'y',
#                                                            l_mask_bits );
#       libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                            (const unsigned char *) l_one_array ,
#                                                            "my_ones",
#                                                            'y',
#                                                            l_vreg_one );
#     } else {
#       char neg_ones[64];
#       unsigned char ones[64];
#       unsigned int __i;

#       for (__i = 0; __i < 64; __i++) {
#         neg_ones[__i] = -1;
#         ones[__i] = 1;
#       }

#       libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                            (const unsigned char *) neg_ones ,
#                                                            "my_neg_ones",
#                                                            'z',
#                                                            0 );
#       libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                            (const unsigned char *) ones ,
#                                                            "my_ones",
#                                                            'z',
#                                                            1 );
#     }
#   }

#   if (l_is_Ai8_Bbf16_gemm > 0) {
#     unsigned int l_is_Ai8_Bbf16_gemm_bf16fma = (l_micro_kernel_config.vmul_instruction == LIBXSMM_X86_INSTR_VDPBF16PS) ? 1 : 0;
#     if (l_is_Ai8_Bbf16_gemm_bf16fma > 0) {
#       unsigned short l_bf16_zip_512[32] = { 0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47 };
#       unsigned short l_bf16_zip_256[16] = { 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23 };
#       libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
#                                                            (const unsigned char *) ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? l_bf16_zip_512 : l_bf16_zip_256 ),
#                                                            "my_bf16_zip",
#                                                            l_micro_kernel_config.vector_name,
#                                                            0 );
#     }
#   }

#   if ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) && (( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) || (LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) )) ) {
#     /* In this case we have one scaling factor per full tensor, load it here */
#     if ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) == 0) {
#       libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, i_gp_reg_mapping->gp_reg_scf );
#       libxsmm_x86_instruction_vec_move( io_generated_code,
#           l_micro_kernel_config.instruction_set,
#           LIBXSMM_X86_INSTR_VPBROADCASTW,
#           i_gp_reg_mapping->gp_reg_scf,
#           LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
#           l_micro_kernel_config.vector_name,
#           (l_is_Ai4_Bf16_gemm > 0) ? 2 : 0, 0, 1, 0 );
#       if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc->datatype) || io_generated_code->arch < LIBXSMM_X86_AVX512_SPR ) {
#         char vname_cvt = (l_micro_kernel_config.vector_name == 'y') ? 'z' : ((l_micro_kernel_config.vector_name == 'x') ? 'y' : 'z');
#         libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, (l_is_Ai4_Bf16_gemm > 0) ? 2 : 0, (l_is_Ai4_Bf16_gemm > 0) ? 2 : 0 );
#       }
#     }
#   }

#   if ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) && (l_is_Amxfp4_Bbf16_gemm == 0)) && ( LIBXSMM_DATATYPE_BF16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) && (( LIBXSMM_DATATYPE_BF16  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) || (LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) )) ) {
#     /* In this case we have one scaling factor per full tensor, load it here */
#     if ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) == 0) {
#       libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, i_gp_reg_mapping->gp_reg_scf );
#       libxsmm_x86_instruction_vec_move( io_generated_code,
#           l_micro_kernel_config.instruction_set,
#           LIBXSMM_X86_INSTR_VPBROADCASTD,
#           i_gp_reg_mapping->gp_reg_scf,
#           LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
#           l_micro_kernel_config.vector_name,
#           1, 0, 1, 0 );
#     }
#   }

#   /* Load the actual batch-reduce trip count */
#   if ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
#     libxsmm_x86_instruction_alu_mem( io_generated_code,
#         l_micro_kernel_config.alu_mov_instruction,
#         i_gp_reg_mapping->gp_reg_reduce_count,
#         LIBXSMM_X86_GP_REG_UNDEF, 0,
#         0,
#         i_gp_reg_mapping->gp_reg_reduce_count,
#         0 );
#   }

#   /* apply n_blocking */
#   while (l_n_done != (unsigned int)l_xgemm_desc->n) {
#     unsigned int l_n_blocking = l_n_n[l_n_count];
#     unsigned int l_m_done = 0;
#     unsigned int l_m_done_old = 0;
#     unsigned int l_m_blocking = 0;

#     /* open N loop */
#     libxsmm_generator_gemm_header_nloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_n_done, l_n_blocking );
#     if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) {
#       libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_decompressed_elts, 0 );
#     }

#     /* advance N */
#     l_n_done += l_n_N[l_n_count];
#     l_n_count++;

#     /* define the micro kernel code gen properties, especially m-blocking affects the vector instruction length */
#     l_m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking( &l_micro_kernel_config, l_xgemm_desc, io_generated_code->arch, 0 );
#     l_micro_kernel_config.m_bitmask_advance = 0; /* @TODO: FOR SSE ONLY and relumask */

#     /* apply m_blocking */
#     while (l_m_done != (unsigned int)l_xgemm_desc->m) {
#       if ( l_m_blocking == 0 ) {
#         LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
#         return;
#       }

#       l_m_done_old = l_m_done;
#       l_micro_kernel_config.current_m = l_m_done_old;
#       LIBXSMM_ASSERT(0 != l_m_blocking);
#       /* coverity[divide_by_zero] */
#       l_m_done = l_m_done + (((l_xgemm_desc->m - l_m_done_old) / l_m_blocking) * l_m_blocking);
#       l_micro_kernel_config.m_bitmask_advance += ((l_m_done-l_m_done_old+3)/8); /* @TODO: FOR SSE ONLY and relumask */
#       /*printf(" advance: %i %i %i\n", l_m_done, l_m_blocking, l_micro_kernel_config.m_bitmask_advance);*/

#       if ( (l_m_done != l_m_done_old) && (l_m_done > 0) ) {
#         /* when on AVX512, load mask, if needed */
#         if ( ( l_micro_kernel_config.use_masking_a_c != 0 ) && ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
#           /* compute the mask count, depends on vlen as block in M */
#           unsigned int l_corrected_vlen = l_micro_kernel_config.vector_length;
#           unsigned int l_mask_count = l_corrected_vlen - ( l_m_blocking % l_corrected_vlen );

#           if ( ( ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
#                ( ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
#                ( ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
#                ( ( LIBXSMM_DATATYPE_I8   == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_I8   == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
#                ( ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
#                ( ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
#                ( ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
#                ( ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
#                ( ( LIBXSMM_DATATYPE_BF16  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
#                ( ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
#                ( ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
#                ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
#                ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
#                ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_BF16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_BF16  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
#                ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_BF16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
#                ( ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ) {
#             unsigned int l_is_Ai8_Bf16_gemm = ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) )  ? 1 : 0;
#             unsigned int l_is_Abf8_Bf16_gemm = ( ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) ) ? 1 : 0;
#             unsigned int l_is_Af16_Bf16_gemm = ( ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) )) ? 1 : 0;
#             unsigned int l_is_compute_f16_gemm = ((l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0 || l_is_Af16_Bf16_gemm > 0) && ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc->datatype) && ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR ) )) ? 1 : 0;
#             libxsmm_generator_initialize_avx512_mask( io_generated_code, i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_AVX512_MASK, l_mask_count, (libxsmm_datatype) ((l_is_compute_f16_gemm > 0) ? LIBXSMM_DATATYPE_F16 : LIBXSMM_DATATYPE_I32) );
#             if (l_is_compute_f16_gemm > 0 && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype )) {
#               /* Adjust mask for C handling */
#               unsigned int l_c_vlen_adjusted = (l_is_compute_f16_gemm > 0) ? l_corrected_vlen/2 : l_corrected_vlen;
#               libxsmm_generator_initialize_avx512_mask( io_generated_code, i_gp_reg_mapping->gp_reg_help_1, 2, (l_m_blocking % l_corrected_vlen >= l_c_vlen_adjusted) ? 0 : l_c_vlen_adjusted - ( l_m_blocking % l_c_vlen_adjusted ), (libxsmm_datatype)LIBXSMM_DATATYPE_I32 );
#               libxsmm_generator_initialize_avx512_mask( io_generated_code, i_gp_reg_mapping->gp_reg_help_1, 3, (l_m_blocking % l_corrected_vlen >= l_c_vlen_adjusted) ? l_c_vlen_adjusted - ( l_m_blocking % l_c_vlen_adjusted ) : l_c_vlen_adjusted, (libxsmm_datatype)LIBXSMM_DATATYPE_I32 );
#             } else {
#               /* we have to adjust mask count as for now we are using ymm for 16bit and xmm for 8bit */
#               if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512_SKX ) ) {
#                 l_mask_count = ( (((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_HF8 != LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype )  ) && (LIBXSMM_DATATYPE_BF8 != LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ))) || ( l_is_Ai8_Bbf16_gemm > 0)) ) ? l_mask_count + 8 : l_mask_count + 24;
#               } else {
#                 l_mask_count = ( (((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_HF8 != LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype )  ) && (LIBXSMM_DATATYPE_BF8 != LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ))) || ( l_is_Ai8_Bbf16_gemm > 0)) ) ? l_mask_count + 16 : l_mask_count + 48;
#               }
#               libxsmm_generator_initialize_avx512_mask( io_generated_code, i_gp_reg_mapping->gp_reg_help_1, 2, l_mask_count, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype) );
#             }
#           } else {
#             libxsmm_generator_initialize_avx512_mask( io_generated_code, i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_AVX512_MASK, l_mask_count, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype) );
#           }
#         } else if ( ( l_micro_kernel_config.use_masking_a_c != 0 ) && ( io_generated_code->arch >= LIBXSMM_X86_AVX ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX )  ) {
#           unsigned int l_corrected_vlen = l_micro_kernel_config.vector_length;
#           unsigned int l_mask_count = l_m_blocking % l_corrected_vlen;

#           if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype )) ||
#                (LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype )) ||
#                (LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ))    ) {
#             libxsmm_generator_initialize_avx_mask( io_generated_code, (l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0 ) ? 2 : 0, l_mask_count, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) );
#           } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) {
#             libxsmm_generator_initialize_avx_mask( io_generated_code, (l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0 ) ? 2 : 0, l_mask_count, LIBXSMM_DATATYPE_I32 );
#           } else {
#             /* should not happen */
#           }
#           /* store mask into stack frame */
#           libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AVX2_MASK_PTR, i_gp_reg_mapping->gp_reg_help_1 );
#           libxsmm_x86_instruction_vec_move( io_generated_code, l_micro_kernel_config.instruction_set, LIBXSMM_X86_INSTR_VMOVUPS,
#                                             i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'y', (l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0 ) ? 2 : 0, 0, 0, 1 );
#         }

#         libxsmm_generator_gemm_header_mloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_m_done_old, l_m_blocking );
#         libxsmm_generator_gemm_load_C( io_generated_code, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc, l_m_blocking, l_n_blocking );

#         if ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
#           if ( l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) {
#             libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b);
#             libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_a);
#           } else if ( l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE ) {
#             libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, 32 );
#             libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 24, i_gp_reg_mapping->gp_reg_a, 1);
#             libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 16, i_gp_reg_mapping->gp_reg_b, 1);
#             libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, 0);
#             libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 8, i_gp_reg_mapping->gp_reg_help_0, 1);
#             libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, i_gp_reg_mapping->gp_reg_help_0, 1);
#           } else {
#             /* nothing to do */
#           }
#           /* This is the reduce loop */
#           libxsmm_generator_gemm_header_reduceloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config );
#           if (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
#             libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_a);
#             libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b);

#             if (adjust_A_pf_ptrs) {
#               /* coverity[dead_error_line] */
#               libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_a_prefetch );
#             }
#             if (adjust_B_pf_ptrs) {
#               libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b_prefetch );
#             }
#             /* load to reg_a the proper array based on the reduce loop index */
#             libxsmm_x86_instruction_alu_mem( io_generated_code,
#                 l_micro_kernel_config.alu_mov_instruction,
#                 i_gp_reg_mapping->gp_reg_a,
#                 i_gp_reg_mapping->gp_reg_reduce_loop, 8,
#                 0,
#                 i_gp_reg_mapping->gp_reg_a,
#                 0 );
#             if (l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0 ) {
#               libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_1 );
#               libxsmm_x86_instruction_alu_mem( io_generated_code,
#                   l_micro_kernel_config.alu_mov_instruction,
#                   i_gp_reg_mapping->gp_reg_help_1,
#                   i_gp_reg_mapping->gp_reg_reduce_loop, 8,
#                   0,
#                   i_gp_reg_mapping->gp_reg_scf,
#                   0 );
#               if (l_is_Amxfp4_Bi8_gemm > 0) {
#                 libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BSCALE_PTR, i_gp_reg_mapping->gp_reg_help_1 );
#                 libxsmm_x86_instruction_alu_mem( io_generated_code,
#                     l_micro_kernel_config.alu_mov_instruction,
#                     i_gp_reg_mapping->gp_reg_help_1,
#                     i_gp_reg_mapping->gp_reg_reduce_loop, 8,
#                     0,
#                     i_gp_reg_mapping->gp_reg_help_1,
#                     0 );
#                 libxsmm_generator_gemm_setval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BSCALE_BRGEMM_PTR, i_gp_reg_mapping->gp_reg_help_1 );
#               }
#             }
#             if (l_is_Ai4_Bi8_gemm > 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0 ) {
#               libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_1 );
#               libxsmm_x86_instruction_alu_mem( io_generated_code,
#                   l_micro_kernel_config.alu_mov_instruction,
#                   i_gp_reg_mapping->gp_reg_help_1,
#                   i_gp_reg_mapping->gp_reg_reduce_loop, 8,
#                   0,
#                   i_gp_reg_mapping->gp_reg_help_1,
#                   0 );
#               libxsmm_generator_gemm_setval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_BRGEMM_PTR, i_gp_reg_mapping->gp_reg_help_1 );
#             }
#             /* load to reg_b the proper array based on the reduce loop index */
#             libxsmm_x86_instruction_alu_mem( io_generated_code,
#                 l_micro_kernel_config.alu_mov_instruction,
#                 i_gp_reg_mapping->gp_reg_b,
#                 i_gp_reg_mapping->gp_reg_reduce_loop, 8,
#                 0,
#                 i_gp_reg_mapping->gp_reg_b,
#                 0 );
#             if (adjust_A_pf_ptrs) {
#               libxsmm_x86_instruction_alu_mem( io_generated_code,
#                   l_micro_kernel_config.alu_mov_instruction,
#                   i_gp_reg_mapping->gp_reg_a_prefetch,
#                   i_gp_reg_mapping->gp_reg_reduce_loop, 8,
#                   0,
#                   i_gp_reg_mapping->gp_reg_a_prefetch,
#                   0 );
#             }
#             if (adjust_B_pf_ptrs) {
#               /* coverity[dead_error_line] */
#               libxsmm_x86_instruction_alu_mem( io_generated_code,
#                   l_micro_kernel_config.alu_mov_instruction,
#                   i_gp_reg_mapping->gp_reg_b_prefetch,
#                   i_gp_reg_mapping->gp_reg_reduce_loop, 8,
#                   0,
#                   i_gp_reg_mapping->gp_reg_b_prefetch,
#                   0 );
#             }
#           } else if (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
#             libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_a);
#             libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b);
#             libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b);
#             libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_a);
#             /* Calculate to reg_a the proper address based on the reduce loop index */
#             libxsmm_x86_instruction_alu_mem( io_generated_code,
#                 l_micro_kernel_config.alu_mov_instruction,
#                 i_gp_reg_mapping->gp_reg_a_offset,
#                 i_gp_reg_mapping->gp_reg_reduce_loop, 8,
#                 0,
#                 i_gp_reg_mapping->gp_reg_help_0,
#                 0 );
#             libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a);

#             if (l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0) {
#               libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_1 );
#               libxsmm_x86_instruction_alu_mem( io_generated_code,
#                   l_micro_kernel_config.alu_mov_instruction,
#                   i_gp_reg_mapping->gp_reg_a_offset,
#                   i_gp_reg_mapping->gp_reg_reduce_loop, 8,
#                   0,
#                   i_gp_reg_mapping->gp_reg_scf,
#                   0 );
#               libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SARQ, i_gp_reg_mapping->gp_reg_scf, 4);
#               libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_scf);
#               if ( l_is_Amxfp4_Bi8_gemm > 0 ) {
#                 libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
#                 libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BSCALE_PTR, i_gp_reg_mapping->gp_reg_help_1 );
#                 libxsmm_x86_instruction_alu_mem( io_generated_code,
#                     l_micro_kernel_config.alu_mov_instruction,
#                     i_gp_reg_mapping->gp_reg_b_offset,
#                     i_gp_reg_mapping->gp_reg_reduce_loop, 8,
#                     0,
#                     LIBXSMM_X86_GP_REG_RDX,
#                     0 );
#                 libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SARQ, LIBXSMM_X86_GP_REG_RDX, 3);
#                 libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_RDX);
#                 libxsmm_generator_gemm_setval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BSCALE_BRGEMM_PTR, LIBXSMM_X86_GP_REG_RDX );
#                 libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
#               }
#             }

#             if (l_is_Ai4_Bi8_gemm > 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0 ) {
#               libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
#               libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
#               libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction, LIBXSMM_X86_GP_REG_RDX, 0);
#               libxsmm_x86_instruction_alu_mem( io_generated_code,
#                   l_micro_kernel_config.alu_mov_instruction,
#                   i_gp_reg_mapping->gp_reg_a_offset,
#                   i_gp_reg_mapping->gp_reg_reduce_loop, 8,
#                   0,
#                   LIBXSMM_X86_GP_REG_RAX,
#                   0 );
#               libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_1, i_xgemm_desc->k);
#               libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SARQ, i_gp_reg_mapping->gp_reg_help_1, 1);
#               libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_IDIVQ, LIBXSMM_X86_GP_REG_UNDEF, i_gp_reg_mapping->gp_reg_help_1);
#               libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_1 );
#               libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, LIBXSMM_X86_GP_REG_RAX, i_gp_reg_mapping->gp_reg_help_1);
#               libxsmm_generator_gemm_setval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_BRGEMM_PTR, i_gp_reg_mapping->gp_reg_help_1 );
#               libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
#               libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
#             }


#             /* Calculate to reg_b the proper address based on the reduce loop index */
#             libxsmm_x86_instruction_alu_mem( io_generated_code,
#                 l_micro_kernel_config.alu_mov_instruction,
#                 i_gp_reg_mapping->gp_reg_b_offset,
#                 i_gp_reg_mapping->gp_reg_reduce_loop, 8,
#                 0,
#                 i_gp_reg_mapping->gp_reg_help_0,
#                 0 );
#             libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_b);
#           } else if (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) {
#             /* reloading A and B from stack */
#             libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 24, i_gp_reg_mapping->gp_reg_a, 0);
#             libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 16, i_gp_reg_mapping->gp_reg_b, 0);
#             /* Calculate to reg_a the proper address based on the reduce loop index */
#             libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 8, i_gp_reg_mapping->gp_reg_help_0, 0);
#             libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a);
#             /* Calculate to reg_b the proper address based on the reduce loop index */
#             libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, i_gp_reg_mapping->gp_reg_help_0, 0);
#             libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_b);
#             if (l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0) {
#               libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_1 );
#               libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, i_gp_reg_mapping->gp_reg_scf);
#               libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_scf, i_xgemm_desc->c1/16);
#               libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_scf);
#               if ( l_is_Amxfp4_Bi8_gemm > 0 ) {
#                 libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
#                 libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BSCALE_PTR, i_gp_reg_mapping->gp_reg_help_1 );
#                 libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, LIBXSMM_X86_GP_REG_RAX);
#                 libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, LIBXSMM_X86_GP_REG_RAX, i_xgemm_desc->c2/8);
#                 libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_RAX);
#                 libxsmm_generator_gemm_setval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BSCALE_BRGEMM_PTR, LIBXSMM_X86_GP_REG_RAX );
#                 libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
#               }
#             }

#             if (l_is_Ai4_Bi8_gemm > 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0 ) {
#               libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
#               libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_1 );
#               libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, LIBXSMM_X86_GP_REG_RAX);
#               libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, LIBXSMM_X86_GP_REG_RAX, i_xgemm_desc->lda);
#               libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_RAX);
#               libxsmm_generator_gemm_setval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_BRGEMM_PTR, LIBXSMM_X86_GP_REG_RAX );
#               libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
#             }
#           }
#         }

#         libxsmm_generator_gemm_sse_avx_avx2_avx512_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config,
#                                                            l_xgemm_desc, l_m_blocking, l_n_blocking );

#         if ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
#           if (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
#             if (adjust_B_pf_ptrs) {
#               /* coverity[dead_error_begin] */
#               libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
#               libxsmm_x86_instruction_alu_mem( io_generated_code,
#                   l_micro_kernel_config.alu_mov_instruction,
#                   i_gp_reg_mapping->gp_reg_help_0,
#                   i_gp_reg_mapping->gp_reg_reduce_loop, 8,
#                   0,
#                   i_gp_reg_mapping->gp_reg_b_prefetch,
#                   1 );
#               libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_b_prefetch);
#             }
#             if (adjust_A_pf_ptrs) {
#               libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
#               libxsmm_x86_instruction_alu_mem( io_generated_code,
#                   l_micro_kernel_config.alu_mov_instruction,
#                   i_gp_reg_mapping->gp_reg_help_0,
#                   i_gp_reg_mapping->gp_reg_reduce_loop, 8,
#                   0,
#                   i_gp_reg_mapping->gp_reg_a_prefetch,
#                   1 );
#               libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a_prefetch);
#             }
#             /* Pop address of B_array to help_0 and store proper address of B */
#             libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
#             libxsmm_x86_instruction_alu_mem( io_generated_code,
#                 l_micro_kernel_config.alu_mov_instruction,
#                 i_gp_reg_mapping->gp_reg_help_0,
#                 i_gp_reg_mapping->gp_reg_reduce_loop, 8,
#                 0,
#                 i_gp_reg_mapping->gp_reg_b,
#                 1 );
#             /* Move to reg_b the address of B_array */
#             libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_b);
#             /* Pop address of A_array to help_0 and store proper address of A */
#             libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
#             libxsmm_x86_instruction_alu_mem( io_generated_code,
#                 l_micro_kernel_config.alu_mov_instruction,
#                 i_gp_reg_mapping->gp_reg_help_0,
#                 i_gp_reg_mapping->gp_reg_reduce_loop, 8,
#                 0,
#                 i_gp_reg_mapping->gp_reg_a,
#                 1 );
#             /* Move to reg_a the address of A_array */
#             libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a);
#           } else if (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) {
#            /* Calculate to reg_a the proper address based on the reduce loop index */
#             libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 8, i_gp_reg_mapping->gp_reg_help_0, 0);
#             libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_mapping->gp_reg_help_0, l_xgemm_desc->c1);
#             libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 8, i_gp_reg_mapping->gp_reg_help_0, 1);
#             /* Calculate to reg_b the proper address based on the reduce loop index */
#             libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, i_gp_reg_mapping->gp_reg_help_0, 0);
#             libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_mapping->gp_reg_help_0, l_xgemm_desc->c2);
#             libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, i_gp_reg_mapping->gp_reg_help_0, 1);
#           }
#           libxsmm_generator_gemm_footer_reduceloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc);
#           if (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
#             /* Calculate to reg_a the proper A advance from the microkernel */
#             libxsmm_x86_instruction_alu_mem( io_generated_code,
#                 l_micro_kernel_config.alu_mov_instruction,
#                 i_gp_reg_mapping->gp_reg_a_offset,
#                 i_gp_reg_mapping->gp_reg_reduce_loop, 8,
#                 -8,
#                 i_gp_reg_mapping->gp_reg_help_0,
#                 0 );
#             libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a);
#             /* Calculate to reg_b the proper B advance from the microkernel */
#             libxsmm_x86_instruction_alu_mem( io_generated_code,
#                 l_micro_kernel_config.alu_mov_instruction,
#                 i_gp_reg_mapping->gp_reg_b_offset,
#                 i_gp_reg_mapping->gp_reg_reduce_loop, 8,
#                 -8,
#                 i_gp_reg_mapping->gp_reg_help_0,
#                 0 );
#             libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_b);
#             /* Consume the last two pushes from the stack */
#             libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
#             libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
#           }
#           if (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) {
#             /* Calculate to reg_a the proper A advance from the microkernel */
#             libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_count, i_gp_reg_mapping->gp_reg_help_0);
#             libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_help_0, l_xgemm_desc->c1);
#             libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, l_xgemm_desc->c1);
#             libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a);
#             /* Calculate to reg_b the proper B advance from the microkernel */
#             libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_count, i_gp_reg_mapping->gp_reg_help_0);
#             libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_help_0, l_xgemm_desc->c2);
#             libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, l_xgemm_desc->c2);
#             libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_b);
#             /* reset stack */
#             libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, LIBXSMM_X86_GP_REG_RSP, 32 );
#           }
#         }

#         libxsmm_generator_gemm_store_C( io_generated_code, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc, l_m_blocking, l_n_blocking );
#         libxsmm_generator_gemm_footer_mloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc, l_m_blocking, l_m_done );
#       }

#       /* switch to next smaller m_blocking */
#       l_m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking( &l_micro_kernel_config, l_xgemm_desc, io_generated_code->arch, l_m_blocking );
#     }
#     libxsmm_generator_gemm_footer_nloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc, l_n_blocking, l_n_done );
#   } /* while l_n_done */

#   /* In this case we vnni-format C from scratch */
#   if (l_micro_kernel_config.vnni_format_C > 0) {
#     l_xgemm_desc->ldc = i_xgemm_desc->ldc;
#     libxsmm_generator_gemm_vnni_store_C_from_scratch( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc);
#   }

#   /* destroy stack frame */
#   libxsmm_generator_gemm_destroy_stack_frame( io_generated_code, l_xgemm_desc, i_gp_reg_mapping, &l_micro_kernel_config );
# }


def libxsmm_generator_gemm_sse_avx_avx2_avx512_kloop(
    generated_code: GeneratedCode,
    label_tracker: LoopLabelTracker,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    m_blocking: int,
    n_blocking: int,
) -> None:
    # some hard coded parameters for k-blocking
    k_blocking = 0
    k_threshold = 0
    _k_pack_factor = 1
    is_Amxfp4_Bbf16_gemm = desc.is_Amxfp4_Bbf16_gemm
    is_Ai8_Bbf16_gemm = (
        desc.datatype.a == "I8"
        and not is_Amxfp4_Bbf16_gemm
        and desc.datatype.b == "BF16"
    )
    is_Ai4_Bf16_gemm = (
        (desc.flags & GEMMFlag.INTERPRETE_A_AS_INT4_VNNI2)
        and desc.datatype.a == "I8"
        and desc.datatype.b == "F16"
        and (desc.datatype.c == "F16" or desc.datatype.c == "F32")
    )
    is_Ai8_Bbf16_gemm_bf16fma = (
        False  # micro_kernel_config.vmul_instruction == "VDPBF16PS"
    )
    is_Amxfp4_Bfp32_gemm = desc.is_Amxfp4_Bfp32_gemm()
    is_Amxfp4_Bi8_gemm = desc.is_Amxfp4_Bi8_gemm()
    is_Ai4_Bi8_gemm = desc.is_Ai4_Bi8_gemm()
    is_i8_uu_ss_gemm = desc.datatype.ab == "I8" and (
        GEMMFlag.A_UNSIGNED in desc.flags == GEMMFlag.B_UNSIGNED in desc.flags
    )

    # a very simple k unrolling model */
    k_blocking = 4
    k_threshold = 23

    if GEMMFlag.VNNI_A in desc.flags:
        # VNNI kernel should maintain the same amount of unrolled instructions
        raise NotImplementedError

    if is_i8_uu_ss_gemm and generated_code.arch in (
        Arch.LIBXSMM_X86_AVX512_SKX,
        Arch.LIBXSMM_X86_AVX512_VL256_SKX,
    ):
        # for uu ss int 8 we need to limit the unrolling, software emulation code is very large
        k_blocking = 8
        k_threshold = 23

    if is_Ai8_Bbf16_gemm:
        raise NotImplementedError

    assert k_blocking <= k_threshold

    if is_Ai4_Bf16_gemm:
        k_blocking = 4
        k_threshold = 8
    if is_Amxfp4_Bfp32_gemm or is_Amxfp4_Bbf16_gemm or is_Amxfp4_Bi8_gemm:
        k_blocking = 32
        k_threshold = desc.k

    # Set up architecture dependent compute micro kernel generator.
    assert generated_code.arch >= Arch.LIBXSMM_TARGET_ARCH_GENERIC

    if is_Amxfp4_Bfp32_gemm or is_Amxfp4_Bbf16_gemm or is_Amxfp4_Bi8_gemm:
        # generator_kloop_kernel = libxsmm_generator_gemm_avx2_kloop_kernel
        raise NotImplementedError
    elif generated_code.arch <= Arch.LIBXSMM_X86_SSE42:
        # generator_kloop_kernel = libxsmm_generator_gemm_sse_kloop_kernel
        raise NotImplementedError
    elif generated_code.arch == Arch.LIBXSMM_X86_AVX:
        # generator_kloop_kernel = libxsmm_generator_gemm_avx_kloop_kernel
        raise NotImplementedError
    elif (
        generated_code.arch >= Arch.LIBXSMM_X86_AVX2
        and generated_code.arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX
    ):
        # generator_kloop_kernel = libxsmm_generator_gemm_avx2_kloop_kernel
        raise NotImplementedError
    elif (
        generated_code.arch >= Arch.LIBXSMM_X86_AVX512_VL256_SKX
        and generated_code.arch <= Arch.LIBXSMM_X86_ALLFEAT
    ):
        generator_kloop_kernel = libxsmm_generator_gemm_avx512_kloop_kernel
    else:
        assert False, (
            f"Unsupported architecture {generated_code.arch} for micro-kernel generation"
        )

    if is_Ai4_Bi8_gemm and GEMMFlag.USE_MxK_ZPT in desc.flags:
        raise NotImplementedError

    if is_Ai8_Bbf16_gemm and is_Ai8_Bbf16_gemm_bf16fma:
        raise NotImplementedError

    # Apply multiple k_blocking strategies

    if not desc.k % k_blocking and k_threshold < desc.k:
        # 1. we are larger the k_threshold and a multiple of a predefined blocking parameter
        libxsmm_generator_gemm_header_kloop(
            generated_code,
            label_tracker,
            gp_reg_mapping,
            micro_kernel_config,
            m_blocking,
            k_blocking,
        )
        generator_kloop_kernel(
            generated_code,
            gp_reg_mapping,
            micro_kernel_config,
            desc,
            m_blocking,
            n_blocking,
            k_blocking,
        )
        libxsmm_generator_gemm_footer_kloop(
            generated_code,
            label_tracker,
            gp_reg_mapping,
            micro_kernel_config,
            desc,
            m_blocking,
            desc.k,
            True,
        )
    else:
        b_offset = 0
        # 2. we want to fully unroll below the threshold
        if desc.k <= k_threshold:
            generator_kloop_kernel(
                generated_code,
                gp_reg_mapping,
                micro_kernel_config,
                desc,
                m_blocking,
                n_blocking,
                desc.k,
            )
        # 3. we are larger than the threshold but not a multiple of the blocking factor -> largest possible blocking + remainder handling
        else:
            # Largest possible blocking
            l_max_blocked_k = (desc.k // k_blocking) * k_blocking

            # We can block as k is large enough
            if l_max_blocked_k > 0:
                libxsmm_generator_gemm_header_kloop(
                    generated_code,
                    label_tracker,
                    gp_reg_mapping,
                    micro_kernel_config,
                    m_blocking,
                    k_blocking,
                )

                generator_kloop_kernel(
                    generated_code,
                    gp_reg_mapping,
                    micro_kernel_config,
                    desc,
                    m_blocking,
                    n_blocking,
                    k_blocking,
                )

                libxsmm_generator_gemm_footer_kloop(
                    generated_code,
                    label_tracker,
                    gp_reg_mapping,
                    micro_kernel_config,
                    desc,
                    m_blocking,
                    l_max_blocked_k,
                    False,
                )

            # Now handle the remainder
            generator_kloop_kernel(
                generated_code,
                gp_reg_mapping,
                micro_kernel_config,
                desc,
                m_blocking,
                n_blocking,
                desc.k - l_max_blocked_k,
            )

            # Reset B pointer
            if GEMMFlag.TRANS_B in desc.flags:
                b_offset = desc.ldb * desc.k * micro_kernel_config.datatype_size_in2
            else:
                b_offset = desc.k * micro_kernel_config.datatype_size_in2

            generated_code.insert(
                x86.ops.RI_SubOp(
                    generated_code.current_val_by_reg[gp_reg_mapping.gp_reg_b],
                    b_offset,
                    register_out=gp_reg_mapping.gp_reg_b,
                )
            )

    if is_Ai8_Bbf16_gemm and not is_Ai8_Bbf16_gemm_bf16fma:
        raise NotImplementedError


def libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking(
    config: MicroKernelConfig, desc: GEMMDescriptor, arch: Arch, current_m_blocking: int
):
    use_masking_a_c = 0
    m_blocking = current_m_blocking
    is_Amxfp4_Bfp32_gemm = desc.is_Amxfp4_Bfp32_gemm()
    is_Amxfp4_Bi8_gemm = desc.is_Amxfp4_Bi8_gemm()
    is_Amxfp4_Bbf16_gemm = desc.is_Amxfp4_Bbf16_gemm()

    if (arch <= Arch.LIBXSMM_X86_SSE42) and (Datatype.F32 == desc.datatype.ab):
        # if ( io_micro_kernel_config->fused_relu == 1 ) {
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 4, &l_m_blocking, &l_use_masking_a_c );
        # } else {
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 12, 4, &l_m_blocking, &l_use_masking_a_c );
        # }
        raise NotImplementedError
    elif (arch <= Arch.LIBXSMM_X86_SSE42) and (Datatype.F64 == desc.datatype.ab):
        # if ( io_micro_kernel_config->fused_relu == 1 ) {
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 4, 2, &l_m_blocking, &l_use_masking_a_c );
        # } else {
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 6, 2, &l_m_blocking, &l_use_masking_a_c );
        # }
        raise NotImplementedError
    elif (arch <= Arch.LIBXSMM_X86_SSE42) and (Datatype.I8 == desc.datatype.ab):
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 4, &l_m_blocking, &l_use_masking_a_c );
        raise NotImplementedError
    elif (arch <= Arch.LIBXSMM_X86_SSE42) and (Datatype.I16 == desc.datatype.ab):
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 4, &l_m_blocking, &l_use_masking_a_c );
        raise NotImplementedError
    elif (arch <= Arch.LIBXSMM_X86_SSE42) and (Datatype.BF16 == desc.datatype.ab):
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 4, &l_m_blocking, &l_use_masking_a_c );
        raise NotImplementedError
    elif (arch == Arch.LIBXSMM_X86_AVX) and (Datatype.F32 == desc.datatype.ab):
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 24, 8, &l_m_blocking, &l_use_masking_a_c );
        raise NotImplementedError
    elif (arch == Arch.LIBXSMM_X86_AVX) and (Datatype.F64 == desc.datatype.ab):
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 12, 4, &l_m_blocking, &l_use_masking_a_c );
        raise NotImplementedError
    elif (arch >= Arch.LIBXSMM_X86_AVX2) and (
        is_Amxfp4_Bfp32_gemm > 0 or is_Amxfp4_Bbf16_gemm > 0 or is_Amxfp4_Bi8_gemm > 0
    ):
        # if (i_xgemm_desc->n == 1) {
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 8, &l_m_blocking, &l_use_masking_a_c );
        # } else {
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 8, &l_m_blocking, &l_use_masking_a_c );
        # }
        raise NotImplementedError
    elif (arch == Arch.LIBXSMM_X86_AVX2_SRF) and (
        desc.datatype.c in (Datatype.F32, Datatype.BF16, Datatype.I32)
    ):
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, (libxsmm_cpuid_x86_srf_gemm_set_n_max_blocking() <= 3) ? 32 : ( (libxsmm_cpuid_x86_srf_gemm_set_n_max_blocking() <= 5) ? 16 : 8), 8, &l_m_blocking, &l_use_masking_a_c );
        raise NotImplementedError
    elif (
        (arch >= Arch.LIBXSMM_X86_AVX2) and (arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX)
    ) and (desc.datatype.c in (Datatype.F32, Datatype.BF16, Datatype.I32)):
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 8, &l_m_blocking, &l_use_masking_a_c );
        raise NotImplementedError
    elif (arch == Arch.LIBXSMM_X86_AVX2_SRF) and (Datatype.F64 == desc.datatype.ab):
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, (libxsmm_cpuid_x86_srf_gemm_set_n_max_blocking() <= 3) ? 16 : ( (libxsmm_cpuid_x86_srf_gemm_set_n_max_blocking() <= 5) ? 8 : 4), 4, &l_m_blocking, &l_use_masking_a_c );
        raise NotImplementedError
    elif (
        (arch >= Arch.LIBXSMM_X86_AVX2) and (arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX)
    ) and (Datatype.F64 == desc.datatype.ab):
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 16, 4, &l_m_blocking, &l_use_masking_a_c );
        raise NotImplementedError
    elif (
        (
            (arch >= Arch.LIBXSMM_X86_AVX512_VL256_SKX)
            and (arch < Arch.LIBXSMM_X86_AVX512_SKX)
        )
        and (Datatype.BF16 == desc.datatype.ab)
        and ((desc.flags & GEMMFlag.VNNI_A) == 0)
    ):
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 8, &l_m_blocking, &l_use_masking_a_c );
        raise NotImplementedError
    elif (
        ((arch >= Arch.LIBXSMM_X86_AVX512_SKX) and (arch <= Arch.LIBXSMM_X86_ALLFEAT))
        and (Datatype.BF16 == desc.datatype.ab)
        and ((desc.flags & GEMMFlag.VNNI_A) == 0)
        and ((desc.flags & GEMMFlag.DECOMPRESS_A_VIA_BITMASK) == 0)
    ):
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 16, 16, &l_m_blocking, &l_use_masking_a_c );
        raise NotImplementedError
    elif (arch == Arch.LIBXSMM_X86_AVX512_VL256_SKX) and (
        desc.datatype.ab in (Datatype.I8, Datatype.I16)
    ):
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 8, &l_m_blocking, &l_use_masking_a_c );
        raise NotImplementedError
    elif (
        (arch >= Arch.LIBXSMM_X86_AVX512_SKX)
        and (arch <= Arch.LIBXSMM_X86_AVX512_SKX)
        and (desc.datatype.ab in (Datatype.I8, Datatype.I16))
    ):
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 16, 16, &l_m_blocking, &l_use_masking_a_c );
        raise NotImplementedError
    elif (
        (desc.datatype.c in (Datatype.F16, Datatype.F32))
        and (desc.datatype.ab == Datatype.F16)
    ) or (
        (desc.datatype.c in (Datatype.F16, Datatype.F32))
        and (desc.datatype.b == Datatype.F16)
        and desc.datatype.a in (Datatype.I8, Datatype.BF8)
    ):
        # if (LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype )) {
        # if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( i_arch >= LIBXSMM_X86_AVX512_SPR ) ) {
        #     libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 128, 32, &l_m_blocking, &l_use_masking_a_c );
        # } else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( i_arch >= LIBXSMM_X86_AVX512_SKX ) ) {
        #     libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 64, 16, &l_m_blocking, &l_use_masking_a_c );
        # } else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( i_arch >= LIBXSMM_X86_AVX512_VL256_SKX ) ) {
        #     libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 8, &l_m_blocking, &l_use_masking_a_c );
        # } else {
        #     /* Do nothing  */
        # }
        # } else {
        # if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( i_arch >= LIBXSMM_X86_AVX512_SPR ) ) {
        #     libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 64, 16, &l_m_blocking, &l_use_masking_a_c );
        # } else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( i_arch >= LIBXSMM_X86_AVX512_SKX ) ) {
        #     libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 64, 16, &l_m_blocking, &l_use_masking_a_c );
        # } else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( i_arch >= LIBXSMM_X86_AVX512_VL256_SKX ) ) {
        #     libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 8, &l_m_blocking, &l_use_masking_a_c );
        # } else {
        #     /* Do nothing  */
        # }
        # }
        raise NotImplementedError
    elif (
        (arch >= Arch.LIBXSMM_X86_AVX512_VL256_SKX)
        and (arch < Arch.LIBXSMM_X86_AVX512_SKX)
    ) and (
        desc.datatype.c in (Datatype.F32, Datatype.I32)
        or (desc.datatype.c == Datatype.BF16 and (desc.flags & GEMMFlag.VNNI_A) > 0)
        or (desc.datatype.c == Datatype.BF8 and (desc.flags & GEMMFlag.VNNI_A) > 0)
        or (desc.datatype.c == Datatype.BF8 and (desc.flags & GEMMFlag.VNNI_A) == 0)
        or (desc.datatype.c == Datatype.HF8 and (desc.flags & GEMMFlag.VNNI_A) > 0)
        or (desc.datatype.c == Datatype.HF8 and (desc.flags & GEMMFlag.VNNI_A) == 0)
    ):
        # /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32/Int32 out kernel with the same logic */
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 64, 8, &l_m_blocking, &l_use_masking_a_c );
        raise NotImplementedError
    elif (
        (arch >= Arch.LIBXSMM_X86_AVX512_VL256_SKX)
        and (arch < Arch.LIBXSMM_X86_AVX512_SKX)
        and (Datatype.F64 == desc.datatype.ab)
    ):
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 4, &l_m_blocking, &l_use_masking_a_c );
        raise NotImplementedError
    elif (
        (desc.datatype.c in (Datatype.BF16, Datatype.F32))
        and (desc.datatype.b == Datatype.BF16)
        and (desc.datatype.a == Datatype.I8)
    ):
        # if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( i_arch >= LIBXSMM_X86_AVX512_SKX ) ) {
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 64, 16, &l_m_blocking, &l_use_masking_a_c );
        # } else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( i_arch >= LIBXSMM_X86_AVX512_VL256_SKX ) ) {
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 8, &l_m_blocking, &l_use_masking_a_c );
        # } else {
        # /* Do nothing  */
        # }
        raise NotImplementedError
    elif (arch <= Arch.LIBXSMM_X86_ALLFEAT) and (
        desc.datatype.c == Datatype.F32
        or desc.datatype.c == Datatype.I32
        or (desc.datatype.c == Datatype.BF16 and (desc.flags & GEMMFlag.VNNI_A) > 0)
        or (
            desc.datatype.c == Datatype.BF16
            and (desc.flags & GEMMFlag.VNNI_A) == 0
            and ((desc.flags & GEMMFlag.DECOMPRESS_A_VIA_BITMASK) > 0)
        )
        or (desc.datatype.c == Datatype.BF8 and (desc.flags & GEMMFlag.VNNI_A) > 0)
        or (desc.datatype.c == Datatype.BF8 and (desc.flags & GEMMFlag.VNNI_A) == 0)
        or (desc.datatype.c == Datatype.HF8 and (desc.flags & GEMMFlag.VNNI_A) > 0)
        or (desc.datatype.c == Datatype.HF8 and (desc.flags & GEMMFlag.VNNI_A) == 0)
    ):
        # /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32/Int32 out kernel with the same logic */
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 64, 16, &l_m_blocking, &l_use_masking_a_c );
        raise NotImplementedError
    elif (arch <= Arch.LIBXSMM_X86_ALLFEAT) and (Datatype.F64 == desc.datatype.ab):
        # libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 8, &l_m_blocking, &l_use_masking_a_c );
        raise NotImplementedError
    else:
        # /* we should never end up here, if we do let the user know */
        assert False

    config.use_masking_a_c = use_masking_a_c
    if use_masking_a_c:
        raise NotImplementedError

    return m_blocking


def libxsmm_generator_gemm_sse_avx_avx2_avx512_get_max_n_blocking(
    config: MicroKernelConfig, desc: GEMMDescriptor, arch: Arch
) -> int:
    l_is_Amxfp4_Bfp32_gemm = desc.is_Amxfp4_Bfp32_gemm()
    l_is_Amxfp4_Bbf16_gemm = desc.is_Amxfp4_Bbf16_gemm()
    l_is_Amxfp4_Bi8_gemm = desc.is_Amxfp4_Bi8_gemm()
    l_is_Ai2_Bi8_gemm = desc.is_Ai2_Bi8_gemm()
    l_is_Ai1_Bi8_gemm = desc.is_Ai1_Bi8_gemm()

    if arch >= Arch.LIBXSMM_X86_GENERIC and (
        l_is_Amxfp4_Bfp32_gemm or l_is_Amxfp4_Bbf16_gemm or l_is_Amxfp4_Bi8_gemm
    ):
        return 4
    elif arch >= Arch.LIBXSMM_X86_GENERIC and arch < Arch.LIBXSMM_X86_AVX512_VL256_SKX:
        if arch == Arch.LIBXSMM_X86_AVX2_SRF:
            if l_is_Ai2_Bi8_gemm:
                return 2
            elif l_is_Ai1_Bi8_gemm:
                return 6
            else:
                # There's some strange logic in this function that seems to be to do
                # with specifying what to return based on a command-line flag
                # return libxsmm_cpuid_x86_srf_gemm_set_n_max_blocking()
                return 3
        else:
            return 3
    elif arch >= Arch.LIBXSMM_X86_AVX512_VL256_SKX and arch <= Arch.LIBXSMM_X86_ALLFEAT:
        return 28
    else:
        # shouldn't happen
        pass
    return 0
