import os
from collections.abc import Iterator

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
    UNALLOCATED_REG64,
    GeneralRegisterType,
)
from xdsl.dialects.x86_func import FuncOp
from xdsl.ir import SSAValue
from xdsl.rewriter import InsertPoint
from autotuner.libxsmm_gemm.generator_common import (
    LIBXSMM_X86_AVX512_MASK_REG,
    GPRegMapping,
    KPhase,
    KPhaseKind,
    KPlan,
    LoopLabelTracker,
    MicroKernelConfig,
    MLoop,
    NLoop,
    libxsmm_compute_equalized_blocking,
)
from autotuner.libxsmm_gemm.generator_common_x86 import (
    libxsmm_generator_initialize_avx512_mask,
)
from autotuner.libxsmm_gemm.generator_gemm_avx512_microkernel import (
    libxsmm_generator_gemm_avx512_kloop_kernel,
)
from autotuner.libxsmm_gemm.generator_gemm_common import (
    libxsmm_generator_gemm_destroy_stack_frame,
    libxsmm_generator_gemm_footer_kloop,
    libxsmm_generator_gemm_footer_mloop,
    libxsmm_generator_gemm_footer_nloop,
    libxsmm_generator_gemm_get_blocking_and_mask,
    libxsmm_generator_gemm_header_kloop,
    libxsmm_generator_gemm_header_mloop,
    libxsmm_generator_gemm_header_nloop,
    libxsmm_generator_gemm_init_micro_kernel_config,
    libxsmm_generator_gemm_load_C,
    libxsmm_generator_gemm_setup_stack_frame,
    libxsmm_generator_gemm_store_C,
)
from autotuner.libxsmm_gemm.generator_x86_instructions import (
    libxsmm_x86_instruction_open_stream_gemm,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_generator import (
    GeneratedCode,
    KLoopVals,
    MLoopVals,
    NLoopVals,
)
from autotuner.libxsmm_gemm.libxsmm_main import (
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

    if os.environ.get("SWAP_A_B") == "1":
        gp_reg_a = RSI
        gp_reg_b = RDI
    else:
        gp_reg_a = RDI
        gp_reg_b = RSI

    gp_reg_mapping.gp_reg_param_struct = gp_reg_a

    gp_reg_mapping.gp_reg_a = gp_reg_mapping.gp_reg_param_struct
    gp_reg_mapping.gp_reg_b = gp_reg_b
    gp_reg_mapping.gp_reg_c = RDX
    gp_reg_mapping.gp_reg_a_prefetch = RCX
    gp_reg_mapping.gp_reg_b_prefetch = R8
    gp_reg_mapping.gp_reg_zpt = R9

    # Python translation of register assignment logic
    if (
        desc.datatype.a == desc.datatype.b == "I8"
        and desc.datatype.c == "F32"
        and not is_amxfp4_bi8_gemm
    ):
        gp_reg_mapping.gp_reg_scf = RCX
        gp_reg_mapping.gp_reg_a_prefetch = R8
        gp_reg_mapping.gp_reg_b_prefetch = R9
    elif is_amxfp4_bfp32_gemm or is_amxfp4_bbf16_gemm or is_amxfp4_bi8_gemm:
        gp_reg_mapping.gp_reg_scf = RCX
        if is_amxfp4_bi8_gemm:
            gp_reg_mapping.gp_reg_zpt = RBX
    elif (
        desc.datatype.a == "I8"
        and desc.datatype.b == "F16"
        and desc.datatype.c in ("F16", "F32")
    ) or (
        desc.datatype.a == "I8"
        and desc.datatype.b == "BF16"
        and desc.datatype.c in ("BF16", "F32")
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

    builder = Builder(InsertPoint.at_end(func_op.body.block))

    generated_code = GeneratedCode(func_op, builder, arch)

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
    micro_kernel_config = MicroKernelConfig()

    a_arg, b_arg, c_arg = generated_code.func_op.body.block.args
    a_val = SSAValue.get(a_arg, type=GeneralRegisterType)
    b_val = SSAValue.get(b_arg, type=GeneralRegisterType)
    c_val = SSAValue.get(c_arg, type=GeneralRegisterType)

    is_Ai4_Bf16_gemm = (
        ((desc.flags & GEMMFlag.INTERPRETE_A_AS_INT4_VNNI2) > 0)
        and (Datatype.I8 == desc.datatype.a)
        and (Datatype.F16 == desc.datatype.b)
        and (desc.datatype.c in (Datatype.F16, Datatype.F32))
    )

    is_Ai4_Bi8_gemm = desc.is_Ai4_Bi8_gemm()
    is_Ai2_Bi8_gemm = desc.is_Ai2_Bi8_gemm()
    is_Ai1_Bi8_gemm = desc.is_Ai1_Bi8_gemm()
    _is_Amxfp4_Bfp32_gemm = desc.is_Amxfp4_Bfp32_gemm()
    is_Amxfp4_Bbf16_gemm = desc.is_Amxfp4_Bbf16_gemm()
    is_Amxfp4_Bi8_gemm = desc.is_Amxfp4_Bi8_gemm()

    avnni_gemm_stack_alloc_tensors = (
        ((desc.flags & GEMMFlag.TRANS_A) == 0)
        and ((desc.flags & GEMMFlag.VNNI_A) == 0)
        and ((desc.flags & GEMMFlag.TRANS_B) == 0)
        and ((desc.flags & GEMMFlag.VNNI_B) == 0)
        and (desc.k % 2 == 0)
        and (Datatype.BF16 == desc.datatype.ab)
    )

    atvnni_gemm_stack_alloc_tensors = (
        ((desc.flags & GEMMFlag.TRANS_A) != 0)
        and ((desc.flags & GEMMFlag.VNNI_A) == 0)
        and ((desc.flags & GEMMFlag.TRANS_B) == 0)
        and ((desc.flags & GEMMFlag.VNNI_B) == 0)
        and (desc.k % 2 == 0)
        and (Datatype.BF16 == desc.datatype.ab)
    )

    avnni_btrans_gemm_stack_alloc_tensors = (
        ((desc.flags & GEMMFlag.TRANS_A) == 0)
        and ((desc.flags & GEMMFlag.VNNI_A) == 0)
        and ((desc.flags & GEMMFlag.TRANS_B) != 0)
        and ((desc.flags & GEMMFlag.VNNI_B) == 0)
        and (desc.k % 2 == 0)
        and (Datatype.BF16 == desc.datatype.ab)
    )

    atvnni_btrans_gemm_stack_alloc_tensors = (
        ((desc.flags & GEMMFlag.TRANS_A) != 0)
        and ((desc.flags & GEMMFlag.VNNI_A) == 0)
        and ((desc.flags & GEMMFlag.TRANS_B) != 0)
        and ((desc.flags & GEMMFlag.VNNI_B) == 0)
        and (desc.k % 2 == 0)
        and (Datatype.BF16 == desc.datatype.ab)
    )

    _adjust_A_pf_ptrs = 0
    _adjust_B_pf_ptrs = 0
    max_n_blocking = 0

    is_Ai8_Bbf16_gemm = (
        (Datatype.I8 == desc.datatype.a and not is_Amxfp4_Bbf16_gemm)
        and (Datatype.BF16 == desc.datatype.b)
        and (desc.datatype.c in (Datatype.BF16, Datatype.F32))
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
            _adjust_A_pf_ptrs = 1

    # In case of F16 and IMPLICIT compute set proper compute
    if (
        (
            (desc.datatype.a in (Datatype.I8, Datatype.BF8))
            and desc.datatype.b == Datatype.F16
        )
        or (desc.datatype.a == Datatype.F16 and desc.datatype.b == Datatype.F16)
    ) and desc.datatype.comp == Datatype.IMPLICIT:
        # if architecture is AMX (Sapphire Rapids or newer), compute in F16. Otherwise F32.
        if generated_code.arch >= Arch.LIBXSMM_X86_AVX512_SPR:
            # Set desc.datatype fields for computation in F16
            desc.datatype.comp = Datatype.F16
        else:
            # Set desc.datatype fields for computation in F32
            desc.datatype.comp = Datatype.F32

    # In case of BF8/HF8 we might need to set different precisions
    if (
        desc.datatype.a == Datatype.BF8
        and desc.datatype.b in (Datatype.BF8, Datatype.HF8)
    ) or (
        desc.datatype.a == Datatype.HF8
        and desc.datatype.b in (Datatype.BF8, Datatype.HF8)
    ):
        raise NotImplementedError

    # TODO: check if we can make this smarter and don't need two times the same if
    if (
        avnni_gemm_stack_alloc_tensors
        or atvnni_gemm_stack_alloc_tensors
        or avnni_btrans_gemm_stack_alloc_tensors
        or atvnni_btrans_gemm_stack_alloc_tensors
    ):
        desc.flags |= GEMMFlag.VNNI_A

    if atvnni_gemm_stack_alloc_tensors or atvnni_btrans_gemm_stack_alloc_tensors:
        desc.flags &= ~GEMMFlag.TRANS_A
        assert desc.lda % 2 == 0, "LIBXSMM_ERR_VNNI_A"

    if avnni_btrans_gemm_stack_alloc_tensors or atvnni_btrans_gemm_stack_alloc_tensors:
        desc.flags &= ~GEMMFlag.TRANS_B

    # Define the micro kernel code gen properties
    libxsmm_generator_gemm_init_micro_kernel_config(
        micro_kernel_config, generated_code.arch, desc, False
    )

    # setup hf8 / bf8 conversion on stack before GEMM, we need to recheck as we now can update the field in ukernel config, need to use the original GEMM descriptor
    if desc.datatype.ab == Datatype.BF8:
        micro_kernel_config.bf8_gemm_via_stack_alloc_tensors = 1

    if desc.datatype.ab == Datatype.HF8:
        micro_kernel_config.bf8_gemm_via_stack_alloc_tensors = 1

    # in case when A needs to be transposed, we need to change temporarily the descriptor dimensions for gemm
    if GEMMFlag.TRANS_A in desc.flags:
        assert desc.datatype.abc in (Datatype.F32, Datatype.F64)
        desc.lda = desc.m
        desc.flags &= ~GEMMFlag.TRANS_A
        micro_kernel_config.atrans_gemm_stack_alloc_tensors = 1
    elif GEMMFlag.TRANS_B | GEMMFlag.VNNI_B in desc.flags:
        raise NotImplementedError

    # handle A VNNI on stack */
    if avnni_gemm_stack_alloc_tensors:
        desc.lda = desc.m
        micro_kernel_config.avnni_gemm_stack_alloc_tensors = True

    if atvnni_gemm_stack_alloc_tensors:
        desc.lda = desc.m
        micro_kernel_config.atvnni_gemm_stack_alloc_tensors = True

    if avnni_btrans_gemm_stack_alloc_tensors:
        desc.lda = desc.m
        desc.ldb = desc.k
        micro_kernel_config.avnni_btrans_gemm_stack_alloc_tensors = True

    if atvnni_btrans_gemm_stack_alloc_tensors:
        desc.lda = desc.m
        desc.ldb = desc.k
        micro_kernel_config.atvnni_btrans_gemm_stack_alloc_tensors = True

    # Block according to the number of available registers or given limits
    max_n_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_compute_max_n_blocking(
        micro_kernel_config, desc, generated_code.arch
    )
    n_loops = libxsmm_generator_gemm_sse_avx_avx2_avx512_iter_n_loops(
        desc.n, max_n_blocking
    )

    # implementing load from struct
    if GEMMFlag.USE_XGEMM_ABI in desc.flags or GEMMFlag.USE_XGEMM_EXT_ABI in desc.flags:
        raise NotImplementedError

    if GEMMFlag.USE_XGEMM_EXT_ABI in desc.flags or micro_kernel_config.vnni_format_C:
        raise NotImplementedError

    # Setting up the stack frame
    rbp_val, rsp_val = libxsmm_generator_gemm_setup_stack_frame(
        generated_code, desc, gp_reg_mapping, micro_kernel_config
    )

    # In this case we store C to scratch */
    if micro_kernel_config.vnni_format_C:
        raise NotImplementedError

    # Apply potential opA / opB
    # libxsmm_generator_gemm_apply_opA_opB( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc, i_xgemm_desc);

    loop_label_tracker = LoopLabelTracker()

    # generate hoisted BF16 emulation mask for AVX512
    if desc.datatype.ab == Datatype.BF16:
        raise NotImplementedError

    # generate hoisted UU SS i8 emulation mask for AVX512
    if desc.datatype.ab == Datatype.I8:
        raise NotImplementedError

    if is_Amxfp4_Bi8_gemm:
        raise NotImplementedError

    if is_Ai4_Bf16_gemm:
        raise NotImplementedError

    if is_Ai4_Bi8_gemm:
        raise NotImplementedError

    if is_Ai2_Bi8_gemm:
        raise NotImplementedError

    if is_Ai1_Bi8_gemm:
        raise NotImplementedError

    if is_Ai8_Bbf16_gemm:
        raise NotImplementedError

    if (
        desc.datatype.a == Datatype.I8
        and desc.datatype.b == Datatype.F16
        and desc.datatype.c in (Datatype.F16, Datatype.F32)
    ):
        raise NotImplementedError

    if (
        (desc.datatype.a == Datatype.I8 and not is_Amxfp4_Bbf16_gemm)
        and desc.datatype.b == Datatype.BF16
        and (desc.datatype in (Datatype.BF16, Datatype.F32))
    ):
        raise NotImplementedError

    # Load the actual batch-reduce trip count
    if (
        GEMMFlag.BATCH_REDUCE_ADDRESS in desc.flags
        or GEMMFlag.BATCH_REDUCE_OFFSET in desc.flags
        or GEMMFlag.BATCH_REDUCE_STRIDE in desc.flags
    ):
        raise NotImplementedError

    # apply n_blocking
    for n_loop in n_loops:
        n_blocking = n_loop.n_blocking
        n_done = n_loop.start

        # open N loop
        nloop_vals = libxsmm_generator_gemm_header_nloop(
            generated_code,
            loop_label_tracker,
            gp_reg_mapping,
            micro_kernel_config,
            n_done,
            n_blocking,
            a_val,
            b_val,
            c_val,
            rbp_val,
            rsp_val,
        )
        a_val = nloop_vals.a
        b_val = nloop_vals.b
        c_val = nloop_vals.c
        rbp_val = nloop_vals.rbp
        rsp_val = nloop_vals.rsp
        n_counter_val = nloop_vals.n_counter

        if GEMMFlag.DECOMPRESS_A_VIA_BITMASK in desc.flags:
            raise NotImplementedError

        # advance N
        n_done += n_loop.extent

        micro_kernel_config.m_bitmask_advance = 0  # @TODO: FOR SSE ONLY and relumask

        # apply m_blocking (M split shared with the visualizer; the generator here
        # drives it, so get_m_blocking's config side effects interleave as before)
        for m_loop in libxsmm_generator_gemm_sse_avx_avx2_avx512_iter_m_loops(
            micro_kernel_config, desc, generated_code.arch
        ):
            m_blocking = m_loop.m_blocking
            m_done_old = m_loop.start
            micro_kernel_config.current_m = m_done_old

            # coverity[divide_by_zero]
            m_done = m_loop.start + m_loop.count * m_loop.m_blocking
            micro_kernel_config.m_bitmask_advance += (
                m_done - m_done_old
            ) // 8  # @TODO: FOR SSE ONLY and relumask

            if (m_done != m_done_old) and (m_done > 0):
                mask_k1_val = None
                # when on AVX512, load mask, if needed
                if (
                    micro_kernel_config.use_masking_a_c
                    and Arch.LIBXSMM_X86_AVX512_VL256_SKX
                    <= generated_code.arch
                    <= Arch.LIBXSMM_X86_ALLFEAT
                ):
                    # compute the mask count, depends on vlen as block in M
                    corrected_vlen = micro_kernel_config.vector_length
                    mask_count = corrected_vlen - (m_blocking % corrected_vlen)

                    if (
                        (
                            (Datatype.F16 == desc.datatype.ab)
                            and (Datatype.F16 == desc.datatype.c)
                        )
                        or (
                            (Datatype.F16 == desc.datatype.ab)
                            and (Datatype.F32 == desc.datatype.c)
                        )
                        or (
                            (Datatype.BF16 == desc.datatype.ab)
                            and (Datatype.BF16 == desc.datatype.c)
                        )
                        or (
                            (Datatype.I8 == desc.datatype.ab)
                            and (Datatype.I8 == desc.datatype.c)
                        )
                        or (
                            (Datatype.BF8 == desc.datatype.ab)
                            and (Datatype.BF8 == desc.datatype.c)
                        )
                        or (
                            (Datatype.F32 == desc.datatype.ab)
                            and (Datatype.BF8 == desc.datatype.c)
                        )
                        or (
                            (Datatype.BF16 == desc.datatype.ab)
                            and (Datatype.BF8 == desc.datatype.c)
                        )
                        or (
                            (Datatype.HF8 == desc.datatype.ab)
                            and (Datatype.HF8 == desc.datatype.c)
                        )
                        or (
                            (Datatype.BF16 == desc.datatype.ab)
                            and (Datatype.HF8 == desc.datatype.c)
                        )
                        or (
                            (Datatype.BF8 == desc.datatype.a)
                            and (Datatype.F16 == desc.datatype.b)
                            and (Datatype.F16 == desc.datatype.c)
                        )
                        or (
                            (Datatype.BF8 == desc.datatype.a)
                            and (Datatype.F16 == desc.datatype.b)
                            and (Datatype.F32 == desc.datatype.c)
                        )
                        or (
                            (Datatype.I8 == desc.datatype.a)
                            and (Datatype.F16 == desc.datatype.b)
                            and (Datatype.F16 == desc.datatype.c)
                        )
                        or (
                            (Datatype.I8 == desc.datatype.a)
                            and (Datatype.F16 == desc.datatype.b)
                            and (Datatype.F32 == desc.datatype.c)
                        )
                        or (
                            (Datatype.I8 == desc.datatype.a)
                            and (Datatype.BF16 == desc.datatype.b)
                            and (Datatype.BF16 == desc.datatype.c)
                        )
                        or (
                            (Datatype.I8 == desc.datatype.a)
                            and (Datatype.BF16 == desc.datatype.b)
                            and (Datatype.F32 == desc.datatype.c)
                        )
                        or (
                            (Datatype.F32 == desc.datatype.ab)
                            and (Datatype.HF8 == desc.datatype.c)
                        )
                    ):
                        is_Ai8_Bf16_gemm = (Datatype.I8 == desc.datatype.a) and (
                            Datatype.F16 == desc.datatype.b
                        )
                        is_Abf8_Bf16_gemm = (Datatype.BF8 == desc.datatype.a) and (
                            Datatype.F16 == desc.datatype.b
                        )
                        is_Af16_Bf16_gemm = (Datatype.F16 == desc.datatype.a) and (
                            Datatype.F16 == desc.datatype.b
                        )
                        is_compute_f16_gemm = (
                            is_Ai8_Bf16_gemm or is_Abf8_Bf16_gemm or is_Af16_Bf16_gemm
                        ) and (
                            Datatype.F16 == desc.datatype.comp
                            and (generated_code.arch >= Arch.LIBXSMM_X86_AVX512_SPR)
                        )
                        mask_k1_val = libxsmm_generator_initialize_avx512_mask(
                            generated_code,
                            gp_reg_mapping.gp_reg_help_1,
                            LIBXSMM_X86_AVX512_MASK_REG,
                            mask_count,
                            Datatype.F16 if is_compute_f16_gemm else Datatype.I32,
                        )
                        if is_compute_f16_gemm and Datatype.F32 == desc.datatype.c:
                            # Adjust mask for C handling
                            c_vlen_adjusted = (
                                corrected_vlen // 2
                                if is_compute_f16_gemm
                                else corrected_vlen
                            )
                            libxsmm_generator_initialize_avx512_mask(
                                generated_code,
                                gp_reg_mapping.gp_reg_help_1,
                                x86.registers.AVX512MaskRegisterType.from_index(2),
                                0
                                if m_blocking % corrected_vlen >= c_vlen_adjusted
                                else c_vlen_adjusted - (m_blocking % c_vlen_adjusted),
                                Datatype.I32,
                            )
                            libxsmm_generator_initialize_avx512_mask(
                                generated_code,
                                gp_reg_mapping.gp_reg_help_1,
                                x86.registers.AVX512MaskRegisterType.from_index(3),
                                c_vlen_adjusted - (m_blocking % c_vlen_adjusted)
                                if m_blocking % corrected_vlen >= c_vlen_adjusted
                                else c_vlen_adjusted,
                                Datatype.I32,
                            )
                        else:
                            # we have to adjust mask count as for now we are using ymm for 16bit and xmm for 8bit
                            if (
                                generated_code.arch >= Arch.LIBXSMM_X86_AVX512_VL256_SKX
                            ) and (generated_code.arch < Arch.LIBXSMM_X86_AVX512_SKX):
                                mask_count = (
                                    mask_count + 8
                                    if (
                                        (
                                            (Datatype.BF16 == desc.datatype.ab)
                                            and (Datatype.HF8 != desc.datatype.c)
                                            and (Datatype.BF8 != desc.datatype.c)
                                        )
                                        or (is_Ai8_Bbf16_gemm)
                                    )
                                    else mask_count + 24
                                )
                            else:
                                mask_count = (
                                    mask_count + 16
                                    if (
                                        (
                                            (Datatype.BF16 == desc.datatype.ab)
                                            and (Datatype.HF8 != desc.datatype.c)
                                            and (Datatype.BF8 != desc.datatype.c)
                                        )
                                        or (is_Ai8_Bbf16_gemm)
                                    )
                                    else mask_count + 48
                                )

                        libxsmm_generator_initialize_avx512_mask(
                            generated_code,
                            gp_reg_mapping.gp_reg_help_1,
                            x86.registers.AVX512MaskRegisterType.from_index(2),
                            mask_count,
                            desc.datatype.c,
                        )
                    else:
                        mask_k1_val = libxsmm_generator_initialize_avx512_mask(
                            generated_code,
                            gp_reg_mapping.gp_reg_help_1,
                            LIBXSMM_X86_AVX512_MASK_REG,
                            mask_count,
                            desc.datatype.c,
                        )
                elif (
                    micro_kernel_config.use_masking_a_c
                    and Arch.LIBXSMM_X86_AVX
                    <= generated_code.arch
                    < Arch.LIBXSMM_X86_AVX512_VL256_SKX
                ):
                    raise NotImplementedError

                mloop_vals = libxsmm_generator_gemm_header_mloop(
                    generated_code,
                    loop_label_tracker,
                    gp_reg_mapping,
                    micro_kernel_config,
                    m_done_old,
                    m_blocking,
                    NLoopVals(a_val, b_val, c_val, rbp_val, rsp_val, n_counter_val),
                    mask_k1_val,
                )
                a_val = mloop_vals.a
                b_val = mloop_vals.b
                c_val = mloop_vals.c
                rbp_val = mloop_vals.rbp
                rsp_val = mloop_vals.rsp
                n_counter_val = mloop_vals.n_counter
                m_counter_val = mloop_vals.m_counter
                mask_k1_val = mloop_vals.mask_k1

                acc_vals = libxsmm_generator_gemm_load_C(
                    generated_code,
                    gp_reg_mapping,
                    micro_kernel_config,
                    desc,
                    m_blocking,
                    n_blocking,
                    c_val=c_val,
                    mask_k1=mask_k1_val,
                )

                if (
                    GEMMFlag.BATCH_REDUCE_ADDRESS in desc.flags
                    or GEMMFlag.BATCH_REDUCE_OFFSET in desc.flags
                    or GEMMFlag.BATCH_REDUCE_STRIDE in desc.flags
                ):
                    raise NotImplementedError

                kloop_vals = libxsmm_generator_gemm_sse_avx_avx2_avx512_kloop(
                    generated_code,
                    loop_label_tracker,
                    gp_reg_mapping,
                    micro_kernel_config,
                    desc,
                    m_blocking,
                    n_blocking,
                    KLoopVals(
                        a_val,
                        b_val,
                        c_val,
                        rbp_val,
                        rsp_val,
                        n_counter_val,
                        m_counter_val,
                        mask_k1_val,
                        acc_vals,
                    ),
                )
                a_val = kloop_vals.a
                b_val = kloop_vals.b
                c_val = kloop_vals.c
                rbp_val = kloop_vals.rbp
                rsp_val = kloop_vals.rsp
                n_counter_val = kloop_vals.n_counter
                m_counter_val = kloop_vals.m_counter
                mask_k1_val = kloop_vals.mask_k1

                if (
                    GEMMFlag.BATCH_REDUCE_ADDRESS in desc.flags
                    or GEMMFlag.BATCH_REDUCE_OFFSET in desc.flags
                    or GEMMFlag.BATCH_REDUCE_STRIDE in desc.flags
                ):
                    raise NotImplementedError

                libxsmm_generator_gemm_store_C(
                    generated_code,
                    gp_reg_mapping,
                    micro_kernel_config,
                    desc,
                    m_blocking,
                    n_blocking,
                    c_val=c_val,
                    acc_vectors=kloop_vals.acc_vectors,
                    mask_k1=mask_k1_val,
                )
                mloop_result = libxsmm_generator_gemm_footer_mloop(
                    generated_code,
                    loop_label_tracker,
                    gp_reg_mapping,
                    micro_kernel_config,
                    desc,
                    m_blocking,
                    m_done,
                    MLoopVals(
                        a_val,
                        b_val,
                        c_val,
                        rbp_val,
                        rsp_val,
                        n_counter_val,
                        m_counter_val,
                        mask_k1_val,
                    ),
                )
                a_val = mloop_result.a
                b_val = mloop_result.b
                c_val = mloop_result.c
                rbp_val = mloop_result.rbp
                rsp_val = mloop_result.rsp
                n_counter_val = mloop_result.n_counter
                m_counter_val = mloop_result.m_counter
                mask_k1_val = mloop_result.mask_k1

        nloop_result = libxsmm_generator_gemm_footer_nloop(
            generated_code,
            loop_label_tracker,
            gp_reg_mapping,
            micro_kernel_config,
            desc,
            n_blocking,
            n_done,
            NLoopVals(a_val, b_val, c_val, rbp_val, rsp_val, n_counter_val),
        )
        a_val = nloop_result.a
        b_val = nloop_result.b
        c_val = nloop_result.c
        rbp_val = nloop_result.rbp
        rsp_val = nloop_result.rsp
        n_counter_val = nloop_result.n_counter

    # In this case we vnni-format C from scratch
    if micro_kernel_config.vnni_format_C:
        raise NotImplementedError

    # destroy stack frame
    libxsmm_generator_gemm_destroy_stack_frame(
        generated_code, desc, gp_reg_mapping, micro_kernel_config, rbp_val
    )


def libxsmm_generator_gemm_sse_avx_avx2_avx512_kloop(
    generated_code: GeneratedCode,
    label_tracker: LoopLabelTracker,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    m_blocking: int,
    n_blocking: int,
    kloop_vals: KLoopVals,
) -> KLoopVals:
    _k_pack_factor = 1
    is_Amxfp4_Bbf16_gemm = desc.is_Amxfp4_Bbf16_gemm()
    is_Ai8_Bbf16_gemm = (
        desc.datatype.a == "I8"
        and not is_Amxfp4_Bbf16_gemm
        and desc.datatype.b == "BF16"
    )
    is_Ai8_Bbf16_gemm_bf16fma = (
        False  # micro_kernel_config.vmul_instruction == "VDPBF16PS"
    )
    is_Amxfp4_Bfp32_gemm = desc.is_Amxfp4_Bfp32_gemm()
    is_Amxfp4_Bi8_gemm = desc.is_Amxfp4_Bi8_gemm()
    is_Ai4_Bi8_gemm = desc.is_Ai4_Bi8_gemm()

    # K split (a very simple k unrolling model): parameters + emission phases
    k_plan = libxsmm_generator_gemm_sse_avx_avx2_avx512_compute_k_plan(
        desc, generated_code.arch
    )
    k_blocking = k_plan.k_blocking

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

    # Apply the k_blocking strategy chosen by compute_k_plan: emit one hardware K-loop
    # per "looped" phase and a bare unrolled body per "unrolled"/"remainder" phase.
    for phase in k_plan.phases:
        if phase.kind is KPhaseKind.LOOPED:
            block_vals = libxsmm_generator_gemm_header_kloop(
                generated_code,
                label_tracker,
                gp_reg_mapping,
                micro_kernel_config,
                m_blocking,
                k_blocking,
                kloop_vals,
            )
            block_vals = generator_kloop_kernel(
                generated_code,
                gp_reg_mapping,
                micro_kernel_config,
                desc,
                m_blocking,
                n_blocking,
                k_blocking,
                block_vals,
            )
            kloop_vals = libxsmm_generator_gemm_footer_kloop(
                generated_code,
                label_tracker,
                gp_reg_mapping,
                micro_kernel_config,
                desc,
                m_blocking,
                phase.extent,
                phase.full,
                block_vals,
            )
        else:
            # fully-unrolled body (strategy 2) or the strategy-3 remainder
            kloop_vals = generator_kloop_kernel(
                generated_code,
                gp_reg_mapping,
                micro_kernel_config,
                desc,
                m_blocking,
                n_blocking,
                phase.size,
                kloop_vals,
            )

    # Reset B pointer after a blocked + remainder (strategy 3) decomposition
    if k_plan.strategy == 3:
        if GEMMFlag.TRANS_B in desc.flags:
            b_offset = desc.ldb * desc.k * micro_kernel_config.datatype_size_in2
        else:
            b_offset = desc.k * micro_kernel_config.datatype_size_in2

        kloop_vals.b = generated_code.insert(
            x86.ops.RI_SubOp(
                kloop_vals.b,
                b_offset,
                register_out=gp_reg_mapping.gp_reg_b,
            )
        ).register_out

    if is_Ai8_Bbf16_gemm and not is_Ai8_Bbf16_gemm_bf16fma:
        raise NotImplementedError

    return kloop_vals


def libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking(
    config: MicroKernelConfig, desc: GEMMDescriptor, arch: Arch, current_m_blocking: int
):
    use_masking_a_c = False
    m_blocking = current_m_blocking
    is_Amxfp4_Bfp32_gemm = desc.is_Amxfp4_Bfp32_gemm()
    is_Amxfp4_Bi8_gemm = desc.is_Amxfp4_Bi8_gemm()
    is_Amxfp4_Bbf16_gemm = desc.is_Amxfp4_Bbf16_gemm()

    if (arch <= Arch.LIBXSMM_X86_SSE42) and (Datatype.F32 == desc.datatype.ab):
        if config.fused_relu:
            m_blocking, use_masking_a_c = libxsmm_generator_gemm_get_blocking_and_mask(
                desc.m, 8, 4, m_blocking
            )
        else:
            m_blocking, use_masking_a_c = libxsmm_generator_gemm_get_blocking_and_mask(
                desc.m, 12, 4, m_blocking
            )
    elif (arch <= Arch.LIBXSMM_X86_SSE42) and (Datatype.F64 == desc.datatype.ab):
        if config.fused_relu:
            m_blocking, use_masking_a_c = libxsmm_generator_gemm_get_blocking_and_mask(
                desc.m, 4, 2, m_blocking
            )
        else:
            m_blocking, use_masking_a_c = libxsmm_generator_gemm_get_blocking_and_mask(
                desc.m, 6, 2, m_blocking
            )
    elif (arch <= Arch.LIBXSMM_X86_SSE42) and (Datatype.I8 == desc.datatype.ab):
        m_blocking, use_masking_a_c = libxsmm_generator_gemm_get_blocking_and_mask(
            desc.m, 8, 4, m_blocking
        )
    elif (arch <= Arch.LIBXSMM_X86_SSE42) and (Datatype.I16 == desc.datatype.ab):
        m_blocking, use_masking_a_c = libxsmm_generator_gemm_get_blocking_and_mask(
            desc.m, 8, 4, m_blocking
        )
    elif (arch <= Arch.LIBXSMM_X86_SSE42) and (Datatype.BF16 == desc.datatype.ab):
        raise NotImplementedError
    elif (arch == Arch.LIBXSMM_X86_AVX) and (Datatype.F32 == desc.datatype.ab):
        m_blocking, use_masking_a_c = libxsmm_generator_gemm_get_blocking_and_mask(
            desc.m, 24, 8, m_blocking
        )
    elif (arch == Arch.LIBXSMM_X86_AVX) and (Datatype.F64 == desc.datatype.ab):
        m_blocking, use_masking_a_c = libxsmm_generator_gemm_get_blocking_and_mask(
            desc.m, 12, 4, m_blocking
        )
    elif (arch >= Arch.LIBXSMM_X86_AVX2) and (
        is_Amxfp4_Bfp32_gemm > 0 or is_Amxfp4_Bbf16_gemm > 0 or is_Amxfp4_Bi8_gemm > 0
    ):
        raise NotImplementedError
    elif (arch == Arch.LIBXSMM_X86_AVX2_SRF) and (
        desc.datatype.c in (Datatype.F32, Datatype.BF16, Datatype.I32)
    ):
        raise NotImplementedError
    elif (
        (arch >= Arch.LIBXSMM_X86_AVX2) and (arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX)
    ) and (desc.datatype.c in (Datatype.F32, Datatype.BF16, Datatype.I32)):
        m_blocking, use_masking_a_c = libxsmm_generator_gemm_get_blocking_and_mask(
            desc.m, 32, 8, m_blocking
        )
        raise NotImplementedError
    elif (arch == Arch.LIBXSMM_X86_AVX2_SRF) and (Datatype.F64 == desc.datatype.ab):
        raise NotImplementedError
    elif (
        (arch >= Arch.LIBXSMM_X86_AVX2) and (arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX)
    ) and (Datatype.F64 == desc.datatype.ab):
        m_blocking, use_masking_a_c = libxsmm_generator_gemm_get_blocking_and_mask(
            desc.m, 16, 4, m_blocking
        )
    elif (
        (
            (arch >= Arch.LIBXSMM_X86_AVX512_VL256_SKX)
            and (arch < Arch.LIBXSMM_X86_AVX512_SKX)
        )
        and (Datatype.BF16 == desc.datatype.ab)
        and ((desc.flags & GEMMFlag.VNNI_A) == 0)
    ):
        raise NotImplementedError
    elif (
        ((arch >= Arch.LIBXSMM_X86_AVX512_SKX) and (arch <= Arch.LIBXSMM_X86_ALLFEAT))
        and (Datatype.BF16 == desc.datatype.ab)
        and ((desc.flags & GEMMFlag.VNNI_A) == 0)
        and ((desc.flags & GEMMFlag.DECOMPRESS_A_VIA_BITMASK) == 0)
    ):
        raise NotImplementedError
    elif (arch == Arch.LIBXSMM_X86_AVX512_VL256_SKX) and (
        desc.datatype.ab in (Datatype.I8, Datatype.I16)
    ):
        raise NotImplementedError
    elif (
        (arch >= Arch.LIBXSMM_X86_AVX512_SKX)
        and (arch <= Arch.LIBXSMM_X86_AVX512_SKX)
        and (desc.datatype.ab in (Datatype.I8, Datatype.I16))
    ):
        raise NotImplementedError
    elif (
        (desc.datatype.c in (Datatype.F16, Datatype.F32))
        and (desc.datatype.ab == Datatype.F16)
    ) or (
        (desc.datatype.c in (Datatype.F16, Datatype.F32))
        and (desc.datatype.b == Datatype.F16)
        and desc.datatype.a in (Datatype.I8, Datatype.BF8)
    ):
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
        raise NotImplementedError
    elif (
        (arch >= Arch.LIBXSMM_X86_AVX512_VL256_SKX)
        and (arch < Arch.LIBXSMM_X86_AVX512_SKX)
        and (Datatype.F64 == desc.datatype.ab)
    ):
        m_blocking, use_masking_a_c = libxsmm_generator_gemm_get_blocking_and_mask(
            desc.m, 32, 4, m_blocking
        )
    elif (
        (desc.datatype.c in (Datatype.BF16, Datatype.F32))
        and (desc.datatype.b == Datatype.BF16)
        and (desc.datatype.a == Datatype.I8)
    ):
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
        m_blocking, use_masking_a_c = libxsmm_generator_gemm_get_blocking_and_mask(
            desc.m, 64, 16, m_blocking
        )
    elif (arch <= Arch.LIBXSMM_X86_ALLFEAT) and (Datatype.F64 == desc.datatype.ab):
        m_blocking, use_masking_a_c = libxsmm_generator_gemm_get_blocking_and_mask(
            desc.m, 32, 8, m_blocking
        )
    else:
        # /* we should never end up here, if we do let the user know */
        assert False

    config.use_masking_a_c = use_masking_a_c
    if use_masking_a_c:
        if config.c_vmove_nts_instruction == x86.ops.MS_VmovntpsOp:
            config.c_vmove_nts_instruction = x86.ops.MS_VmovapsOp
        elif config.c_vmove_nts_instruction == x86.ops.MS_VmovntpdOp:
            config.c_vmove_nts_instruction = x86.ops.MS_VmovapdOp
    else:
        if config.c_vmove_nts_instruction == x86.ops.MS_VmovapsOp:
            config.c_vmove_nts_instruction = x86.ops.MS_VmovntpsOp
        elif config.c_vmove_nts_instruction == x86.ops.MS_VmovapdOp:
            config.c_vmove_nts_instruction = x86.ops.MS_VmovntpdOp

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


# --------------------------------------------------------------------------------------
# Tiling planner


def libxsmm_generator_gemm_sse_avx_avx2_avx512_compute_max_n_blocking(
    config: MicroKernelConfig, desc: GEMMDescriptor, arch: Arch
) -> int:
    """N accumulator blocking (mirrors the top of ..._kernel).

    Starts from the architecture cap, then shrinks until the N accumulators plus the M
    block registers fit into the vector register file. The ``get_m_blocking(config, …,
    0)`` probe mutates ``config.use_masking_a_c`` as a side effect, exactly as the
    emitting kernel does just before the N loop, so callers get identical config state.
    """
    max_n_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_max_n_blocking(
        config, desc, arch
    )
    if max_n_blocking > 3:
        init_m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking(
            config, desc, arch, 0
        )
        init_m_blocks = (
            init_m_blocking + config.vector_length - 1
        ) // config.vector_length
        is_Ai8_Bf16_gemm = (
            desc.datatype.a == Datatype.I8
            and desc.datatype.b == Datatype.F16
            and (desc.datatype.c in (Datatype.F16, Datatype.F32))
        )
        is_Ai8_Bbf16_gemm = (
            (desc.datatype.a == Datatype.I8 and not desc.is_Amxfp4_Bbf16_gemm())
            and (desc.datatype.b == Datatype.BF16)
            and (desc.datatype.c in (Datatype.BF16, Datatype.F32))
        )

        if is_Ai8_Bf16_gemm:
            raise NotImplementedError
        elif is_Ai8_Bbf16_gemm:
            raise NotImplementedError
        elif desc.is_Ai4_Bi8_gemm():
            raise NotImplementedError
        elif desc.is_Ai2_Bi8_gemm():
            raise NotImplementedError
        elif desc.is_Ai1_Bi8_gemm():
            raise NotImplementedError
        else:
            if Arch.LIBXSMM_X86_AVX2_SRF <= arch < Arch.LIBXSMM_X86_AVX512_SKX:
                while (
                    init_m_blocks * max_n_blocking + max_n_blocking + 1
                ) > config.vector_reg_count:
                    max_n_blocking -= 1
            else:
                while (
                    init_m_blocks * max_n_blocking + init_m_blocks + 1
                ) > config.vector_reg_count:
                    max_n_blocking -= 1

    assert max_n_blocking
    return max_n_blocking


def libxsmm_generator_gemm_sse_avx_avx2_avx512_iter_n_loops(
    n: int, max_n_blocking: int
) -> list[NLoop]:
    """The N split: one ``NLoop`` per equalized-blocking tier (mirrors ..._kernel)."""
    blocking = libxsmm_compute_equalized_blocking(n, max_n_blocking)
    n_N = [blocking.range_1, blocking.range_2]
    n_n = [blocking.block_1, blocking.block_2]
    assert n_N[0]

    loops: list[NLoop] = []
    n_done = 0
    n_count = 0
    while n_done != n:
        loops.append(NLoop(n_done, n_n[n_count], n_N[n_count]))
        n_done += n_N[n_count]
        n_count += 1
    return loops


def libxsmm_generator_gemm_sse_avx_avx2_avx512_iter_m_loops(
    config: MicroKernelConfig, desc: GEMMDescriptor, arch: Arch
) -> Iterator[MLoop]:
    """The M split (mirrors the ``while m_done`` walk in ..._kernel).

    A generator so the per-group ``get_m_blocking`` calls — which mutate
    ``config.use_masking_a_c`` / ``config.c_vmove_nts_instruction`` — interleave with the
    caller's per-group work exactly as in the emitting kernel: at each ``yield`` the
    config reflects the group being yielded. One ``MLoop`` per emitted hardware M-loop.
    """
    m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking(
        config, desc, arch, 0
    )
    m_done = 0
    while m_done != desc.m:
        assert m_blocking
        m_done_old = m_done
        # Consume all full m_blocking chunks at once (one hardware M-loop).
        count = (desc.m - m_done_old) // m_blocking
        m_done = m_done_old + count * m_blocking
        if m_done != m_done_old:
            yield MLoop(m_done_old, m_blocking, count, config.use_masking_a_c)
        # Recompute to obtain the (smaller) remainder block width for the next group.
        m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking(
            config, desc, arch, m_blocking
        )


def libxsmm_generator_gemm_sse_avx_avx2_avx512_compute_k_plan(
    desc: GEMMDescriptor, arch: Arch
) -> KPlan:
    """The K split (mirrors the parameters + 3-strategy branch of ..._kloop).

    Returns the ``k_blocking``/``k_threshold`` and the emission phases without emitting
    or selecting the micro-kernel.
    """
    is_Amxfp4_Bbf16_gemm = desc.is_Amxfp4_Bbf16_gemm()
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
    is_Amxfp4_Bfp32_gemm = desc.is_Amxfp4_Bfp32_gemm()
    is_Amxfp4_Bi8_gemm = desc.is_Amxfp4_Bi8_gemm()
    is_i8_uu_ss_gemm = desc.datatype.ab == "I8" and (
        GEMMFlag.A_UNSIGNED in desc.flags == GEMMFlag.B_UNSIGNED in desc.flags
    )

    # a very simple k unrolling model
    k_blocking = 4
    k_threshold = 23

    if GEMMFlag.VNNI_A in desc.flags:
        # VNNI kernel should maintain the same amount of unrolled instructions
        raise NotImplementedError

    if is_i8_uu_ss_gemm and arch in (
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

    k = desc.k
    if not k % k_blocking and k_threshold < k:
        # 1. larger than the threshold and a multiple of the blocking parameter
        strategy = 1
        phases = [KPhase(KPhaseKind.LOOPED, k_blocking, k // k_blocking, k, True)]
    elif k <= k_threshold:
        # 2. fully unroll below the threshold
        strategy = 2
        phases = [KPhase(KPhaseKind.UNROLLED, k, 1, k, False)]
    else:
        # 3. largest possible blocking + remainder handling
        strategy = 3
        l_max_blocked_k = (k // k_blocking) * k_blocking
        phases: list[KPhase] = []
        if l_max_blocked_k > 0:
            phases.append(
                KPhase(
                    KPhaseKind.LOOPED,
                    k_blocking,
                    l_max_blocked_k // k_blocking,
                    l_max_blocked_k,
                    False,
                )
            )
        phases.append(
            KPhase(
                KPhaseKind.REMAINDER, k - l_max_blocked_k, 1, k - l_max_blocked_k, False
            )
        )

    return KPlan(k_blocking, k_threshold, strategy, phases)
