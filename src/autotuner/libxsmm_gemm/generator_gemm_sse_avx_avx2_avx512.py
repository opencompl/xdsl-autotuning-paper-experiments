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
    libxsmm_compute_equalized_blocking,
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
    micro_kernel_config = MicroKernelConfig()
    # These values may be modified below
    m, n, k, lda, ldb, ldc, dt, flags, prefetch = desc

    is_Ai4_Bf16_gemm = (
        ((desc.flags & GEMMFlag.INTERPRETE_A_AS_INT4_VNNI2) > 0)
        and (Datatype.I8 == dt.a)
        and (Datatype.F16 == dt.b)
        and (dt.c in (Datatype.F16, Datatype.F32))
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
    n_count = 0  # array counter for blocking arrays
    n_done = 0  # progress tracker
    n_n = [0, 0]  # blocking sizes for blocks
    n_N = [0, 0]  # size of blocks

    _adjust_A_pf_ptrs = 0
    _adjust_B_pf_ptrs = 0
    max_n_blocking = 0

    is_Ai8_Bbf16_gemm = (
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
            _adjust_A_pf_ptrs = 1

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

    # Define the micro kernel code gen properties
    libxsmm_generator_gemm_init_micro_kernel_config(
        micro_kernel_config, generated_code.arch, desc, False
    )

    # setup hf8 / bf8 conversion on stack before GEMM, we need to recheck as we now can update the field in ukernel config, need to use the original GEMM descriptor
    if dt.ab == Datatype.BF8:
        micro_kernel_config.bf8_gemm_via_stack_alloc_tensors = 1

    if dt.ab == Datatype.HF8:
        micro_kernel_config.bf8_gemm_via_stack_alloc_tensors = 1

    # in case when A needs to be transposed, we need to change temporarily the descriptor dimensions for gemm
    if GEMMFlag.TRANS_A in flags:
        assert dt.abc in (Datatype.F32, Datatype.F64)
        lda = m
        flags &= ~GEMMFlag.TRANS_A
        micro_kernel_config.atrans_gemm_stack_alloc_tensors = 1
    elif GEMMFlag.TRANS_B | GEMMFlag.VNNI_B in flags:
        raise NotImplementedError

    # handle A VNNI on stack */
    if avnni_gemm_stack_alloc_tensors:
        lda = m
        micro_kernel_config.avnni_gemm_stack_alloc_tensors = True

    if atvnni_gemm_stack_alloc_tensors:
        lda = m
        micro_kernel_config.atvnni_gemm_stack_alloc_tensors = True

    if avnni_btrans_gemm_stack_alloc_tensors:
        lda = m
        _ldb = k
        micro_kernel_config.avnni_btrans_gemm_stack_alloc_tensors = True

    if atvnni_btrans_gemm_stack_alloc_tensors:
        lda = m
        _ldb = k
        micro_kernel_config.atvnni_btrans_gemm_stack_alloc_tensors = True

    # Block according to the number of available registers or given limits
    max_n_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_max_n_blocking(
        micro_kernel_config, desc, generated_code.arch
    )
    if max_n_blocking > 3:
        init_m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking(
            micro_kernel_config, desc, generated_code.arch, 0
        )
        init_m_blocks = (
            init_m_blocking + micro_kernel_config.vector_length - 1
        ) // micro_kernel_config.vector_length
        is_Ai8_Bf16_gemm = (
            dt.a == Datatype.I8
            and dt.b == Datatype.F16
            and (dt.c in (Datatype.F16, Datatype.F32))
        )

        if is_Ai8_Bf16_gemm:
            raise NotImplementedError
        elif is_Ai8_Bbf16_gemm:
            raise NotImplementedError
        elif is_Ai4_Bi8_gemm:
            print(is_Ai4_Bi8_gemm, desc)
            raise NotImplementedError
        elif is_Ai2_Bi8_gemm:
            raise NotImplementedError
        elif is_Ai1_Bi8_gemm:
            raise NotImplementedError
        else:
            if (
                Arch.LIBXSMM_X86_AVX2_SRF
                <= generated_code.arch
                < Arch.LIBXSMM_X86_AVX512_SKX
            ):
                while (
                    init_m_blocks * max_n_blocking + max_n_blocking + 1
                ) > micro_kernel_config.vector_reg_count:
                    max_n_blocking -= 1
            else:
                while (
                    init_m_blocks * max_n_blocking + init_m_blocks + 1
                ) > micro_kernel_config.vector_reg_count:
                    max_n_blocking -= 1

    assert max_n_blocking

    blocking = libxsmm_compute_equalized_blocking(n, max_n_blocking)
    n_N[0] = blocking.range_1
    n_n[0] = blocking.block_1
    n_N[1] = blocking.range_2
    n_n[1] = blocking.block_2

    # check that l_n_N1 is non-zero
    assert n_N[0]

    # implementing load from struct
    if GEMMFlag.USE_XGEMM_ABI in flags or GEMMFlag.USE_XGEMM_EXT_ABI in flags:
        raise NotImplementedError

    if GEMMFlag.USE_XGEMM_EXT_ABI in flags or micro_kernel_config.vnni_format_C:
        raise NotImplementedError

    # Setting up the stack frame
    libxsmm_generator_gemm_setup_stack_frame(
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
    while n_done != desc.n:
        n_blocking = n_n[n_count]
        m_done = 0
        m_done_old = 0
        m_blocking = 0

        # open N loop
        libxsmm_generator_gemm_header_nloop(
            generated_code,
            loop_label_tracker,
            gp_reg_mapping,
            micro_kernel_config,
            n_done,
            n_blocking,
        )

        if GEMMFlag.DECOMPRESS_A_VIA_BITMASK in desc.flags:
            raise NotImplementedError

        # advance N
        n_done += n_N[n_count]
        n_count += 1

        # define the micro kernel code gen properties, especially m-blocking affects the vector instruction length
        m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking(
            micro_kernel_config, desc, generated_code.arch, 0
        )
        micro_kernel_config.m_bitmask_advance = 0  # @TODO: FOR SSE ONLY and relumask

        # apply m_blocking
        while m_done != desc.m:
            assert m_blocking

            m_done_old = m_done
            micro_kernel_config.current_m = m_done_old
            assert m_blocking

            # coverity[divide_by_zero]
            m_done = m_done + (desc.m - m_done_old) // m_blocking * m_blocking
            micro_kernel_config.m_bitmask_advance += (
                m_done - m_done_old
            ) // 8  # @TODO: FOR SSE ONLY and relumask

            if (m_done != m_done_old) and (m_done > 0):
                # when on AVX512, load mask, if needed
                if (
                    micro_kernel_config.use_masking_a_c
                    and Arch.LIBXSMM_X86_AVX512_VL256_SKX
                    <= generated_code.arch
                    <= Arch.LIBXSMM_X86_ALLFEAT
                ):
                    raise NotImplementedError
                elif (
                    micro_kernel_config.use_masking_a_c
                    and Arch.LIBXSMM_X86_AVX
                    <= generated_code.arch
                    < Arch.LIBXSMM_X86_AVX512_VL256_SKX
                ):
                    raise NotImplementedError

                libxsmm_generator_gemm_header_mloop(
                    generated_code,
                    loop_label_tracker,
                    gp_reg_mapping,
                    micro_kernel_config,
                    m_done_old,
                    m_blocking,
                )
                libxsmm_generator_gemm_load_C(
                    generated_code,
                    gp_reg_mapping,
                    micro_kernel_config,
                    desc,
                    m_blocking,
                    n_blocking,
                )

                if (
                    GEMMFlag.BATCH_REDUCE_ADDRESS in desc.flags
                    or GEMMFlag.BATCH_REDUCE_OFFSET in desc.flags
                    or GEMMFlag.BATCH_REDUCE_STRIDE in desc.flags
                ):
                    raise NotImplementedError

                libxsmm_generator_gemm_sse_avx_avx2_avx512_kloop(
                    generated_code,
                    loop_label_tracker,
                    gp_reg_mapping,
                    micro_kernel_config,
                    desc,
                    m_blocking,
                    n_blocking,
                )

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
                )
                libxsmm_generator_gemm_footer_mloop(
                    generated_code,
                    loop_label_tracker,
                    gp_reg_mapping,
                    micro_kernel_config,
                    desc,
                    m_blocking,
                    m_done,
                )

            # switch to next smaller m_blocking
            m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking(
                micro_kernel_config, desc, generated_code.arch, m_blocking
            )

        libxsmm_generator_gemm_footer_nloop(
            generated_code,
            loop_label_tracker,
            gp_reg_mapping,
            micro_kernel_config,
            desc,
            n_blocking,
            n_done,
        )

    # In this case we vnni-format C from scratch
    if micro_kernel_config.vnni_format_C:
        raise NotImplementedError

    # destroy stack frame
    libxsmm_generator_gemm_destroy_stack_frame(
        generated_code, desc, gp_reg_mapping, micro_kernel_config
    )


def libxsmm_generator_gemm_sse_avx_avx2_avx512_kloop(
    generated_code: GeneratedCode,
    label_tracker: LoopLabelTracker,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    m_blocking: int,
    n_blocking: int,
) -> None:
    return
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
        m_blocking, use_masking_a_c = libxsmm_generator_gemm_get_blocking_and_mask(
            desc.m, 32, 8, m_blocking
        )
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
