import os

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
)
from xdsl.dialects.x86_func import FuncOp
from xdsl.ir import SSAValue
from xdsl.rewriter import InsertPoint
from autotuner.compxsmm_gemm.generator_gemm_common import (
    compxsmm_generator_gemm_header_mloop,
    compxsmm_generator_gemm_footer_kloop,
    compxsmm_generator_gemm_footer_mloop,
    compxsmm_generator_gemm_footer_nloop,
    compxsmm_generator_gemm_header_nloop,
    compxsmm_generator_gemm_kloop,
)
from autotuner.libxsmm_gemm.generator_common import (
    LIBXSMM_X86_AVX512_MASK_REG,
    GPRegMapping,
    LoopLabelTracker,
    MicroKernelConfig,
    libxsmm_compute_equalized_blocking,
)
from autotuner.compxsmm_gemm.generator_common_x86 import (
    compxsmm_generator_initialize_avx512_mask,
)
from autotuner.compxsmm_gemm.generator_gemm_avx512_microkernel import (
    compxsmm_generator_gemm_avx512_kloop_kernel,
)
from autotuner.compxsmm_gemm.generator_gemm_common import (
    compxsmm_generator_gemm_destroy_stack_frame,
    compxsmm_generator_gemm_init_micro_kernel_config,
    compxsmm_generator_gemm_load_C,
    compxsmm_generator_gemm_setup_stack_frame,
    compxsmm_generator_gemm_store_C,
)
from autotuner.libxsmm_gemm.generator_gemm_sse_avx_avx2_avx512 import (
    libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking,
    libxsmm_generator_gemm_sse_avx_avx2_avx512_get_max_n_blocking,
)
from autotuner.compxsmm_gemm.generator_x86_instructions import (
    compxsmm_x86_instruction_open_stream_gemm,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.compxsmm_gemm.libxsmm_generator import (
    GeneratedCode,
    KLoopVals,
    MLoopVals,
    NLoopVals,
)
from autotuner.libxsmm_gemm.libxsmm_main import (
    DescDatatype,
    GEMMDescriptor,
    GEMMFlag,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


def compxsmm_generator_gemm_sse_avx_avx2_avx512_kernel_wrapper(
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

    builder = Builder(InsertPoint.at_end(func_op.body.block))
    generated_code = GeneratedCode(
        func_op, builder, arch, {arg.type: arg for arg in func_op.body.block.args}
    )

    compxsmm_x86_instruction_open_stream_gemm(
        generated_code, gp_reg_mapping, False, desc.prefetch
    )
    compxsmm_generator_gemm_sse_avx_avx2_avx512_kernel(
        generated_code, loop_label_tracker, gp_reg_mapping, desc
    )

    # In C, the stream is closed with the inline assembly register string, but we don't
    # Need to do this
    # libxsmm_x86_instruction_close_stream_gemm(
    #     generated_code, gp_reg_mapping, False, desc.prefetch
    # )


def compxsmm_generator_gemm_sse_avx_avx2_avx512_kernel(
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
    compxsmm_generator_gemm_init_micro_kernel_config(
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
    rbp_val, rsp = compxsmm_generator_gemm_setup_stack_frame(
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
        nloop_op, nloop_block_vals = compxsmm_generator_gemm_header_nloop(
            generated_code,
            gp_reg_mapping,
            micro_kernel_config,
            n_init=n_done,
            n_blocking=n_blocking,
            n_done=n_done + n_N[n_count],
        )

        a_val = nloop_block_vals.a
        b_val = nloop_block_vals.b
        c_val = nloop_block_vals.c
        rbp_val = nloop_block_vals.rbp
        rsp_val = nloop_block_vals.rsp

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
                        compxsmm_generator_initialize_avx512_mask(
                            generated_code,
                            gp_reg_mapping.gp_reg_help_1,
                            x86.registers.AVX512MaskRegisterType.from_index(2),
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
                            compxsmm_generator_initialize_avx512_mask(
                                generated_code,
                                gp_reg_mapping.gp_reg_help_1,
                                x86.registers.AVX512MaskRegisterType.from_index(2),
                                0
                                if m_blocking % corrected_vlen >= c_vlen_adjusted
                                else c_vlen_adjusted - (m_blocking % c_vlen_adjusted),
                                Datatype.I32,
                            )
                            compxsmm_generator_initialize_avx512_mask(
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

                        compxsmm_generator_initialize_avx512_mask(
                            generated_code,
                            gp_reg_mapping.gp_reg_help_1,
                            x86.registers.AVX512MaskRegisterType.from_index(2),
                            mask_count,
                            desc.datatype.c,
                        )
                    else:
                        mask_k1_val = compxsmm_generator_initialize_avx512_mask(
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

                mloop_op, mloop_block_vals = compxsmm_generator_gemm_header_mloop(
                    generated_code,
                    gp_reg_mapping,
                    micro_kernel_config,
                    args=MLoopVals(
                        a_val,
                        b_val,
                        c_val,
                        rbp_val,
                        rsp_val,
                        mask_k1_val,
                    ),
                    m_init=m_done_old,
                    m_blocking=m_blocking,
                    m_done=m_done,
                )
                compxsmm_generator_gemm_load_C(
                    generated_code,
                    gp_reg_mapping,
                    micro_kernel_config,
                    desc,
                    m_blocking,
                    n_blocking,
                    c_val=mloop_block_vals.c,
                    mask_k1=mloop_block_vals.mask_k1,
                )

                if (
                    GEMMFlag.BATCH_REDUCE_ADDRESS in desc.flags
                    or GEMMFlag.BATCH_REDUCE_OFFSET in desc.flags
                    or GEMMFlag.BATCH_REDUCE_STRIDE in desc.flags
                ):
                    raise NotImplementedError

                m_blocking_vec = (
                    m_blocking // micro_kernel_config.vector_length
                    if (m_blocking % micro_kernel_config.vector_length == 0)
                    else (m_blocking // micro_kernel_config.vector_length) + 1
                )
                vec_reg_acc_start = (
                    micro_kernel_config.vector_reg_count - n_blocking * m_blocking_vec
                )
                match micro_kernel_config.vector_name:
                    case "x":
                        dest_type = x86.registers.SSERegisterType
                    case "y":
                        dest_type = x86.registers.AVX2RegisterType
                    case "z":
                        dest_type = x86.registers.AVX512RegisterType
                acc_vals = tuple(
                    generated_code.get_val(
                        dest_type.from_index(
                            vec_reg_acc_start + m + (m_blocking_vec * n)
                        )
                    )
                    for n in range(n_blocking)
                    for m in range(m_blocking_vec)
                )
                kloop_vals = KLoopVals(
                    mloop_block_vals.a,
                    mloop_block_vals.b,
                    mloop_block_vals.c,
                    mloop_block_vals.rbp,
                    mloop_block_vals.rsp,
                    mloop_block_vals.mask_k1,
                    acc_vals,
                )

                kloop_vals = compxsmm_generator_gemm_sse_avx_avx2_avx512_kloop(
                    generated_code,
                    loop_label_tracker,
                    gp_reg_mapping,
                    micro_kernel_config,
                    desc,
                    m_blocking,
                    n_blocking,
                    kloop_vals=kloop_vals,
                )

                if (
                    GEMMFlag.BATCH_REDUCE_ADDRESS in desc.flags
                    or GEMMFlag.BATCH_REDUCE_OFFSET in desc.flags
                    or GEMMFlag.BATCH_REDUCE_STRIDE in desc.flags
                ):
                    raise NotImplementedError

                a_val = kloop_vals.a
                b_val = kloop_vals.b
                c_val = kloop_vals.c
                rbp_val = kloop_vals.rbp
                rsp_val = kloop_vals.rsp

                compxsmm_generator_gemm_store_C(
                    generated_code,
                    gp_reg_mapping,
                    micro_kernel_config,
                    desc,
                    m_blocking,
                    n_blocking,
                    c_val=generated_code.get_val(gp_reg_mapping.gp_reg_c),
                    mask_k1=(
                        generated_code.get_val(LIBXSMM_X86_AVX512_MASK_REG)
                        if mloop_block_vals.mask_k1 is not None
                        else None
                    ),
                )

                a_val = generated_code.get_val(gp_reg_mapping.gp_reg_a)
                b_val = generated_code.get_val(gp_reg_mapping.gp_reg_b)
                c_val = generated_code.get_val(gp_reg_mapping.gp_reg_c)
                rbp_val = generated_code.get_val(x86.registers.RBP)
                rsp_val = generated_code.get_val(x86.registers.RSP)

                mloop_vals = compxsmm_generator_gemm_footer_mloop(
                    generated_code,
                    gp_reg_mapping,
                    micro_kernel_config,
                    desc,
                    m_blocking=m_blocking,
                    vals=MLoopVals(
                        a_val,
                        b_val,
                        c_val,
                        rbp_val,
                        rsp_val,
                        generated_code.get_val(LIBXSMM_X86_AVX512_MASK_REG)
                        if mloop_block_vals.mask_k1 is not None
                        else None,
                    ),
                )

                a_val = mloop_vals.a
                b_val = mloop_vals.b
                c_val = mloop_vals.c
                rbp_val = mloop_vals.rbp
                rsp_val = mloop_vals.rsp
                mask_k1_val = mloop_vals.mask_k1

            # switch to next smaller m_blocking
            m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking(
                micro_kernel_config, desc, generated_code.arch, m_blocking
            )

        curr_vals = generated_code.current_val_by_reg

        a_val, b_val, c_val, rbp_val, rsp_val, *mask_vals = curr_vals.values()
        a_val = SSAValue.get(a_val, type=x86.registers.GeneralRegisterType)
        b_val = SSAValue.get(b_val, type=x86.registers.GeneralRegisterType)
        c_val = SSAValue.get(c_val, type=x86.registers.GeneralRegisterType)
        rbp_val = SSAValue.get(rbp_val, type=x86.registers.GeneralRegisterType)
        assert rbp_val.type == x86.registers.RBP
        rsp_val = SSAValue.get(rsp_val, type=x86.registers.GeneralRegisterType)
        assert rsp_val.type == x86.registers.RSP

        nloop_inner_vals = NLoopVals(a_val, b_val, c_val, rbp_val, rsp_val)
        nloop_result_vals = compxsmm_generator_gemm_footer_nloop(
            generated_code,
            gp_reg_mapping,
            micro_kernel_config,
            desc,
            n_blocking=n_blocking,
            vals=nloop_inner_vals,
        )
        rbp_val = nloop_result_vals.rbp

    # In this case we vnni-format C from scratch
    if micro_kernel_config.vnni_format_C:
        raise NotImplementedError

    # destroy stack frame
    compxsmm_generator_gemm_destroy_stack_frame(
        generated_code, desc, gp_reg_mapping, micro_kernel_config, rbp_val
    )


def compxsmm_generator_gemm_sse_avx_avx2_avx512_kloop(
    generated_code: GeneratedCode,
    label_tracker: LoopLabelTracker,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    m_blocking: int,
    n_blocking: int,
    *,
    kloop_vals: KLoopVals,
) -> KLoopVals:
    # some hard coded parameters for k-blocking
    k_blocking = 0
    k_threshold = 0
    _k_pack_factor = 1
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
        generator_kloop_kernel = compxsmm_generator_gemm_avx512_kloop_kernel
    else:
        assert False, (
            f"Unsupported architecture {generated_code.arch} for micro-kernel generation"
        )

    if is_Ai4_Bi8_gemm and GEMMFlag.USE_MxK_ZPT in desc.flags:
        raise NotImplementedError

    if is_Ai8_Bbf16_gemm and is_Ai8_Bbf16_gemm_bf16fma:
        raise NotImplementedError

    # Apply multiple k_blocking strategies
    m_blocking_vec = (
        m_blocking // micro_kernel_config.vector_length
        if (m_blocking % micro_kernel_config.vector_length == 0)
        else (m_blocking // micro_kernel_config.vector_length) + 1
    )
    vec_reg_acc_start = (
        micro_kernel_config.vector_reg_count - n_blocking * m_blocking_vec
    )
    match micro_kernel_config.vector_name:
        case "x":
            dest_type = x86.registers.SSERegisterType
        case "y":
            dest_type = x86.registers.AVX2RegisterType
        case "z":
            dest_type = x86.registers.AVX512RegisterType
    acc_vals = tuple(
        generated_code.get_val(
            dest_type.from_index(vec_reg_acc_start + m + (m_blocking_vec * n))
        )
        for n in range(n_blocking)
        for m in range(m_blocking_vec)
    )
    kloop_vals = KLoopVals(
        kloop_vals.a,
        kloop_vals.b,
        kloop_vals.c,
        kloop_vals.rbp,
        kloop_vals.rsp,
        kloop_vals.mask_k1,
        acc_vals,
    )

    if not desc.k % k_blocking and k_threshold < desc.k:
        # 1. we are larger the k_threshold and a multiple of a predefined blocking parameter
        kloop_op, kloop_block_vals = compxsmm_generator_gemm_kloop(
            generated_code,
            gp_reg_mapping,
            vals=kloop_vals,
            k_blocking=k_blocking,
            max_blocked_k=desc.k,
        )
        kloop_yielded_vals = generator_kloop_kernel(
            generated_code,
            gp_reg_mapping,
            micro_kernel_config,
            desc,
            m_blocking,
            n_blocking,
            k_blocking,
            kloop_block_vals,
        )

        kloop_vals = compxsmm_generator_gemm_footer_kloop(
            generated_code,
            gp_reg_mapping,
            micro_kernel_config,
            desc,
            m_blocking,
            True,
            kloop_yielded_vals,
        )
    else:
        b_offset = 0
        # 2. we want to fully unroll below the threshold
        if desc.k <= k_threshold:
            kloop_vals = generator_kloop_kernel(
                generated_code,
                gp_reg_mapping,
                micro_kernel_config,
                desc,
                m_blocking,
                n_blocking,
                desc.k,
                kloop_vals,
            )
        # 3. we are larger than the threshold but not a multiple of the blocking factor -> largest possible blocking + remainder handling
        else:
            # Largest possible blocking
            l_max_blocked_k = (desc.k // k_blocking) * k_blocking

            # We can block as k is large enough
            if l_max_blocked_k > 0:
                kloop_op, kloop_block_vals = compxsmm_generator_gemm_kloop(
                    generated_code,
                    gp_reg_mapping,
                    vals=kloop_vals,
                    k_blocking=k_blocking,
                    max_blocked_k=l_max_blocked_k,
                )

                kloop_yielded_vals = generator_kloop_kernel(
                    generated_code,
                    gp_reg_mapping,
                    micro_kernel_config,
                    desc,
                    m_blocking,
                    n_blocking,
                    k_blocking,
                    kloop_block_vals,
                )

                kloop_vals = compxsmm_generator_gemm_footer_kloop(
                    generated_code,
                    gp_reg_mapping,
                    micro_kernel_config,
                    desc,
                    m_blocking=m_blocking,
                    k_loop_complete=False,
                    kloop_yielded_vals=kloop_yielded_vals,
                )

            # Now handle the remainder
            kloop_vals = generator_kloop_kernel(
                generated_code,
                gp_reg_mapping,
                micro_kernel_config,
                desc,
                m_blocking,
                n_blocking,
                desc.k - l_max_blocked_k,
                kloop_vals,
            )

            # Reset B pointer
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
