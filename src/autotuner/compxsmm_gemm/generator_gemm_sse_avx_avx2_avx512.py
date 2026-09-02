import os

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
    GeneralRegisterType,
)
from xdsl.dialects.x86_func import FuncOp
from xdsl.ir import SSAValue
from xdsl.rewriter import InsertPoint

from autotuner.dialects.xsmm import MatmulIterator, MatmulOp
from autotuner.libxsmm_gemm.generator_common import (
    GPRegMapping,
    MicroKernelConfig,
)
from autotuner.libxsmm_gemm.generator_gemm_common import (
    libxsmm_generator_gemm_destroy_stack_frame,
    libxsmm_generator_gemm_init_micro_kernel_config,
    libxsmm_generator_gemm_setup_stack_frame,
)
from autotuner.libxsmm_gemm.generator_x86_instructions import (
    libxsmm_x86_instruction_open_stream_gemm,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_generator import (
    GeneratedCode,
)
from autotuner.libxsmm_gemm.libxsmm_main import (
    GEMMDescriptor,
    GEMMFlag,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


def compxsmm_generator_gemm_sse_avx_avx2_avx512_kernel_wrapper(
    func_op: FuncOp, arch: Arch, desc: GEMMDescriptor, *, disable_regalloc: bool
) -> None:
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

    if not disable_regalloc:
        gp_reg_mapping.gp_reg_mloop = R10
        gp_reg_mapping.gp_reg_nloop = R11
        gp_reg_mapping.gp_reg_kloop = R12
        gp_reg_mapping.gp_reg_help_0 = R14
        gp_reg_mapping.gp_reg_help_1 = R15
        gp_reg_mapping.gp_reg_help_2 = RBX

    builder = Builder(InsertPoint.at_end(func_op.body.block))

    generated_code = GeneratedCode(builder, arch)

    # Respect SWAP_A_B
    arg_by_reg = {arg.type: arg for arg in func_op.body.block.args}
    a_val = SSAValue.get(arg_by_reg[gp_reg_mapping.gp_reg_a], type=GeneralRegisterType)
    b_val = SSAValue.get(arg_by_reg[gp_reg_mapping.gp_reg_b], type=GeneralRegisterType)
    c_val = SSAValue.get(arg_by_reg[gp_reg_mapping.gp_reg_c], type=GeneralRegisterType)

    libxsmm_x86_instruction_open_stream_gemm(
        generated_code, gp_reg_mapping, False, desc.prefetch
    )
    compxsmm_generator_gemm_sse_avx_avx2_avx512_kernel(
        generated_code,
        gp_reg_mapping,
        desc,
        disable_regalloc=disable_regalloc,
        a_val=a_val,
        b_val=b_val,
        c_val=c_val,
    )

    # In C, the stream is closed with the inline assembly register string, but we don't
    # Need to do this
    # libxsmm_x86_instruction_close_stream_gemm(
    #     generated_code, gp_reg_mapping, False, desc.prefetch
    # )


def compxsmm_generator_gemm_sse_avx_avx2_avx512_kernel(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    desc: GEMMDescriptor,
    *,
    disable_regalloc: bool,
    a_val: SSAValue[GeneralRegisterType],
    b_val: SSAValue[GeneralRegisterType],
    c_val: SSAValue[GeneralRegisterType],
) -> None:
    micro_kernel_config = MicroKernelConfig()

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

    if GEMMFlag.DECOMPRESS_A_VIA_BITMASK in desc.flags:
        raise NotImplementedError
    datatype = desc.datatype.ab
    assert datatype is not None
    if datatype not in (Datatype.F32, Datatype.F64):
        raise NotImplementedError
    if desc.prefetch == GEMMPrefetchType.AL2:
        raise NotImplementedError
    matmul_n = generated_code.insert(
        MatmulOp(
            a_val,
            b_val,
            c_val,
            rbp_val,
            rsp_val,
            m=desc.m,
            n=desc.n,
            k=desc.k,
            lda=desc.lda,
            ldb=desc.ldb,
            ldc=desc.ldc,
            datatype=datatype.builtin_type,
            aligned_a=GEMMFlag.ALIGN_A in desc.flags,
            aligned_c=GEMMFlag.ALIGN_C in desc.flags,
            iterator=MatmulIterator.N,
        )
    )
    a_val = matmul_n.a_out
    b_val = matmul_n.b_out
    c_val = matmul_n.c_out
    rbp_val = matmul_n.rbp_out
    rsp_val = matmul_n.rsp_out

    # In this case we vnni-format C from scratch
    if micro_kernel_config.vnni_format_C:
        raise NotImplementedError

    # destroy stack frame
    libxsmm_generator_gemm_destroy_stack_frame(
        generated_code, desc, gp_reg_mapping, micro_kernel_config, rbp_val
    )
