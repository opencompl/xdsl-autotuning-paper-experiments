from xdsl.dialects.x86.ops import si32
from xdsl.ir import Block, Region, SSAValue
from xdsl.parser import IntegerAttr
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.dialects import x86_scf
from autotuner.libxsmm_gemm.generator_common import (
    GEMMStackVar,
    GPRegMapping,
    LoopLabelTracker,
    MicroKernelConfig,
)
from autotuner.libxsmm_gemm.generator_gemm_common import (
    libxsmm_generator_gemm_getval_stack_var,
    libxsmm_generator_gemm_setval_stack_var,
)
from autotuner.libxsmm_gemm.generator_x86_instructions import (
    libxsmm_x86_instruction_jump_back_to_label,
    libxsmm_x86_instruction_register_jump_back_label,
)
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode

from xdsl.dialects import x86

from autotuner.libxsmm_gemm.libxsmm_main import (
    GEMMDescriptor,
    GEMMFlag,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


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
        if val.type not in (k_arg_reg,)
    )
    arg_types = tuple(arg.type for arg in args)

    assert k_init_op.next_op is None, (
        "Not sure how this can happen, adding assert to catch later (this assert adding when refactoring to x86_scf generation)"
    )

    body_block = Block(arg_types=(k_arg_reg, *arg_types))

    kloop_op = generated_code.builder.insert(
        x86_scf.ForOp(
            k_init_op.destination,
            IntegerAttr(max_blocked_k, si32),
            IntegerAttr(k_blocking, si32),
            args,
            Region(body_block),
        )
    )

    # Set up builder to build inside of loop
    generated_code.builder.insertion_point = InsertPoint.at_start(body_block)
    curr_vals.clear()
    curr_vals |= {arg.type: arg for arg in body_block.args}

    return kloop_op, body_block.args


def compxsmm_generator_gemm_footer_kloop(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    gemm_desc: GEMMDescriptor,
    m_blocking: int,
    max_blocked_k: int,
    k_loop_complete: bool,
) -> None:
    body_block = generated_code.builder.insertion_point.block
    yielded_arg_types = body_block.arg_types[1:]
    yielded_args = tuple(
        generated_code.current_val_by_reg[val_type] for val_type in yielded_arg_types
    )
    generated_code.insert(yield_op := x86_scf.YieldOp(*yielded_args))

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


def compxsmm_generator_gemm_header_nloop(
    generated_code: GeneratedCode,
    loop_label_tracker: LoopLabelTracker,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    n_init: int,
    n_blocking: int,
) -> None:
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

    args = tuple(generated_code.current_val_by_reg.values())
    arg_types = tuple(arg.type for arg in args)

    assert n_init_op.next_op is None, n_init_op.next_op

    # Insert new block
    body_block = Block(arg_types=arg_types)
    parent_region.insert_block_after(body_block, existing_block)
    loop_label_tracker.dest_blocks.append(body_block)

    # Jump/fallthrough to the newly created block
    # TODO: make sure that we don't print the jump in xDSL if the destination is the
    # next block
    Rewriter.insert_op(
        x86.ops.FallthroughOp(args, body_block),
        InsertPoint.at_end(existing_block),
    )

    generated_code.builder.insertion_point = InsertPoint.at_start(body_block)
    curr_vals.clear()
    curr_vals |= {arg.type: arg for arg in body_block.args}

    libxsmm_x86_instruction_register_jump_back_label(generated_code, loop_label_tracker)
    generated_code.insert(
        x86.ops.RI_AddOp(curr_vals[n_arg_reg], n_blocking, register_out=n_arg_reg)
    )


def compxsmm_generator_gemm_footer_nloop(
    generated_code: GeneratedCode,
    loop_label_tracker: LoopLabelTracker,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    n_blocking: int,
    n_done: int,
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
        output_ptr = libxsmm_generator_gemm_getval_stack_var(
            generated_code,
            micro_kernel_config,
            GEMMStackVar.ELT_OUTPUT_PTR,
            gp_reg_mapping.gp_reg_help_0,
        )
        output_ptr = generated_code.insert(
            x86.ops.RI_AddOp(output_ptr, (n_blocking * (desc.ldc) * 2) - ((desc.m) * 2))
        ).register_out
        libxsmm_generator_gemm_setval_stack_var(
            generated_code,
            micro_kernel_config,
            GEMMStackVar.ELT_OUTPUT_PTR,
            output_ptr,
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

    generated_code.insert(
        x86.ops.SI_CmpOp(
            generated_code.current_val_by_reg[gp_reg_mapping.gp_reg_nloop],
            n_done,
        )
    )

    libxsmm_x86_instruction_jump_back_to_label(
        generated_code, x86.ops.C_JlOp, loop_label_tracker
    )


def compxsmm_generator_gemm_header_mloop(
    generated_code: GeneratedCode,
    loop_label_tracker: LoopLabelTracker,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    m_init: int,
    m_blocking: int,
) -> None:
    """
    In original, adds three lines of assembly: set counter to m_init, add label, add m_blocking to loop counter.
    We create the same ops, but also create a block to hold the body of the loop, and set the insertion point at end of the new block.
    """
    m_arg_reg = gp_reg_mapping.gp_reg_mloop

    curr_vals = generated_code.current_val_by_reg
    generated_code.insert(m_init_op := x86.ops.DI_MovOp(m_init, destination=m_arg_reg))

    existing_block = m_init_op.parent
    assert existing_block is not None
    parent_region = existing_block.parent
    assert parent_region is not None

    args = tuple(generated_code.current_val_by_reg.values())
    arg_types = tuple(arg.type for arg in args)

    assert m_init_op.next_op is None

    # Insert new block
    body_block = Block(arg_types=arg_types)
    parent_region.insert_block_after(body_block, existing_block)
    loop_label_tracker.dest_blocks.append(body_block)

    # Jump/fallthrough to the newly created block
    # TODO: make sure that we don't print the jump in xDSL if the destination is the
    # next block
    Rewriter.insert_op(
        x86.ops.FallthroughOp(args, body_block),
        InsertPoint.at_end(existing_block),
    )

    generated_code.builder.insertion_point = InsertPoint.at_start(body_block)
    curr_vals.clear()
    curr_vals |= {arg.type: arg for arg in body_block.args}

    libxsmm_x86_instruction_register_jump_back_label(generated_code, loop_label_tracker)
    generated_code.insert(
        x86.ops.RI_AddOp(curr_vals[m_arg_reg], m_blocking, register_out=m_arg_reg)
    )


def compxsmm_generator_gemm_footer_mloop(
    generated_code: GeneratedCode,
    loop_label_tracker: LoopLabelTracker,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    m_blocking: int,
    m_done: int,
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

    # loop handling

    generated_code.insert(
        x86.ops.SI_CmpOp(
            generated_code.current_val_by_reg[gp_reg_mapping.gp_reg_mloop],
            m_done,
        )
    )

    libxsmm_x86_instruction_jump_back_to_label(
        generated_code, x86.ops.C_JlOp, loop_label_tracker
    )
