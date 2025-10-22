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
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode

from xdsl.dialects import x86

from autotuner.libxsmm_gemm.libxsmm_main import GEMMDescriptor, GemmFlag


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
    builder = generated_code.builder
    curr_vals = generated_code.current_val_by_reg
    builder.insert(k_init_op := x86.ops.DI_MovOp(0, destination=k_arg_reg))
    curr_vals[k_arg_reg] = k_init_op.destination

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

    builder.insertion_point = InsertPoint.at_start(new_block)
    curr_vals.clear()
    curr_vals |= {arg.type: arg for arg in new_block.args}

    libxsmm_x86_instruction_register_jump_back_label(generated_code, loop_label_tracker)
    add_op = builder.insert(
        x86.ops.RI_AddOp(curr_vals[k_arg_reg], 4, register_out=k_arg_reg)
    )
    curr_vals[k_arg_reg] = add_op.register_out


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
    generated_code.builder.insert(
        x86.ops.SI_CmpOp(
            generated_code.current_val_by_reg[gp_reg_mapping.gp_reg_kloop],
            max_blocked_k,
            result=x86.registers.RFLAGS,
        )
    )

    libxsmm_x86_instruction_jump_back_to_label(
        generated_code, x86.ops.C_JlOp, loop_label_tracker
    )
    if k_loop_complete:
        b_offset = 0
        if GemmFlag.TRANS_B in gemm_desc.flags:
            b_offset = (
                gemm_desc.ldb * gemm_desc.k * micro_kernel_config.datatype_size_in2
            )
        else:
            b_offset = gemm_desc.ldb * micro_kernel_config.datatype_size_in2

        sub_op = generated_code.builder.insert(
            x86.ops.RI_SubOp(
                generated_code.current_val_by_reg[gp_reg_mapping.gp_reg_b],
                b_offset,
                register_out=gp_reg_mapping.gp_reg_b,
            )
        )
        generated_code.current_val_by_reg[gp_reg_mapping.gp_reg_b] = sub_op.register_out
