from xdsl.ir import Block
from xdsl.rewriter import InsertPoint, Rewriter
from autotuner.libxsmm_gemm.generator_common import (
    GPRegMapping,
    LoopLabelTracker,
    MicroKernelConfig,
)
from autotuner.libxsmm_gemm.generator_x86_instructions import (
    libxsmm_x86_instruction_register_jump_back_label,
)
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode

from xdsl.dialects import x86


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
    generated_code.builder.insert(
        k_init_op := x86.ops.DI_MovOp(0, destination=k_arg_reg)
    )

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

    generated_code.builder.insertion_point = InsertPoint.at_end(new_block)

    libxsmm_x86_instruction_register_jump_back_label(generated_code, loop_label_tracker)
    generated_code.builder.insert(
        x86.ops.RI_AddOp(new_block.args[0], 4, register_out=k_arg_reg)
    )
