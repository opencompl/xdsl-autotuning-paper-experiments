from xdsl.dialects import x86
from xdsl.rewriter import InsertPoint
from autotuner.libxsmm_gemm.generator_common import GPRegMapping, LoopLabelTracker
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode
from autotuner.libxsmm_gemm.libxsmm_main import GEMMPrefetchType


def libxsmm_x86_instruction_open_stream_gemm(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    skip_callee_save: bool,
    prefetch: GEMMPrefetchType,
) -> None:
    args = generated_code.func_op.body.block.args
    a, b, c = args

    builder = generated_code.builder

    builder.insert(x86.ops.DS_MovOp(a, destination=gp_reg_mapping.gp_reg_a))
    builder.insert(x86.ops.DS_MovOp(b, destination=gp_reg_mapping.gp_reg_b))
    builder.insert(x86.ops.DS_MovOp(c, destination=gp_reg_mapping.gp_reg_c))

    match prefetch:
        case GEMMPrefetchType.BL2 | GEMMPrefetchType.AL2:
            raise NotImplementedError


def libxsmm_x86_instruction_jump_back_to_label(
    generated_code: GeneratedCode,
    jmp_instr: type[x86.ops.ConditionalJumpOperation],
    loop_label_tracker: LoopLabelTracker,
):
    """
    In contrast to libxsmm, also inserts the comparison instruction
    """
    dest_block = loop_label_tracker.dest_blocks.pop()
    builder = generated_code.builder
    curr_vals = generated_code.current_val_by_reg

    curr_args = tuple(curr_vals[arg.type] for arg in dest_block.args)

    curr_block = builder.insertion_point.block
    fallthrough_block = curr_block.next_block
    assert fallthrough_block is not None

    assert (cmp_op := curr_block.last_op) is not None
    assert len(cmp_op.results) == 1

    builder.insert(
        jmp_instr(cmp_op, curr_args, curr_args, dest_block, fallthrough_block)
    )

    # set insert point to fallthrough block and update current values
    builder.insertion_point = InsertPoint.at_start(fallthrough_block)
    curr_vals.clear()
    curr_vals |= {arg.type: arg for arg in fallthrough_block.args}


def libxsmm_x86_instruction_register_jump_back_label(
    generated_code: GeneratedCode, loop_label_tracker: LoopLabelTracker
) -> None:
    generated_code.builder.insert(
        x86.ops.LabelOp(f"{loop_label_tracker.current_loop_number}")
    )
