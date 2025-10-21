from xdsl.dialects import x86
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
    assert loop_label_tracker.label_address

    label_index = loop_label_tracker.label_address.pop()
    label_str = f"{label_index}"
    dest_blocks = [
        block
        for block in generated_code.func_op.body.blocks
        if isinstance(block.first_op, x86.ops.LabelOp)
        and block.first_op.label.data == label_str
    ]
    assert len(dest_blocks) == 1
    dest_block = dest_blocks.pop()

    builder = generated_code.builder

    curr_block = builder.insertion_point.block
    fallthrough_block = curr_block.next_block
    assert fallthrough_block is not None

    assert (cmp_op := curr_block.last_op) is not None
    assert len(cmp_op.results) == 1

    # TODO: handle block args
    builder.insert(jmp_instr(cmp_op, (), (), dest_block, fallthrough_block))


def libxsmm_x86_instruction_register_jump_back_label(
    generated_code: GeneratedCode, loop_label_tracker: LoopLabelTracker
) -> None:
    new_label = len(loop_label_tracker.label_address) + 32 + 1
    loop_label_tracker.label_address.append(new_label)
    generated_code.builder.insert(x86.ops.LabelOp(f"{new_label}"))
