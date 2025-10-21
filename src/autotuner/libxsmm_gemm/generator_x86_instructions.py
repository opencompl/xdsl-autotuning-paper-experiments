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


def libxsmm_x86_instruction_register_jump_back_label(
    generated_code: GeneratedCode, loop_label_tracker: LoopLabelTracker
) -> None:
    new_label = len(loop_label_tracker.label_address) + 32 + 1
    loop_label_tracker.label_address.append(new_label)
    generated_code.builder.insert(x86.ops.LabelOp(f"{new_label}"))
