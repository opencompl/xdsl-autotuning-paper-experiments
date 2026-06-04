from xdsl.dialects import x86
from xdsl.dialects.x86.registers import GeneralRegisterType
from xdsl.ir import SSAValue
from autotuner.compxsmm_gemm.generator_x86_instructions import (
    compxsmm_x86_instruction_mask_move_ld,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.compxsmm_gemm.libxsmm_generator import GeneratedCode
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


def compxsmm_generator_initialize_avx512_mask(
    generated_code: GeneratedCode,
    gp_reg_tmp: GeneralRegisterType,
    mask_reg: x86.registers.AVX512MaskRegisterType,
    mask_count: int,
    datatype: Datatype,
) -> SSAValue[x86.registers.AVX512MaskRegisterType]:
    mask = 0

    if (generated_code.arch >= Arch.LIBXSMM_X86_AVX512_SKX) and (
        generated_code.arch <= Arch.LIBXSMM_X86_ALLFEAT
    ):
        if datatype == Datatype.F64 or datatype == Datatype.I64:
            mask = 0xFF
        elif datatype == Datatype.F32 or datatype == Datatype.I32:
            mask = 0xFFFF
        elif (
            datatype == Datatype.F16
            or datatype == Datatype.BF16
            or datatype == Datatype.I16
        ):
            mask = 0xFFFFFFFF
        elif (
            datatype == Datatype.I8
            or datatype == Datatype.BF8
            or datatype == Datatype.HF8
        ):
            mask = 0xFFFFFFFFFFFFFFFF
        else:
            assert False, f"Unsupported datatype {datatype}"

    elif (generated_code.arch >= Arch.LIBXSMM_X86_AVX512_VL256_SKX) and (
        generated_code.arch < Arch.LIBXSMM_X86_AVX512_SKX
    ):
        if datatype == Datatype.F64 or datatype == Datatype.I64:
            mask = 0xF
        elif datatype == Datatype.F32 or datatype == Datatype.I32:
            mask = 0xFF
        elif (
            datatype == Datatype.F16
            or datatype == Datatype.BF16
            or datatype == Datatype.I16
        ):
            mask = 0xFFFF
        elif (
            datatype == Datatype.I8
            or datatype == Datatype.BF8
            or datatype == Datatype.HF8
        ):
            mask = 0xFFFFFFFF
        else:
            assert False, f"Unsupported datatype {datatype}"
    else:
        # should not happen
        assert False, f"Unsupported arch {generated_code.arch}"

    # shift right by "inverse" remainder
    mask = mask >> mask_count

    # /* move mask to GP register */

    mask_tmp_val = generated_code.builder.insert(
        x86.ops.DI_MovOp(mask, destination=gp_reg_tmp)
    ).destination

    # loading the mask register
    if datatype == Datatype.F64 or datatype == Datatype.I64:
        return compxsmm_x86_instruction_mask_move_ld(
            generated_code,
            x86.ops.KS_KMovBOp
            if generated_code.arch >= Arch.LIBXSMM_X86_AVX512_SKX
            else x86.ops.KS_KMovWOp,
            mask_tmp_val,
            mask_reg,
        )
    elif datatype == Datatype.F32 or datatype == Datatype.I32:
        return compxsmm_x86_instruction_mask_move_ld(
            generated_code,
            x86.ops.KS_KMovWOp,
            mask_tmp_val,
            mask_reg,
        )
    elif (
        datatype == Datatype.F16
        or datatype == Datatype.BF16
        or datatype == Datatype.I16
    ):
        raise NotImplementedError
    elif (
        datatype == Datatype.I8 or datatype == Datatype.BF8 or datatype == Datatype.HF8
    ):
        raise NotImplementedError
    else:
        assert False, f"Unsupported datatype {datatype}"
