from dataclasses import dataclass

from xdsl.dialects import x86
from xdsl.dialects.builtin import UnitAttr
from xdsl.dialects.x86.registers import GeneralRegisterType
from xdsl.ir import SSAValue

from autotuner.dialects import x86_ext
from autotuner.libxsmm_gemm.generator_x86_instructions import (
    libxsmm_x86_instruction_full_vec_load_of_constants,
    libxsmm_x86_instruction_mask_move_ld,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


def libxsmm_generator_initialize_avx512_mask(
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
        return libxsmm_x86_instruction_mask_move_ld(
            generated_code,
            x86.ops.KS_KMovBOp
            if generated_code.arch >= Arch.LIBXSMM_X86_AVX512_SKX
            else x86.ops.KS_KMovWOp,
            mask_tmp_val,
            mask_reg,
        )
    elif datatype == Datatype.F32 or datatype == Datatype.I32:
        return libxsmm_x86_instruction_mask_move_ld(
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


@dataclass(frozen=True)
class Exp3Coefficients:
    c0: SSAValue[x86.registers.AVX512RegisterType]
    c1: SSAValue[x86.registers.AVX512RegisterType]
    c2: SSAValue[x86.registers.AVX512RegisterType]
    c3: SSAValue[x86.registers.AVX512RegisterType]
    halves: SSAValue[x86.registers.AVX512RegisterType]
    log2e: SSAValue[x86.registers.AVX512RegisterType]


@dataclass(frozen=True)
class Exp5Coefficients(Exp3Coefficients):
    c4: SSAValue[x86.registers.AVX512RegisterType]
    c5: SSAValue[x86.registers.AVX512RegisterType]
    ln2: SSAValue[x86.registers.AVX512RegisterType]
    expmask: SSAValue[x86.registers.AVX512RegisterType]
    logfmax: SSAValue[x86.registers.AVX512RegisterType]
    logfmin: SSAValue[x86.registers.AVX512RegisterType]


def libxsmm_generator_prepare_coeffs_exp_ps_3dts_avx512(
    code: GeneratedCode,
    vec_c0: x86.registers.AVX512RegisterType,
    vec_c1: x86.registers.AVX512RegisterType,
    vec_c2: x86.registers.AVX512RegisterType,
    vec_c3: x86.registers.AVX512RegisterType,
    vec_halves: x86.registers.AVX512RegisterType,
    vec_log2e: x86.registers.AVX512RegisterType,
) -> Exp3Coefficients:
    load = libxsmm_x86_instruction_full_vec_load_of_constants
    halves = load(code, 0x3F000000, "halves", vec_halves)
    log2e = load(code, 0x3FB8AA3B, "log2e", vec_log2e)
    # Exact FP32 representations of upstream's decimal float literals.
    c0 = load(code, 0x3F34E022, "c0", vec_c0)
    c1 = load(code, 0x3EFD357F, "c1", vec_c1)
    c2 = load(code, 0x3E20BCD5, "c2", vec_c2)
    c3 = load(code, 0x3D635847, "c3", vec_c3)
    return Exp3Coefficients(c0, c1, c2, c3, halves, log2e)


def libxsmm_generator_prepare_coeffs_exp_ps_5dts_avx512(
    code: GeneratedCode,
    vec_c0: x86.registers.AVX512RegisterType,
    vec_c1: x86.registers.AVX512RegisterType,
    vec_c2: x86.registers.AVX512RegisterType,
    vec_c3: x86.registers.AVX512RegisterType,
    vec_c4: x86.registers.AVX512RegisterType,
    vec_c5: x86.registers.AVX512RegisterType,
    vec_halves: x86.registers.AVX512RegisterType,
    vec_log2e: x86.registers.AVX512RegisterType,
    vec_ln2: x86.registers.AVX512RegisterType,
    vec_expmask: x86.registers.AVX512RegisterType,
    vec_logfmax: x86.registers.AVX512RegisterType,
    vec_logfmin: x86.registers.AVX512RegisterType,
) -> Exp5Coefficients:
    load = libxsmm_x86_instruction_full_vec_load_of_constants
    expmask = load(code, 0x0000007F, "expmask", vec_expmask)
    logfmax = load(code, 0x42B17218, "logfmax", vec_logfmax)
    logfmin = load(code, 0xC2AEAC50, "logfmin", vec_logfmin)
    halves = load(code, 0x3F000000, "halves", vec_halves)
    log2e = load(code, 0x3FB8AA3B, "log2e", vec_log2e)
    ln2 = load(code, 0x3F317218, "ln2", vec_ln2)
    c0 = load(code, 0x3F800001, "c0", vec_c0)
    c1 = load(code, 0x3F800000, "c1", vec_c1)
    c2 = load(code, 0x3EFFFE85, "c2", vec_c2)
    c3 = load(code, 0x3E2AAA3E, "c3", vec_c3)
    c4 = load(code, 0x3D2BB1B1, "c4", vec_c4)
    c5 = load(code, 0x3C091EC1, "c5", vec_c5)
    return Exp5Coefficients(
        c0, c1, c2, c3, halves, log2e, c4, c5, ln2, expmask, logfmax, logfmin
    )


def libxsmm_generator_exp_ps_3dts_avx512(
    code: GeneratedCode,
    x: SSAValue[x86.registers.AVX512RegisterType],
    vec_y: x86.registers.AVX512RegisterType,
    vec_z: x86.registers.AVX512RegisterType,
    coeffs: Exp3Coefficients,
) -> SSAValue[x86.registers.AVX512RegisterType]:
    x = code.insert(
        x86_ext.RSS_Vfmadd213psOp(x, coeffs.log2e, coeffs.halves)
    ).register_out
    rounded = x86_ext.DSI_VrndscalepsOp(x, 1, destination=vec_y)
    rounded.attributes["sae"] = UnitAttr()
    y = code.insert(rounded).destination
    y = code.insert(x86_ext.DSS_VsubpsOp(x, y, destination=vec_y)).destination
    z = code.insert(x86_ext.DS_Vmovdqu64Op(y, destination=vec_z)).destination
    z = code.insert(x86_ext.RSS_Vfmadd213psOp(z, coeffs.c3, coeffs.c2)).register_out
    z = code.insert(x86_ext.RSS_Vfmadd213psOp(z, y, coeffs.c1)).register_out
    z = code.insert(x86_ext.RSS_Vfmadd213psOp(z, y, coeffs.c0)).register_out
    return code.insert(x86_ext.DSS_VscalefpsOp(z, x, destination=x.type)).destination


def libxsmm_generator_exp_ps_5dts_avx512(
    code: GeneratedCode,
    src: SSAValue[x86.registers.AVX512RegisterType],
    vec_aux1: x86.registers.AVX512RegisterType,
    vec_aux2: x86.registers.AVX512RegisterType,
    aux_mask: x86.registers.AVX512MaskRegisterType,
    coeffs: Exp5Coefficients,
) -> SSAValue[x86.registers.AVX512RegisterType]:
    mask = code.insert(
        x86_ext.DSSI_VcmppsOp(src, coeffs.logfmin, 1, destination=aux_mask)
    ).destination
    src = code.insert(
        x86_ext.DSS_VminpsOp(coeffs.logfmax, src, destination=src.type)
    ).destination
    src = code.insert(
        x86_ext.DSS_VmaxpsOp(coeffs.logfmin, src, destination=src.type)
    ).destination
    aux1 = code.insert(x86_ext.DS_Vmovdqu64Op(src, destination=vec_aux1)).destination
    src = code.insert(
        x86_ext.RSS_Vfmadd213psOp(src, coeffs.log2e, coeffs.halves)
    ).register_out
    rounded = x86_ext.DSI_VrndscalepsOp(src, 1, destination=vec_aux2)
    rounded.attributes["sae"] = UnitAttr()
    aux2 = code.insert(rounded).destination
    src = code.insert(x86_ext.DS_Vmovdqu64Op(aux2, destination=src.type)).destination
    aux1 = code.insert(x86_ext.RSS_Vfnmadd231psOp(aux1, coeffs.ln2, aux2)).register_out
    aux2 = code.insert(x86_ext.DS_Vcvtps2dqOp(src, destination=vec_aux2)).destination
    aux2 = code.insert(
        x86_ext.DSS_VpadddOp(coeffs.expmask, aux2, destination=vec_aux2)
    ).destination
    aux2 = code.insert(x86_ext.DSI_VpslldOp(aux2, 23, destination=vec_aux2)).destination
    zero = code.insert(x86.ops.DSS_VpxordOp(src, src, destination=src.type)).destination
    aux2 = code.insert(
        x86_ext.DSSK_VblendmpsOp(aux2, zero, mask, destination=vec_aux2)
    ).destination
    src = code.insert(
        x86_ext.DS_Vmovdqu64Op(coeffs.c5, destination=src.type)
    ).destination
    for c in (coeffs.c4, coeffs.c3, coeffs.c2, coeffs.c1, coeffs.c0):
        src = code.insert(x86_ext.RSS_Vfmadd213psOp(src, aux1, c)).register_out
    return code.insert(
        x86_ext.DSS_VmulpsOp(aux2, src, destination=src.type)
    ).destination
