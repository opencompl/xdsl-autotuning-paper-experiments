"""Generate exp arithmetic with progressively expandable XSMM operations.

Keep range reduction and reconstruction in the same order as the literal
generator. Only polynomial evaluation is abstracted here; the literal port
remains an independent instruction-level baseline.
"""

from collections.abc import Sequence

from xdsl.dialects import x86, x86_func
from xdsl.dialects.builtin import FloatAttr, UnitAttr, f32
from xdsl.dialects.x86.registers import AVX512MaskRegisterType, AVX512RegisterType
from xdsl.ir import SSAValue

from autotuner.dialects import x86_ext
from autotuner.dialects.xsmm import HornerOp, PolynomialConstantsOp
from autotuner.libxsmm_gemm.generator_common_x86 import (
    Exp3Coefficients,
    Exp5Coefficients,
)
from autotuner.libxsmm_gemm.generator_mateltwise_unary_binary_avx_avx512 import (
    MeltwKernelConfig,
)
from autotuner.libxsmm_gemm.generator_x86_instructions import (
    libxsmm_x86_instruction_full_vec_load_of_constants,
)
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode


def prepare_polynomial_constants(
    code: GeneratedCode,
    bits: Sequence[int],
    registers: Sequence[AVX512RegisterType],
) -> tuple[SSAValue[AVX512RegisterType], ...]:
    func = code.current_block.parent_op()
    assert isinstance(func, x86_func.FuncOp)
    coefficients = FloatAttr.unpack(
        f32, b"".join(value.to_bytes(4, "little") for value in bits), len(bits)
    )
    return tuple(
        code.insert(
            PolynomialConstantsOp(coefficients, f".L{func.sym_name.data}_c", registers)
        ).registers
    )


def configure_exp_registers(
    code: GeneratedCode, degree: int, *, disable_regalloc: bool
) -> MeltwKernelConfig:
    """Retain native register assignments and constant order, but defer packets."""

    def zmm(index: int) -> AVX512RegisterType:
        return (
            AVX512RegisterType.unallocated()
            if disable_regalloc
            else AVX512RegisterType.from_index(index)
        )

    load = libxsmm_x86_instruction_full_vec_load_of_constants
    if degree == 3:
        halves = load(code, 0x3F000000, "halves", zmm(7))
        log2e = load(code, 0x3FB8AA3B, "log2e", zmm(2))
        c0, c1, c2, c3 = prepare_polynomial_constants(
            code,
            (0x3F34E022, 0x3EFD357F, 0x3E20BCD5, 0x3D635847),
            tuple(zmm(i) for i in (6, 5, 4, 3)),
        )
        return MeltwKernelConfig(
            8,
            zmm(0),
            zmm(1),
            Exp3Coefficients(c0, c1, c2, c3, halves, log2e),
            disable_regalloc,
        )

    assert degree == 5
    expmask = load(code, 0x0000007F, "expmask", zmm(2))
    logfmax = load(code, 0x42B17218, "logfmax", zmm(6))
    logfmin = load(code, 0xC2AEAC50, "logfmin", zmm(5))
    halves = load(code, 0x3F000000, "halves", zmm(15))
    log2e = load(code, 0x3FB8AA3B, "log2e", zmm(8))
    ln2 = load(code, 0x3F317218, "ln2", zmm(7))
    c0, c1, c2, c3, c4, c5 = prepare_polynomial_constants(
        code,
        (0x3F800001, 0x3F800000, 0x3EFFFE85, 0x3E2AAA3E, 0x3D2BB1B1, 0x3C091EC1),
        tuple(zmm(i) for i in (14, 13, 12, 11, 10, 9)),
    )
    return MeltwKernelConfig(
        16,
        zmm(4),
        zmm(3),
        Exp5Coefficients(
            c0, c1, c2, c3, halves, log2e, c4, c5, ln2, expmask, logfmax, logfmin
        ),
        disable_regalloc,
    )


def generate_exp3(
    code: GeneratedCode,
    x: SSAValue[AVX512RegisterType],
    vec_y: AVX512RegisterType,
    vec_z: AVX512RegisterType,
    coeffs: Exp3Coefficients,
) -> SSAValue[AVX512RegisterType]:
    x = code.insert(
        x86_ext.RSS_Vfmadd213psOp(x, coeffs.log2e, coeffs.halves)
    ).register_out
    rounded = x86_ext.DSI_VrndscalepsOp(x, 1, destination=vec_y)
    rounded.attributes["sae"] = UnitAttr()
    y = code.insert(rounded).destination
    y = code.insert(x86_ext.DSS_VsubpsOp(x, y, destination=vec_y)).destination
    z = code.insert(
        HornerOp(
            y, None, (coeffs.c0, coeffs.c1, coeffs.c2, coeffs.c3), destination=vec_z
        )
    ).result
    return code.insert(x86_ext.DSS_VscalefpsOp(z, x, destination=x.type)).destination


def generate_exp5(
    code: GeneratedCode,
    src: SSAValue[AVX512RegisterType],
    vec_aux1: AVX512RegisterType,
    vec_aux2: AVX512RegisterType,
    aux_mask: AVX512MaskRegisterType,
    coeffs: Exp5Coefficients,
) -> SSAValue[AVX512RegisterType]:
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
    src = code.insert(
        HornerOp(aux1, src, (coeffs.c0, coeffs.c1, coeffs.c2, coeffs.c3, coeffs.c4))
    ).result
    return code.insert(
        x86_ext.DSS_VmulpsOp(aux2, src, destination=src.type)
    ).destination


def generate_exp_block(
    code: GeneratedCode,
    values: tuple[SSAValue[AVX512RegisterType], ...],
    config: MeltwKernelConfig,
) -> tuple[SSAValue[AVX512RegisterType], ...]:
    if isinstance(config.coeffs, Exp5Coefficients):
        return tuple(
            generate_exp5(
                code,
                value,
                config.vec_y,
                config.vec_z,
                AVX512MaskRegisterType.unallocated()
                if config.disable_regalloc
                else AVX512MaskRegisterType.from_index(1),
                config.coeffs,
            )
            for value in values
        )
    return tuple(
        generate_exp3(code, value, config.vec_y, config.vec_z, config.coeffs)
        for value in values
    )
