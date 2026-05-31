from pathlib import Path

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.x86_func import FuncOp, RetOp
from xdsl.ir import Block, Region
from xdsl.printer import Printer

from autotuner.compxsmm_gemm.generator_gemm_sse_avx_avx2_avx512 import (
    compxsmm_generator_gemm_sse_avx_avx2_avx512_kernel_wrapper,
)
from autotuner.libxsmm_gemm.generator_common import libxsmm_mmfunction_signature
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_main import DescDatatype, GEMMDescriptor, GEMMFlag
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


def compxsmm_generator_gemm_directasm(
    file_out: Path, routine_name: str, desc: GEMMDescriptor, arch: Arch
):
    module_op = ModuleOp(Region(Block()))

    # Instead of function signature, generate `FuncOp`.
    func_op = libxsmm_mmfunction_signature(module_op, routine_name)

    # Generate the actual kernel code for current description depending on the
    # architecture.
    compxsmm_generator_gemm_kernel(func_op, arch, desc)

    func_op.body.blocks[-1].add_op(RetOp())

    # Append code to source file
    with open(file_out, "w") as f:
        Printer(stream=f).print_op(func_op)


def compxsmm_generator_gemm_kernel(func_op: FuncOp, arch: Arch, desc: GEMMDescriptor):
    m, n, k, lda, ldb, ldc, datatype, flags, prefetch = desc

    vector_length = 1
    aarch64_bfdot = False
    aarch64_i8dot = False
    is_ai4_bi8_gemm = False
    is_ai2_bi8_gemm = False
    is_ai1_bi8_gemm = False
    is_amxfp4_bbf16_gemm = False
    is_amxfp4_bfp32_gemm = False
    is_amxfp4_bi8_gemm = False
    is_abf8_bbf16_gemm = False
    is_abf8_bf16_gemm = False
    is_ahf8_bbf16_gemm = False

    # all leading dimensions are 0
    var_ld = not any((lda, ldb, ldc))

    # Support this precision only in avx2 for now  */
    if is_amxfp4_bfp32_gemm:
        if Arch.LIBXSMM_X86_AVX2 <= arch.value <= Arch.LIBXSMM_AARCH64_ALLFEAT:
            arch = Arch.LIBXSMM_X86_AVX2

    if is_amxfp4_bi8_gemm:
        if arch != Arch.LIBXSMM_X86_AVX2_SRF:
            raise ValueError(f"Invalid arch {arch}")

    if is_amxfp4_bbf16_gemm:
        if Arch.LIBXSMM_X86_AVX2 <= arch.value <= Arch.LIBXSMM_X86_AVX512_SPR:
            if (
                Arch.LIBXSMM_X86_AVX2_SRF
                <= arch.value
                <= Arch.LIBXSMM_X86_AVX512_VL128_SKX
            ):
                arch = Arch.LIBXSMM_X86_AVX2_SRF
            else:
                arch = Arch.LIBXSMM_X86_AVX2

    # Check if it s a supported spmm with bitmap
    if GEMMFlag.DECOMPRESS_A_VIA_BITMASK in flags:
        raise NotImplementedError

    #   /* check for generally supported precisions */
    #   if ( !(
    allowed_precisions = [
        # (a, b, comp, c)
        ("f64", "f64", "f64", "f64"),
        ("f32", "f32", "f32", "f32"),
        ("i16", "i16", "i32", "i32"),
        ("i8", "i8", "i32", "i32"),
        # ("i8", "u8", "i32", "i32"),  # disabled in original with #if 0
        ("i8", "i8", "i32", "f32"),
        ("i8", "i8", "i32", "bf16"),
        # ('i8', 'u8', 'i32', 'f32'),  # disabled in original with #if 0
        ("bf8", "f16", "f16", "f16"),
        ("bf8", "f16", "f32", "f16"),
        ("bf8", "f16", "implicit", "f16"),
        ("bf8", "f16", "f16", "f32"),
        ("bf8", "f16", "f32", "f32"),
        ("bf8", "f16", "implicit", "f32"),
        ("i8", "f16", "f16", "f16"),
        ("i8", "bf16", "f32", "bf16"),
        ("i8", "bf16", "f32", "f32"),
        ("i8", "f32", "f32", "f32"),
        ("i8", "f16", "f32", "f16"),
        ("i8", "f16", "implicit", "f16"),
        ("i8", "f16", "f16", "f32"),
        ("i8", "f16", "f32", "f32"),
        ("i8", "f16", "implicit", "f32"),
        ("f16", "f16", "f16", "f16"),
        ("f16", "f16", "f32", "f16"),
        ("f16", "f16", "implicit", "f16"),
        ("f16", "f16", "f16", "f32"),
        ("f16", "f16", "f32", "f32"),
        ("f16", "f16", "implicit", "f32"),
        ("bf16", "bf16", "f32", "f32"),
        ("bf16", "bf16", "f32", "bf16"),
        ("bf8", "bf16", "f32", "f32"),
        ("bf8", "bf16", "f32", "bf16"),
        ("hf8", "bf16", "f32", "f32"),
        ("hf8", "bf16", "f32", "bf16"),
        ("bf8", "bf8", "f32", "f32"),
        ("bf8", "bf8", "f32", "bf8"),
        ("hf8", "hf8", "f32", "f32"),
        ("hf8", "hf8", "f32", "hf8"),
    ]

    if tuple(datatype) not in allowed_precisions:
        raise ValueError(f"Unsupported precision combination: {datatype}")

    if arch >= Arch.LIBXSMM_RV64_MVL128:
        # Currently, RVV supports F32 without transpose only
        match datatype:
            case DescDatatype(
                Datatype.F32, Datatype.F32, Datatype.F32, Datatype.F32
            ) | DescDatatype(
                Datatype.F64,
                Datatype.F64,
                Datatype.F64,
                Datatype.F64,
            ):
                pass
            case _:
                raise NotImplementedError

    if GEMMFlag.VNNI_A in flags:
        raise NotImplementedError

    if GEMMFlag.VNNI_B in flags:
        raise NotImplementedError

    if Arch.LIBXSMM_X86_GENERIC <= arch <= Arch.LIBXSMM_X86_ALLFEAT:
        if datatype.ab is not None:
            #   /* Supported JITed combos: */
            #   /* i2i8 && m=32 && arch == SRF (B both signed and unsigned) */
            #   /* i1i8 && m=32 && arch == SRF */
            #   /* i2i8 && m=32 && arch >= SPR (B both signed and unsigned) */
            #   /* i2i8 && m=64 && arch >= AVX_512 && arch < SPR && B unsigned */
            #   /* i1i8 && m=64 && arch >= SPR && B unsigned */

            #   /* Check for supported i2i8 and i1i8 combinations */
            if is_ai4_bi8_gemm or is_ai2_bi8_gemm or is_ai1_bi8_gemm:
                raise NotImplementedError
        else:
            # We only handle case where data types are all the same
            raise NotImplementedError

    # We allow b vnniT for x86 and bf16 whenever possible
    if GEMMFlag.VNNI_B in flags and (
        Arch.LIBXSMM_X86_GENERIC <= arch <= Arch.LIBXSMM_X86_ALLFEAT
    ):
        assert GEMMFlag.TRANS_B in flags
        assert datatype.ab == Datatype.BF16
        # We are fine, use avx512 path
        if arch >= Arch.LIBXSMM_X86_AVX512_SPR:
            arch = Arch.LIBXSMM_X86_AVX512_SKX

    # Overwrite VNNI Flag when K == 1
    if datatype.ab == Datatype.BF16 and k == 1 and GEMMFlag.VNNI_A in flags:
        flags &= ~GEMMFlag.VNNI_A

    assert datatype.c not in (Datatype.I16, Datatype.I8)

    # determining vector length depending on architecture and precision
    if arch <= Arch.LIBXSMM_TARGET_ARCH_GENERIC:
        # Nothing to do
        pass
    elif arch < Arch.LIBXSMM_X86_AVX and datatype.ab == Datatype.F64:
        vector_length = 2
    elif arch < Arch.LIBXSMM_X86_AVX and datatype.ab == Datatype.F32:
        vector_length = 4
    elif arch < Arch.LIBXSMM_X86_AVX and datatype.ab == Datatype.I8:
        vector_length = 4
        assert not k % 4
    elif arch < Arch.LIBXSMM_X86_AVX and datatype.ab == Datatype.I16:
        vector_length = 4
        assert not k % 2
    elif arch < Arch.LIBXSMM_X86_AVX and datatype.ab == Datatype.BF16:
        # some checks as we cannot mask everything
        assert not k % 2 or GEMMFlag.VNNI_A not in flags
        if GEMMFlag.VNNI_A not in flags:
            vector_length = 8
        else:
            vector_length = 4
    elif arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX and datatype.ab == Datatype.F64:
        vector_length = 4
    elif arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX and datatype.ab == Datatype.F32:
        vector_length = 8
    elif (
        Arch.LIBXSMM_X86_AVX2 <= arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX
        and datatype.ab == Datatype.I8
    ):
        vector_length = 8
        assert k % 4 == 0, "For AVX2-I8, K must be divisible by 4."
    elif (
        Arch.LIBXSMM_X86_AVX2 <= arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX
        and datatype.ab == Datatype.I16
    ):
        vector_length = 8
        assert k % 2 == 0, "For AVX2-I16, K must be divisible by 2."
    elif (
        Arch.LIBXSMM_X86_AVX2 <= arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX
        and datatype.ab == Datatype.BF16
    ):
        # some checks as we cannot mask everything
        assert not (k % 2 != 0 and GEMMFlag.VNNI_A in flags), (
            "For AVX2-BF16 with VNNI_A, K must be even."
        )
        if GEMMFlag.VNNI_A not in flags:
            vector_length = 16
        else:
            vector_length = 8
    elif (
        arch >= Arch.LIBXSMM_X86_AVX2 and (is_amxfp4_bfp32_gemm or is_amxfp4_bbf16_gemm)
    ) or (arch >= Arch.LIBXSMM_X86_AVX2_SRF and is_amxfp4_bi8_gemm):
        vector_length = 8
        assert k % 32 == 0, (
            "For Amxfp4_Bfp32/Bbf16/Bi8 kernels, K must be divisible by 32."
        )
    elif arch <= Arch.LIBXSMM_X86_AVX512_VL256_SKX and datatype.ab == Datatype.F64:
        vector_length = 4
    elif arch <= Arch.LIBXSMM_X86_AVX512_VL256_SKX and datatype.ab == Datatype.F32:
        vector_length = 8
    elif arch == Arch.LIBXSMM_X86_AVX512_VL256_CLX and datatype.ab == Datatype.F64:
        vector_length = 4
    elif arch == Arch.LIBXSMM_X86_AVX512_VL256_CLX and datatype.ab == Datatype.F32:
        vector_length = 8
    elif arch == Arch.LIBXSMM_X86_AVX512_VL256_CPX and datatype.ab == Datatype.F64:
        vector_length = 4
    elif arch == Arch.LIBXSMM_X86_AVX512_VL256_CPX and datatype.ab == Datatype.F32:
        vector_length = 8
    elif arch <= Arch.LIBXSMM_X86_ALLFEAT and datatype.ab == Datatype.F64:
        vector_length = 8
    elif arch <= Arch.LIBXSMM_X86_ALLFEAT and datatype.ab == Datatype.F32:
        vector_length = 16
    elif (
        arch <= Arch.LIBXSMM_X86_ALLFEAT
        and Arch.LIBXSMM_X86_AVX512_VL256_SKX <= arch < Arch.LIBXSMM_X86_AVX512_SKX
        and datatype.ab == Datatype.I16
    ):
        vector_length = 8
        # some checks as we cannot mask everything
        assert k % 2 == 0, "For AVX512VL_I16, K must be even."
    elif (
        Arch.LIBXSMM_X86_AVX512_SKX <= arch <= Arch.LIBXSMM_X86_ALLFEAT
        and datatype.ab == Datatype.I16
    ):
        vector_length = 16
        # some checks as we cannot mask everything
        assert not k % 2, "For AVX512SKX+ I16, K must be even."
    elif Arch.LIBXSMM_X86_AVX512_VL256_SKX <= arch < Arch.LIBXSMM_X86_AVX512_SKX and (
        datatype.ab == Datatype.I8
        or datatype.ab == Datatype.HF8
        or datatype.ab == Datatype.BF8
    ):
        vector_length = 8
        assert not k % 4, "For AVX512VL_I8/HF8/BF8, K must be divisible by 4."
    elif (
        Arch.LIBXSMM_X86_AVX512_SKX <= arch <= Arch.LIBXSMM_X86_ALLFEAT
        and datatype.ab in (Datatype.I8, Datatype.HF8, Datatype.BF8)
    ):
        vector_length = 16
        # some checks as we cannot mask everything
        if k % 4:
            assert not (
                arch >= Arch.LIBXSMM_X86_AVX512_SPR
                and not k % 2
                and GEMMFlag.VNNI_A not in flags
            ), (
                "For AVX512_SKX+ I8/HF8/BF8, K must be divisible by 4 (unless SPR+ and K even, no VNNI_A)."
            )
        assert not is_ai4_bi8_gemm or not k % 8, (
            "For Ai4_Bi8_gemm, K must be divisible by 8."
        )
        assert not is_ai2_bi8_gemm or not k % 4 and not m % 32, (
            "For Ai2_Bi8_gemm, K must be divisible by 4 and M by 32."
        )
        assert not is_ai1_bi8_gemm or not k % 4 and not m % 16, (
            "For Ai1_Bi8_gemm, K must be divisible by 4 and M by 16."
        )
    elif (
        arch <= Arch.LIBXSMM_X86_ALLFEAT
        and Arch.LIBXSMM_X86_AVX512_VL256_SKX <= arch < Arch.LIBXSMM_X86_AVX512_SKX
        and datatype.ab == Datatype.BF16
    ):
        # some checks as we cannot mask everything
        assert GEMMFlag.VNNI_A not in flags or k % 2, (
            "For AVX512VL_BF16 with VNNI_A, K must be even."
        )
        if GEMMFlag.VNNI_A not in flags:
            vector_length = 16
        else:
            vector_length = 8
    elif (
        Arch.LIBXSMM_X86_AVX512_SKX <= arch <= Arch.LIBXSMM_X86_ALLFEAT
        and datatype.ab == Datatype.BF16
    ):
        # some checks as we cannot mask everything
        assert GEMMFlag.VNNI_A not in flags or k % 2, (
            "For AVX512_SKX+ BF16 with VNNI_A, K must be even."
        )
        if GEMMFlag.VNNI_A not in flags:
            vector_length = 32
        else:
            vector_length = 16
    elif Arch.LIBXSMM_X86_AVX512_GNR <= arch <= Arch.LIBXSMM_X86_ALLFEAT and (
        datatype.ab == Datatype.F16 and not k % 2 and GEMMFlag.VNNI_A in flags
    ):
        vector_length = 16
    elif (
        arch <= Arch.LIBXSMM_X86_ALLFEAT
        and datatype.a == Datatype.I8
        and datatype.b == Datatype.BF16
        and (datatype.c in (Datatype.BF16, Datatype.F32))
    ):
        assert not (GEMMFlag.VNNI_A in flags and not is_amxfp4_bbf16_gemm), (
            "Unsupported: VNNI_A for I8-BF16 unless Amxfp4_Bbf16_gemm."
        )
        if arch >= Arch.LIBXSMM_X86_AVX512_CPX:
            vector_length = 16
        else:
            vector_length = 8
    elif (
        Arch.LIBXSMM_X86_AVX512_SPR <= arch <= Arch.LIBXSMM_X86_ALLFEAT
        and (datatype.a in (Datatype.I8, Datatype.BF8))
        and datatype.b == Datatype.F16
        and (datatype.c in (Datatype.F16, Datatype.F32))
    ):
        assert not (GEMMFlag.VNNI_A in flags and arch < Arch.LIBXSMM_X86_AVX512_GNR), (
            "Unsupported: VNNI_A for I8/BF8-F16 unless >= AVX512_GNR."
        )
        if datatype.comp == Datatype.F16 or datatype.comp == Datatype.IMPLICIT:
            vector_length = 32
        else:
            vector_length = 16
    elif (
        Arch.LIBXSMM_X86_AVX512_SPR <= arch <= Arch.LIBXSMM_X86_ALLFEAT
        and datatype.a in (Datatype.HF8, Datatype.BF8)
        and datatype.b == Datatype.BF16
        and (datatype.c == Datatype.BF16 or datatype.c == Datatype.F32)
    ):
        assert GEMMFlag.VNNI_A in flags and not k % 2, (
            "Unsupported: For HF8/BF8-BF16 with VNNI_A, K must be even."
        )
        vector_length = 16
    elif (
        Arch.LIBXSMM_X86_AVX512_SPR <= arch <= Arch.LIBXSMM_X86_ALLFEAT
        and datatype.ab == Datatype.F16
        and (datatype.c in (Datatype.F16, Datatype.F32))
    ):
        assert GEMMFlag.VNNI_A not in flags, (
            "Unsupported: VNNI_A with F16 input and F16/F32 output on >= AVX512_SPR."
        )
        if datatype.comp == Datatype.F16 or datatype.comp == Datatype.IMPLICIT:
            vector_length = 32
        else:
            vector_length = 16
    elif (
        Arch.LIBXSMM_X86_AVX512_SKX <= arch < Arch.LIBXSMM_X86_AVX512_SPR
        and datatype.ab == Datatype.F16
        and (datatype.c in (Datatype.F16, Datatype.F32))
    ):
        assert GEMMFlag.VNNI_A not in flags, (
            "Unsupported: VNNI_A with F16 input and F16/F32 output on SKX family (before SPR)."
        )
        vector_length = 16
    elif (
        Arch.LIBXSMM_X86_AVX512_VL256_SKX <= arch < Arch.LIBXSMM_X86_AVX512_SKX
        and datatype.ab == Datatype.F16
        and datatype.c in (Datatype.F16, Datatype.F32)
    ):
        assert GEMMFlag.VNNI_A not in flags, (
            "Unsupported: VNNI_A with F16 input and F16/F32 output on <= VL256_SKX and < SKX."
        )
        vector_length = 8
    elif (
        Arch.LIBXSMM_X86_AVX512_SKX <= arch < Arch.LIBXSMM_X86_AVX512_SPR
        and datatype.a in (Datatype.I8, Datatype.BF8)
        and datatype.b == Datatype.F16
        and datatype.c in (Datatype.F16, Datatype.F32)
    ):
        assert GEMMFlag.VNNI_A not in flags, (
            "Unsupported: VNNI_A for I8/BF8-F16 unless >= AVX512_GNR."
        )
        vector_length = 16
    elif (
        Arch.LIBXSMM_X86_AVX512_VL256_SKX <= arch < Arch.LIBXSMM_X86_AVX512_SKX
        and datatype.a in (Datatype.I8, Datatype.BF8)
        and datatype.b == Datatype.F16
        and datatype.c in (Datatype.F16, Datatype.F32)
    ):
        assert GEMMFlag.VNNI_A not in flags, (
            "Unsupported: VNNI_A for I8/BF8-F16 on VL256_SKX <= arch < SKX."
        )
        vector_length = 8
    elif (
        is_ai4_bi8_gemm
        or is_ai2_bi8_gemm
        or is_ai1_bi8_gemm
        or is_abf8_bbf16_gemm
        or is_ahf8_bbf16_gemm
        or is_abf8_bf16_gemm
        or is_amxfp4_bbf16_gemm
        or is_amxfp4_bfp32_gemm
        or is_amxfp4_bi8_gemm
    ) and (Arch.LIBXSMM_AARCH64_V81 <= arch <= Arch.LIBXSMM_AARCH64_ALLFEAT):
        assert False, "Unsupported GEMM combination or architecture precision."
    elif arch == Arch.LIBXSMM_AARCH64_V81 and datatype.ab == Datatype.F32:
        vector_length = 4
    elif arch == Arch.LIBXSMM_AARCH64_V81 and datatype.ab == Datatype.F64:
        vector_length = 2
    elif arch == Arch.LIBXSMM_AARCH64_V82 and datatype.ab == Datatype.F32:
        vector_length = 4
    elif arch == Arch.LIBXSMM_AARCH64_V82 and datatype.ab == Datatype.F64:
        vector_length = 2
    elif arch == Arch.LIBXSMM_AARCH64_APPL_M1 and datatype.ab == Datatype.F32:
        vector_length = 4
    elif arch == Arch.LIBXSMM_AARCH64_APPL_M1 and datatype.ab == Datatype.F64:
        vector_length = 2
    elif (
        arch in (Arch.LIBXSMM_AARCH64_SVE128, Arch.LIBXSMM_AARCH64_NEOV2)
        and datatype.ab == Datatype.F32
    ):
        vector_length = 4
    elif (
        arch in (Arch.LIBXSMM_AARCH64_SVE128, Arch.LIBXSMM_AARCH64_NEOV2)
        and datatype.ab == Datatype.F64
    ):
        vector_length = 2
    elif (
        arch in (Arch.LIBXSMM_AARCH64_SVE256, Arch.LIBXSMM_AARCH64_NEOV1)
        and datatype.ab == Datatype.F32
    ):
        vector_length = 8
    elif (
        arch in (Arch.LIBXSMM_AARCH64_SVE256, Arch.LIBXSMM_AARCH64_NEOV1)
        and datatype.ab == Datatype.F64
    ):
        vector_length = 4
    elif (
        arch in (Arch.LIBXSMM_AARCH64_SVE512, Arch.LIBXSMM_AARCH64_A64FX)
        and datatype.ab == Datatype.F32
    ):
        vector_length = 16
    elif (
        arch in (Arch.LIBXSMM_AARCH64_SVE512, Arch.LIBXSMM_AARCH64_A64FX)
        and datatype.ab == Datatype.F64
    ):
        vector_length = 8
    elif arch == Arch.LIBXSMM_AARCH64_APPL_M4 and datatype.ab == Datatype.F32:
        vector_length = 16
    elif arch == Arch.LIBXSMM_AARCH64_APPL_M4 and datatype.ab == Datatype.F64:
        vector_length = 8
    elif Arch.LIBXSMM_AARCH64_V81 <= arch <= Arch.LIBXSMM_AARCH64_ALLFEAT and (
        (datatype.ab == Datatype.BF16 and aarch64_bfdot)
        or (datatype.ab == Datatype.I8 and aarch64_i8dot)
    ):
        # TODO (BFDOT): add flags and check on MMLA-formated A/B
        # TODO (BFDOT): add support for at least m % 2 == 0, k % 4 == 0 when running BF16
        # TODO (BFDOT): adjust checks for future SVE kernels
        if datatype.ab == Datatype.BF16:
            assert not desc.k % 2 or GEMMFlag.VNNI_A not in flags, (
                "Unsupported: k must be a multiple of 2 when VNNI_A flag is set for BF16/BFDOT on AARCH64_V81..ALLFEAT"
            )
        elif datatype.ab == Datatype.I8:
            assert desc.k % 4 == 0, (
                "Unsupported: k must be a multiple of 4 for I8/i8dot on AARCH64_V81..ALLFEAT"
            )
        # ASIMD + BFDOT
        assert Arch.LIBXSMM_AARCH64_SVE128 <= arch, (
            "Unsupported architecture: ASIMD + BFDOT not available below SVE128"
        )
        # SVE256 + BFDOT
        if arch in (Arch.LIBXSMM_AARCH64_SVE256, Arch.LIBXSMM_AARCH64_NEOV1):
            vector_length = 8
        # SVE128 + BFDOT
        elif arch in (Arch.LIBXSMM_AARCH64_SVE128, Arch.LIBXSMM_AARCH64_NEOV2):
            vector_length = 4
        else:
            assert False, "Unsupported architecture for BFDOT"
    elif Arch.LIBXSMM_AARCH64_V81 <= arch <= Arch.LIBXSMM_AARCH64_ALLFEAT and (
        (datatype.ab == Datatype.BF16 and not aarch64_bfdot)
        or (datatype.ab == Datatype.I8 and not aarch64_i8dot)
    ):
        # TODO (MMLA): add flags and check on MMLA-formated A/B
        # TODO (MMLA): add support for at least m % 2 == 0, k % 4 == 0 when running BF16
        # TODO (MMLA): adjust checks for future SVE kernels
        if datatype.ab == Datatype.BF16:
            assert GEMMFlag.VNNI_A not in flags or not desc.k % 4, (
                "Unsupported: k must be a multiple of 4 for BF16 + VNNI_A"
            )
        elif datatype.ab == Datatype.I8:
            assert desc.k % 8 == 0, "Unsupported: k must be a multiple of 8 for I8"
            assert GEMMFlag.A_UNSIGNED not in flags or GEMMFlag.B_UNSIGNED in flags, (
                "Unsupported: A unsigned but B not for I8"
            )
        # ASIMD + MMLA
        # TODO: These are not properly implemented yet
        assert Arch.LIBXSMM_AARCH64_SVE128 <= arch, (
            "Unsupported architecture: ASIMD + MMLA not available below SVE128"
        )
        # SVE256 + MMLA
        if arch in (Arch.LIBXSMM_AARCH64_SVE256, Arch.LIBXSMM_AARCH64_NEOV1):
            vector_length = 8
        # SVE128 + MMLA
        elif arch in (Arch.LIBXSMM_AARCH64_SVE128, Arch.LIBXSMM_AARCH64_NEOV2):
            vector_length = 4
        else:
            assert False, "Unsupported architecture for MMLA"
    elif (
        arch in (Arch.LIBXSMM_RV64_MVL128, Arch.LIBXSMM_RV64_MVL128_LMUL)
    ) and datatype.ab == Datatype.F32:
        vector_length = 4
    elif (
        arch in (Arch.LIBXSMM_RV64_MVL128, Arch.LIBXSMM_RV64_MVL128_LMUL)
    ) and datatype.ab == Datatype.F64:
        vector_length = 2
    elif (
        arch in (Arch.LIBXSMM_RV64_MVL256, Arch.LIBXSMM_RV64_MVL256_LMUL)
    ) and datatype.ab == Datatype.F32:
        vector_length = 8
    elif (
        arch in (Arch.LIBXSMM_RV64_MVL256, Arch.LIBXSMM_RV64_MVL256_LMUL)
    ) and datatype.ab == Datatype.F64:
        vector_length = 4
    else:
        print(arch)
        assert False, (
            "Unsupported architecture or datatype for vector length determination"
        )

    # Check LDA
    if flags & GEMMFlag.TRANS_A:
        assert k <= lda or var_ld

        if datatype.ab not in (Datatype.F32, Datatype.F64, Datatype.BF16):
            assert datatype.ab == Datatype.F16 and arch >= Arch.LIBXSMM_X86_AVX512_DMR
        else:
            # BF16 A transpose is supported forflat A
            assert GEMMFlag.VNNI_A not in flags
    else:
        assert m <= lda or var_ld

    # Check LDB
    if GEMMFlag.TRANS_B in flags:
        assert n <= ldb or var_ld
    else:
        assert k <= ldb or var_ld

    # Check LDC
    assert ldc >= m or var_ld

    # Check for trans A cases which are not supported in the generator
    if flags & GEMMFlag.TRANS_A:
        if datatype.ab not in (Datatype.F32, Datatype.F64, Datatype.BF16):
            assert datatype.ab == Datatype.F16 and arch >= Arch.LIBXSMM_X86_AVX512_DMR
        else:
            # BF16 A transpose is supported for flat A
            assert not flags & GEMMFlag.VNNI_A

    # Check for trans B cases which are not supported in the generator
    if GEMMFlag.TRANS_B in flags:
        assert datatype.ab not in (Datatype.I16, Datatype.I8)
        if datatype.ab == Datatype.BF16:
            # we are fine, we do support mmla kernels with B in vnni4t
            assert not aarch64_bfdot and GEMMFlag.VNNI_B in flags

    if GEMMFlag.VNNI_B in flags:
        raise NotImplementedError

    # Check if alignment is not possible
    if lda % vector_length:
        flags &= ~GEMMFlag.ALIGN_A

    if ldc % vector_length:
        flags &= ~GEMMFlag.ALIGN_C

    desc_mod = GEMMDescriptor(m, n, k, lda, ldb, ldc, datatype, flags, prefetch)

    if arch <= Arch.LIBXSMM_TARGET_ARCH_GENERIC:
        raise NotImplementedError
    elif arch <= Arch.LIBXSMM_X86_ALLFEAT:
        # call actual kernel generation with revised parameters
        # TODO: check for VNNI format

        if (
            (Arch.LIBXSMM_X86_AVX512_SPR <= arch < Arch.LIBXSMM_X86_ALLFEAT)
            and (
                datatype.ab == Datatype.BF16
                or datatype.ab == Datatype.F16
                and Arch.LIBXSMM_X86_AVX512_GNR <= arch
                or datatype.ab == Datatype.I8
            )
            and (
                GEMMFlag.VNNI_A in flags
                or GEMMFlag.VNNI_A not in flags
                and Arch.LIBXSMM_X86_AVX512_DMR <= arch
            )
        ):
            raise NotImplementedError
            # libxsmm_generator_gemm_amx_kernel_wrapper( io_generated_code, &l_xgemm_desc_mod );
        elif (
            Arch.LIBXSMM_X86_AVX512_SPR <= arch <= Arch.LIBXSMM_X86_ALLFEAT
            and is_abf8_bbf16_gemm
            or is_ahf8_bbf16_gemm
            or is_amxfp4_bbf16_gemm
            or datatype.ab in (Datatype.BF16, Datatype.BF8, Datatype.HF8)
        ):
            raise NotImplementedError
            # libxsmm_generator_gemm_amx_kernel_wrapper( io_generated_code, &l_xgemm_desc_mod );
        elif (
            (Arch.LIBXSMM_X86_AVX512_GNR <= arch < Arch.LIBXSMM_X86_ALLFEAT)
            and is_abf8_bf16_gemm
            and GEMMFlag.VNNI_A in flags
        ):
            raise NotImplementedError
            # libxsmm_generator_gemm_amx_kernel_wrapper( io_generated_code, &l_xgemm_desc_mod );
        else:
            compxsmm_generator_gemm_sse_avx_avx2_avx512_kernel_wrapper(
                func_op, arch, desc_mod
            )
    elif arch in (Arch.LIBXSMM_AARCH64_V81, Arch.LIBXSMM_AARCH64_V82):
        raise NotImplementedError
    elif arch in (Arch.LIBXSMM_AARCH64_V81, Arch.LIBXSMM_AARCH64_V82):
        raise NotImplementedError
    elif arch == Arch.LIBXSMM_AARCH64_APPL_M1:
        raise NotImplementedError
    elif arch in (Arch.LIBXSMM_AARCH64_SVE128, Arch.LIBXSMM_AARCH64_NEOV2):
        raise NotImplementedError
    elif arch in (Arch.LIBXSMM_AARCH64_SVE256, Arch.LIBXSMM_AARCH64_NEOV1):
        raise NotImplementedError
    elif arch in (Arch.LIBXSMM_AARCH64_SVE512, Arch.LIBXSMM_AARCH64_A64FX):
        raise NotImplementedError
    elif arch == Arch.LIBXSMM_AARCH64_APPL_M4:
        raise NotImplementedError
    elif arch == Arch.LIBXSMM_RV64_MVL128:
        raise NotImplementedError
    else:
        assert False, f"Unsupported arch: {arch}"
