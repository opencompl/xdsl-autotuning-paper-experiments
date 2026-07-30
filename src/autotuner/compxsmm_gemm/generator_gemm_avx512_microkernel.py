from autotuner.libxsmm_gemm.generator_common import GPRegMapping, MicroKernelConfig
from autotuner.libxsmm_gemm.generator_gemm_avx512_microkernel import (
    libxsmm_generator_gemm_avx512_microkernel_fsdbcst,
    libxsmm_generator_gemm_avx512_microkernel_nofsdbcst,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_generator import (
    GeneratedCode,
    KLoopVals,
)
from autotuner.libxsmm_gemm.libxsmm_main import (
    GEMMDescriptor,
    GEMMFlag,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


def compxsmm_generator_gemm_avx512_kloop_kernel(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    m_blocking: int,
    n_blocking: int,
    k_blocking: int,
    vals: KLoopVals,
) -> KLoopVals:
    k = 0
    _k_pack_factor = 1
    m_vector = (
        m_blocking // micro_kernel_config.vector_length
        if (m_blocking % micro_kernel_config.vector_length == 0)
        else (m_blocking // micro_kernel_config.vector_length) + 1
    )

    is_Abf8_Bf16_gemm = (
        Datatype.BF8 == desc.datatype.a and Datatype.F16 == desc.datatype.b
    )
    is_Ai8_Bf16_gemm = (
        Datatype.I8 == desc.datatype.a and Datatype.F16 == desc.datatype.b
    )
    is_Ai8_Bbf16_gemm = (
        Datatype.I8 == desc.datatype.a and Datatype.BF16 == desc.datatype.b
    )
    is_Ai4_Bf16_gemm = (
        GEMMFlag.INTERPRETE_A_AS_INT4_VNNI2 in desc.flags
        and Datatype.I8 == desc.datatype.a
        and Datatype.F16 == desc.datatype.b
        and (Datatype.F16 == desc.datatype.c or Datatype.F32 == desc.datatype.c)
    )
    is_Ai4_Bi8_gemm = desc.is_Ai4_Bi8_gemm()
    is_Ai2_Bi8_gemm = desc.is_Ai2_Bi8_gemm()
    is_Ai1_Bi8_gemm = desc.is_Ai1_Bi8_gemm()
    is_Af16_Bf16_gemm = (
        Datatype.F16 == desc.datatype.a and Datatype.F16 == desc.datatype.b
    )
    _is_Ai8_Bbf16_gemm_bf16fma = (
        is_Ai8_Bbf16_gemm and micro_kernel_config.vmul_instruction == "VDPBF16PS"
    )

    is_i8_uu_ss_gemm = desc.datatype.ab == "I8" and (
        (
            GEMMFlag.A_UNSIGNED not in desc.flags
            and GEMMFlag.B_UNSIGNED not in desc.flags
        )
        or (GEMMFlag.A_UNSIGNED in desc.flags and GEMMFlag.B_UNSIGNED in desc.flags)
    )

    is_not_cpx_bf16 = (
        generated_code.arch != "AVX512_CPX"
        and desc.datatype.ab == "BF16"
        and GEMMFlag.VNNI_A in desc.flags
    )

    if GEMMFlag.VNNI_A in desc.flags:
        raise NotImplementedError

    if (
        m_vector == 1
        and GEMMFlag.DECOMPRESS_A_VIA_BITMASK not in desc.flags
        and not is_Abf8_Bf16_gemm
        and (not is_Ai8_Bf16_gemm)
        and (not is_Ai8_Bbf16_gemm)
        and (not is_Ai4_Bf16_gemm)
        and (not is_Ai4_Bi8_gemm)
        and (not is_Af16_Bf16_gemm)
        and (not is_i8_uu_ss_gemm)
        and (not is_Ai2_Bi8_gemm)
        and (not is_Ai1_Bi8_gemm)
        and (Datatype.BF8 != desc.datatype.ab)
        and (not is_not_cpx_bf16)
    ):
        vals = libxsmm_generator_gemm_avx512_microkernel_fsdbcst(
            generated_code,
            gp_reg_mapping,
            micro_kernel_config,
            desc,
            n_blocking,
            k_blocking,
            vals,
        )
    else:
        # void (*l_generator_microkernel)(libxsmm_generated_code*, const libxsmm_gp_reg_mapping*, const libxsmm_micro_kernel_config*,
        #                                 const libxsmm_gemm_descriptor*, const unsigned int, const unsigned int);
        if (
            is_Ai8_Bbf16_gemm
            or is_Ai4_Bf16_gemm
            or GEMMFlag.DECOMPRESS_A_VIA_BITMASK in desc.flags
            or is_Ai4_Bi8_gemm
            or is_Ai2_Bi8_gemm
            or is_Ai1_Bi8_gemm
        ):
            raise NotImplementedError
        elif (
            Arch.LIBXSMM_X86_AVX512_VL256_SKX
            <= generated_code.arch
            < Arch.LIBXSMM_X86_AVX512_SKX
        ):
            raise NotImplementedError
        elif (
            generated_code.arch != Arch.LIBXSMM_X86_AVX512_CPX
            and Datatype.BF16 == desc.datatype.ab
        ):
            raise NotImplementedError
        elif is_i8_uu_ss_gemm:
            raise NotImplementedError
        else:
            generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_nofsdbcst

        for k in range(k_blocking):
            vals = generator_microkernel(
                generated_code,
                gp_reg_mapping,
                micro_kernel_config,
                desc,
                m_blocking,
                n_blocking,
                vals,
            )

    return vals
