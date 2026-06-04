from xdsl.dialects import x86
from xdsl.dialects.x86.registers import AVX512MaskRegisterType

from autotuner.libxsmm_gemm.generator_common import GPRegMapping, MicroKernelConfig
from autotuner.compxsmm_gemm.generator_x86_instructions import (
    compxsmm_x86_instruction_vec_compute_3reg,
    compxsmm_x86_instruction_vec_move_ld,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.compxsmm_gemm.libxsmm_generator import GeneratedCode
from autotuner.libxsmm_gemm.libxsmm_main import (
    GEMMDescriptor,
    GEMMFlag,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


def compxsmm_generator_gemm_avx512_kloop_kernel(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    gemm_desc: GEMMDescriptor,
    m_blocking: int,
    n_blocking: int,
    k_blocking: int,
) -> None:
    k = 0
    _k_pack_factor = 1
    m_vector = (
        m_blocking // micro_kernel_config.vector_length
        if (m_blocking % micro_kernel_config.vector_length == 0)
        else (m_blocking // micro_kernel_config.vector_length) + 1
    )

    is_Abf8_Bf16_gemm = (
        Datatype.BF8 == gemm_desc.datatype.a and Datatype.F16 == gemm_desc.datatype.b
    )
    is_Ai8_Bf16_gemm = (
        Datatype.I8 == gemm_desc.datatype.a and Datatype.F16 == gemm_desc.datatype.b
    )
    is_Ai8_Bbf16_gemm = (
        Datatype.I8 == gemm_desc.datatype.a and Datatype.BF16 == gemm_desc.datatype.b
    )
    is_Ai4_Bf16_gemm = (
        GEMMFlag.INTERPRETE_A_AS_INT4_VNNI2 in gemm_desc.flags
        and Datatype.I8 == gemm_desc.datatype.a
        and Datatype.F16 == gemm_desc.datatype.b
        and (
            Datatype.F16 == gemm_desc.datatype.c or Datatype.F32 == gemm_desc.datatype.c
        )
    )
    is_Ai4_Bi8_gemm = gemm_desc.is_Ai4_Bi8_gemm()
    is_Ai2_Bi8_gemm = gemm_desc.is_Ai2_Bi8_gemm()
    is_Ai1_Bi8_gemm = gemm_desc.is_Ai1_Bi8_gemm()
    is_Af16_Bf16_gemm = (
        Datatype.F16 == gemm_desc.datatype.a and Datatype.F16 == gemm_desc.datatype.b
    )
    _is_Ai8_Bbf16_gemm_bf16fma = (
        is_Ai8_Bbf16_gemm and micro_kernel_config.vmul_instruction == "VDPBF16PS"
    )

    is_i8_uu_ss_gemm = gemm_desc.datatype.ab == "I8" and (
        (
            GEMMFlag.A_UNSIGNED not in gemm_desc.flags
            and GEMMFlag.B_UNSIGNED not in gemm_desc.flags
        )
        or (
            GEMMFlag.A_UNSIGNED in gemm_desc.flags
            and GEMMFlag.B_UNSIGNED in gemm_desc.flags
        )
    )

    is_not_cpx_bf16 = (
        generated_code.arch != "AVX512_CPX"
        and gemm_desc.datatype.ab == "BF16"
        and GEMMFlag.VNNI_A in gemm_desc.flags
    )

    if GEMMFlag.VNNI_A in gemm_desc.flags:
        raise NotImplementedError

    if (
        m_vector == 1
        and GEMMFlag.DECOMPRESS_A_VIA_BITMASK not in gemm_desc.flags
        and not (not is_Abf8_Bf16_gemm)
        and (not is_Ai8_Bf16_gemm)
        and (not is_Ai8_Bbf16_gemm)
        and (not is_Ai4_Bf16_gemm)
        and (not is_Ai4_Bi8_gemm)
        and (not is_Af16_Bf16_gemm)
        and (not is_i8_uu_ss_gemm)
        and (not is_Ai2_Bi8_gemm)
        and (not is_Ai1_Bi8_gemm)
        and (Datatype.BF8 != gemm_desc.datatype.ab)
        and (not is_not_cpx_bf16)
    ):
        raise NotImplementedError
    else:
        # void (*l_generator_microkernel)(libxsmm_generated_code*, const libxsmm_gp_reg_mapping*, const libxsmm_micro_kernel_config*,
        #                                 const libxsmm_gemm_descriptor*, const unsigned int, const unsigned int);
        if (
            is_Ai8_Bbf16_gemm
            or is_Ai4_Bf16_gemm
            or GEMMFlag.DECOMPRESS_A_VIA_BITMASK in gemm_desc.flags
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
            and Datatype.BF16 == gemm_desc.datatype.ab
        ):
            raise NotImplementedError
        elif is_i8_uu_ss_gemm:
            raise NotImplementedError
        else:
            generator_microkernel = compxsmm_generator_gemm_avx512_microkernel_nofsdbcst

        for k in range(k_blocking):
            generator_microkernel(
                generated_code,
                gp_reg_mapping,
                micro_kernel_config,
                gemm_desc,
                m_blocking,
                n_blocking,
            )


def compxsmm_generator_gemm_avx512_microkernel_nofsdbcst(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    m_blocking: int,
    n_blocking: int,
) -> None:
    # deriving register blocking from kernel config
    m_blocking = (
        m_blocking // micro_kernel_config.vector_length
        if (m_blocking % micro_kernel_config.vector_length == 0)
        else (m_blocking // micro_kernel_config.vector_length) + 1
    )
    # register blocking counter in n
    n = 0
    # register blocking counter in m
    m = 0
    k = 0
    # start register of accumulator
    vec_reg_acc_start = micro_kernel_config.vector_reg_count - (n_blocking * m_blocking)
    vreg_ab_offset = 0
    # temp variable for b-offset to handle no-trans/trans B
    b_offset = 0
    # k packing factor for VNNI
    k_pack_factor = 1
    is_Abf8_Bf16_gemm = (Datatype.BF8 == desc.datatype.a) and (
        Datatype.F16 == desc.datatype.b
    )
    is_Ai8_Bf16_gemm = (Datatype.I8 == desc.datatype.a) and (
        Datatype.F16 == desc.datatype.b
    )
    is_Ai4_Bf16_gemm = (GEMMFlag.INTERPRETE_A_AS_INT4_VNNI2 in desc.flags) and (
        (Datatype.I8 == desc.datatype.a)
        and (Datatype.F16 == desc.datatype.b)
        and ((Datatype.F16 == desc.datatype.c) or (Datatype.F32 == desc.datatype.c))
    )
    is_Ai4_Bi8_gemm = desc.is_Ai4_Bi8_gemm()
    is_Ai2_Bi8_gemm = desc.is_Ai2_Bi8_gemm()
    is_Ai1_Bi8_gemm = desc.is_Ai1_Bi8_gemm()
    k_iters = 2 if (is_Ai4_Bf16_gemm or is_Ai4_Bi8_gemm) else 1
    is_Af16_Bf16_gemm = (desc.datatype.a == Datatype.F16) and (
        desc.datatype.b == Datatype.F16
    )
    is_Ai8_Bbf16_gemm = (desc.datatype.a == Datatype.I8) and (
        desc.datatype.b == Datatype.BF16
    )
    is_Ai8_Bbf16_gemm_bf16fma = False
    # (is_Ai8_Bbf16_gemm and (micro_kernel_config.vmul_instruction == X86Instruction.VDPBF16PS))
    use_f16_replacement_fma = (
        (is_Ai8_Bf16_gemm or is_Abf8_Bf16_gemm or is_Af16_Bf16_gemm)
        and (generated_code.arch < Arch.LIBXSMM_X86_AVX512_SPR)
        and (desc.datatype.c == Datatype.F16)
    )
    use_f32_compute_with_f16_inp = (
        is_Ai8_Bf16_gemm or is_Abf8_Bf16_gemm or is_Af16_Bf16_gemm
    ) and (desc.datatype.c == Datatype.F32)
    _mask_load_i1 = 1
    _vname_cvt = micro_kernel_config.vector_name

    if is_Ai8_Bf16_gemm:
        raise NotImplementedError
    if is_Ai8_Bbf16_gemm:
        raise NotImplementedError
    if is_Ai4_Bi8_gemm:
        raise NotImplementedError
    if is_Ai2_Bi8_gemm:
        raise NotImplementedError
    if is_Ai1_Bi8_gemm:
        raise NotImplementedError

    assert n_blocking <= 30 or n_blocking >= 1

    if (
        Arch.LIBXSMM_X86_AVX512_VL256_SKX
        <= generated_code.arch
        < Arch.LIBXSMM_X86_AVX512_SKX
    ):
        raise NotImplementedError
    else:
        assert 0 < m_blocking <= 4

        assert (((m_blocking * n_blocking) + m_blocking + 1) <= 32) or (n_blocking >= 7)

    if GEMMFlag.VNNI_A in desc.flags:
        # for VNNI we are stepping through to pack ks */
        raise NotImplementedError

    for m in range(m_blocking):
        # load column vectors of A upfront
        # const char *const l_env_a_k_pf_dist = getenv("LIBXSMM_GEMM_K_A_PF_DIST");
        # unsigned l_a_k_pf_dist = (l_env_a_k_pf_dist == 0) ? 0 : atoi(l_env_a_k_pf_dist);
        a_k_pf_dist = 0
        a_vname = (
            micro_kernel_config.vector_name
            if not (is_Ai8_Bf16_gemm or is_Abf8_Bf16_gemm)
            else (
                "x"
                if use_f32_compute_with_f16_inp
                else (
                    "x"
                    if use_f16_replacement_fma
                    else ("y" if micro_kernel_config.vector_name == "z" else "x")
                )
            )
        )
        match a_vname:
            case "x":
                a_vec_type = x86.registers.SSERegisterType
            case "y":
                a_vec_type = x86.registers.AVX2RegisterType
            case "z":
                a_vec_type = x86.registers.AVX512RegisterType

        # unsigned int l_a_vmove_instruction = ((l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && (l_m != (l_m_blocking - 1)) ) ? LIBXSMM_X86_INSTR_VMOVSD : i_micro_kernel_config->a_vmove_instruction;
        a_vmove_instruction = micro_kernel_config.a_vmove_instruction

        if is_Af16_Bf16_gemm:
            raise NotImplementedError

        if use_f16_replacement_fma:
            raise NotImplementedError

        if is_Ai8_Bbf16_gemm:
            raise NotImplementedError
        else:
            if GEMMFlag.DECOMPRESS_A_VIA_BITMASK in desc.flags:
                raise NotImplementedError
            else:
                if is_Ai1_Bi8_gemm:
                    raise NotImplementedError
                else:
                    use_masking = micro_kernel_config.use_masking_a_c and (
                        m == (m_blocking - 1)
                    )
                    mask_val = (
                        generated_code.get_val(AVX512MaskRegisterType.from_index(1))
                        if use_masking
                        else None
                    )

                    a_vec_reg = a_vec_type.from_index(
                        m + vreg_ab_offset
                        if (is_Ai2_Bi8_gemm > 0)
                        else 1 + m + vreg_ab_offset
                    )
                    compxsmm_x86_instruction_vec_move_ld(
                        generated_code,
                        micro_kernel_config.instruction_set,
                        a_vmove_instruction,
                        generated_code.get_val(gp_reg_mapping.gp_reg_a),
                        None,
                        0,
                        (micro_kernel_config.datatype_size_in)
                        * (micro_kernel_config.vector_length)
                        * m
                        * k_pack_factor,
                        a_vec_reg,
                        mask_val,
                        True,
                        False,
                    )

                if (
                    (
                        (micro_kernel_config.datatype_size_in)
                        * (micro_kernel_config.vector_length)
                        * m
                        * k_pack_factor
                    )
                    % 64
                    == 0
                ) and (a_k_pf_dist > 0):
                    raise NotImplementedError
                    # libxsmm_x86_instruction_prefetch( io_generated_code,
                    #     LIBXSMM_X86_INSTR_PREFETCHT0,
                    #     i_gp_reg_mapping->gp_reg_a,
                    #     LIBXSMM_X86_GP_REG_UNDEF, 0,
                    #     (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor + l_a_k_pf_dist * i_xgemm_desc->lda * i_micro_kernel_config->datatype_size_in);

        if is_Ai4_Bi8_gemm:
            raise NotImplementedError
        elif is_Ai2_Bi8_gemm:
            raise NotImplementedError
        elif is_Ai1_Bi8_gemm:
            raise NotImplementedError
        else:
            # Process vreg A in case of f16/i8
            ...
            # libxsmm_generator_gemm_avx512_microkernel_process_vreg_A( io_generated_code, i_micro_kernel_config, i_xgemm_desc,
            #     vname_cvt, l_is_Ai8_Bf16_gemm, l_is_Abf8_Bf16_gemm, l_is_Af16_Bf16_gemm, l_use_f16_replacement_fma, l_use_f32_compute_with_f16_inp, l_m, l_m_blocking, 1+l_m+l_vreg_ab_offset);
            # }

            if desc.prefetch == GEMMPrefetchType.AL2:
                # prefetch a different A matrix provided by the prefetch pointers
                raise NotImplementedError

    for k in range(k_iters):
        for n in range(n_blocking):
            b_vname = micro_kernel_config.vector_name
            b_vmove_instruction = micro_kernel_config.b_vmove_instruction
            if is_Af16_Bf16_gemm or is_Ai8_Bf16_gemm or is_Abf8_Bf16_gemm:
                raise NotImplementedError

            if is_Ai8_Bbf16_gemm and not is_Ai8_Bbf16_gemm_bf16fma:
                raise NotImplementedError

            # handle trans B */
            if GEMMFlag.TRANS_B in desc.flags:
                k_pack_advance = (
                    (2 if is_Ai8_Bbf16_gemm_bf16fma else 1)
                    if is_Ai8_Bbf16_gemm
                    else k_pack_factor
                )
                b_offset = (
                    n * micro_kernel_config.datatype_size_in2 * k_pack_advance
                    + (micro_kernel_config.datatype_size_in2 * k * desc.ldb)
                )
            else:
                k_pack_advance = (
                    (2 if (is_Ai8_Bbf16_gemm_bf16fma) else 1)
                    if (is_Ai8_Bbf16_gemm)
                    else k_pack_factor
                )
                b_offset = (
                    desc.ldb * n * micro_kernel_config.datatype_size_in2
                    + k * k_pack_advance * micro_kernel_config.datatype_size_in2
                )

            b_vname = (
                ("y" if b_vname == "z" else "x")
                if (
                    Datatype.BF16 == desc.datatype.ab
                    and (GEMMFlag.DECOMPRESS_A_VIA_BITMASK in desc.flags)
                )
                else b_vname
            )
            match b_vname:
                case "x":
                    dest_type = x86.registers.SSERegisterType
                case "y":
                    dest_type = x86.registers.AVX2RegisterType
                case "z":
                    dest_type = x86.registers.AVX512RegisterType
            b_vec_reg = dest_type.from_index(vreg_ab_offset)

            compxsmm_x86_instruction_vec_move_ld(
                generated_code,
                micro_kernel_config.instruction_set,
                b_vmove_instruction,
                generated_code.get_val(gp_reg_mapping.gp_reg_b),
                x86.registers.UNALLOCATED_REG64,
                0,
                b_offset,
                b_vec_reg,
                None,
                True,
                False,
            )
            if (
                GEMMFlag.DECOMPRESS_A_VIA_BITMASK in desc.flags
                and Datatype.BF16 == desc.datatype.ab
            ):
                raise NotImplementedError

            if (is_Af16_Bf16_gemm or is_Ai8_Bf16_gemm or is_Abf8_Bf16_gemm) and (
                use_f16_replacement_fma or use_f32_compute_with_f16_inp
            ):
                raise NotImplementedError

            if is_Ai8_Bbf16_gemm and not is_Ai8_Bbf16_gemm_bf16fma:
                raise NotImplementedError

            if n == n_blocking - 1:
                # handle trans B
                if GEMMFlag.TRANS_B in desc.flags:
                    b_offset = (
                        desc.ldb * micro_kernel_config.datatype_size_in2 * k_iters
                    )
                else:
                    k_pack_advance = (
                        (2 if is_Ai8_Bbf16_gemm_bf16fma else 1)
                        if is_Ai8_Bbf16_gemm
                        else k_pack_factor
                    )
                    b_offset = (
                        micro_kernel_config.datatype_size_in2 * k_pack_advance * k_iters
                    )

                if k == (k_iters - 1):
                    b = generated_code.current_val_by_reg[gp_reg_mapping.gp_reg_b]
                    b = generated_code.insert(
                        x86.ops.RI_AddOp(b, b_offset)
                    ).register_out

            for m in range(m_blocking):
                # post increment early
                if m == 0 and (n == n_blocking - 1) and k == 0:
                    k_pack_advance = (
                        (2 if is_Ai8_Bbf16_gemm_bf16fma else 1)
                        if is_Ai8_Bbf16_gemm
                        else k_pack_factor
                    )
                    a_adjust = 4 if is_Ai2_Bi8_gemm else 8 if is_Ai1_Bi8_gemm else 1

                    if GEMMFlag.DECOMPRESS_A_VIA_BITMASK in desc.flags:
                        raise NotImplementedError
                    else:
                        a = generated_code.current_val_by_reg[gp_reg_mapping.gp_reg_a]
                        a = generated_code.insert(
                            x86.ops.RI_AddOp(
                                a,
                                desc.lda
                                * micro_kernel_config.datatype_size_in
                                * k_pack_advance
                                // a_adjust,
                            )
                        ).register_out

                    # if we prefetch next A into L2, we need to also increment the prefetch pointer
                    if desc.prefetch == GEMMPrefetchType.AL2:
                        raise NotImplementedError

                # issue fma
                if Datatype.I8 == desc.datatype.ab:
                    raise NotImplementedError
                elif use_f16_replacement_fma:
                    raise NotImplementedError
                else:
                    match micro_kernel_config.vector_name:
                        case "x":
                            source_type = x86.registers.SSERegisterType
                        case "y":
                            source_type = x86.registers.AVX2RegisterType
                        case "z":
                            source_type = x86.registers.AVX512RegisterType
                    reg_src0 = source_type.from_index(
                        1 + m + vreg_ab_offset + k * m_blocking
                    )
                    reg_src1 = source_type.from_index(vreg_ab_offset)
                    reg_dst = source_type.from_index(
                        vec_reg_acc_start + m + (m_blocking * n)
                    )
                    src0 = generated_code.get_val(reg_src0)
                    src1 = generated_code.get_val(reg_src1)
                    dst = generated_code.get_val(reg_dst)

                    compxsmm_x86_instruction_vec_compute_3reg(
                        generated_code,
                        micro_kernel_config.vmul_instruction,
                        src0,
                        src1,
                        dst,
                    )
