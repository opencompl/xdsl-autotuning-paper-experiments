from xdsl.dialects import x86
from xdsl.ir import SSAValue

from autotuner.libxsmm_gemm.generator_common import GPRegMapping, MicroKernelConfig
from autotuner.libxsmm_gemm.generator_x86_instructions import (
    libxsmm_x86_instruction_vec_compute_3reg,
    libxsmm_x86_instruction_vec_compute_mem_2reg,
    libxsmm_x86_instruction_vec_move_ld,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode, KLoopVals, VectorRegT
from autotuner.libxsmm_gemm.libxsmm_main import (
    GEMMDescriptor,
    GEMMFlag,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


def _vec_reg_type(vname: str) -> type[VectorRegT]:
    match vname:
        case "x":
            return x86.registers.SSERegisterType
        case "y":
            return x86.registers.AVX2RegisterType
        case "z":
            return x86.registers.AVX512RegisterType
        case _:
            assert False, f"Unsupported vector name: {vname}"


def libxsmm_generator_gemm_avx512_kloop_kernel(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    gemm_desc: GEMMDescriptor,
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
        and not is_Abf8_Bf16_gemm
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
        vals = libxsmm_generator_gemm_avx512_microkernel_fsdbcst(
            generated_code,
            gp_reg_mapping,
            micro_kernel_config,
            gemm_desc,
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
            generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_nofsdbcst

        for k in range(k_blocking):
            vals = generator_microkernel(
                generated_code,
                gp_reg_mapping,
                micro_kernel_config,
                gemm_desc,
                m_blocking,
                n_blocking,
                vals,
            )

    return vals


def libxsmm_generator_gemm_avx512_microkernel_fsdbcst(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    n_blocking: int,
    k_blocking: int,
    vals: KLoopVals,
) -> KLoopVals:
    vreg_ab_offset = 0
    k_pack_factor = 1
    k_iters = k_blocking

    if GEMMFlag.VNNI_A in desc.flags:
        raise NotImplementedError

    assert n_blocking <= 28

    if n_blocking >= 12:
        n_accs = 1
    elif n_blocking >= 6:
        n_accs = 2
    else:
        n_accs = 4
    if n_accs > k_iters:
        n_accs = k_iters
        n_accs = 1 if n_accs == 0 else n_accs

    vec_reg_count = micro_kernel_config.vector_reg_count
    a_vmove_instruction = micro_kernel_config.a_vmove_instruction
    assert a_vmove_instruction is not None
    assert micro_kernel_config.vxor_instruction is not None
    assert micro_kernel_config.vadd_instruction is not None
    assert micro_kernel_config.vmul_instruction is not None

    vname = micro_kernel_config.vector_name
    vec_type = _vec_reg_type(vname)

    a_val = vals.a
    b_val = vals.b

    # Accumulators (and the transient A vectors) are tracked explicitly by their vector
    # register index. The "main" accumulator set (index vec_reg_count - n_blocking + n)
    # is loaded by load_C and threaded through KLoopVals; the extra accumulator sets are
    # created here via vxor and reduced back into the main set at the end.
    acc_by_idx: dict[int, SSAValue[VectorRegT]] = {}
    # main accumulators come from load_C via KLoopVals (indexed by n, since m_blocking==1)
    for n in range(n_blocking):
        acc_by_idx[vec_reg_count - n_blocking + n] = vals.acc_vectors[n]

    a_by_idx: dict[int, SSAValue[VectorRegT]] = {}

    for k in range(1, n_accs):
        for n in range(n_blocking):
            reg_idx = vec_reg_count - (n_blocking * (k + 1)) + n
            reg = vec_type.from_index(reg_idx)
            acc_by_idx[reg_idx] = libxsmm_x86_instruction_vec_compute_3reg(
                generated_code,
                micro_kernel_config.vxor_instruction,
                reg,
                reg,
                reg,
                None,
                None,
                None,
            )

    for k in range(k_iters):
        if k == 0:
            reg_idx = vreg_ab_offset
            a_by_idx[reg_idx] = libxsmm_x86_instruction_vec_move_ld(
                generated_code,
                micro_kernel_config.instruction_set,
                a_vmove_instruction,
                a_val,
                None,
                0,
                desc.lda * k * micro_kernel_config.datatype_size_in * k_pack_factor,
                vec_type.from_index(reg_idx),
                vals.mask_k1 if micro_kernel_config.use_masking_a_c else None,
                True,
                False,
            )
            if k_iters > 1:
                reg_idx = 1 + vreg_ab_offset
                a_by_idx[reg_idx] = libxsmm_x86_instruction_vec_move_ld(
                    generated_code,
                    micro_kernel_config.instruction_set,
                    a_vmove_instruction,
                    a_val,
                    None,
                    0,
                    desc.lda
                    * (k + 1)
                    * micro_kernel_config.datatype_size_in
                    * k_pack_factor,
                    vec_type.from_index(reg_idx),
                    vals.mask_k1 if micro_kernel_config.use_masking_a_c else None,
                    True,
                    False,
                )
        elif k < (k_iters - 1):
            reg_idx = (k + 1) % 2 + vreg_ab_offset
            a_by_idx[reg_idx] = libxsmm_x86_instruction_vec_move_ld(
                generated_code,
                micro_kernel_config.instruction_set,
                a_vmove_instruction,
                a_val,
                None,
                0,
                desc.lda
                * (k + 1)
                * micro_kernel_config.datatype_size_in
                * k_pack_factor,
                vec_type.from_index(reg_idx),
                vals.mask_k1 if micro_kernel_config.use_masking_a_c else None,
                True,
                False,
            )

        if k == (k_iters - 1):
            a_val = generated_code.insert(
                x86.ops.RI_AddOp(
                    a_val,
                    k_blocking * micro_kernel_config.datatype_size_in * desc.lda,
                )
            ).register_out

        if GEMMFlag.TRANS_B not in desc.flags:
            for n in range(n_blocking):
                b_disp = (
                    k * micro_kernel_config.datatype_size_in2 * k_pack_factor
                    + n * desc.ldb * micro_kernel_config.datatype_size_in2
                )
                if desc.datatype.ab in (Datatype.F64, Datatype.F32):
                    src1_idx = k % 2 + vreg_ab_offset
                    dst_idx = vec_reg_count - (n_blocking * ((k % n_accs) + 1)) + n
                    acc_by_idx[dst_idx] = libxsmm_x86_instruction_vec_compute_mem_2reg(
                        generated_code,
                        micro_kernel_config.vmul_instruction,
                        b_val,
                        None,
                        0,
                        b_disp,
                        1,
                        vec_type.from_index(src1_idx),
                        vec_type.from_index(dst_idx),
                        a_by_idx[src1_idx],
                        acc_by_idx[dst_idx],
                    )
                else:
                    raise NotImplementedError
        else:
            for n in range(n_blocking):
                b_disp = (
                    k * desc.ldb * micro_kernel_config.datatype_size_in2 * k_pack_factor
                    + n * micro_kernel_config.datatype_size_in2
                )
                if desc.datatype.ab in (Datatype.F64, Datatype.F32):
                    src1_idx = k % 2 + vreg_ab_offset
                    dst_idx = vec_reg_count - (n_blocking * ((k % n_accs) + 1)) + n
                    acc_by_idx[dst_idx] = libxsmm_x86_instruction_vec_compute_mem_2reg(
                        generated_code,
                        micro_kernel_config.vmul_instruction,
                        b_val,
                        None,
                        0,
                        b_disp,
                        1,
                        vec_type.from_index(src1_idx),
                        vec_type.from_index(dst_idx),
                        a_by_idx[src1_idx],
                        acc_by_idx[dst_idx],
                    )
                else:
                    raise NotImplementedError

    if GEMMFlag.TRANS_B not in desc.flags:
        b_val = generated_code.insert(
            x86.ops.RI_AddOp(
                b_val,
                k_blocking * micro_kernel_config.datatype_size_in2,
            )
        ).register_out
    else:
        b_val = generated_code.insert(
            x86.ops.RI_AddOp(
                b_val,
                k_blocking * micro_kernel_config.datatype_size_in2 * desc.ldb,
            )
        ).register_out

    for k in range(1, n_accs):
        for n in range(n_blocking):
            if desc.datatype.ab in (Datatype.F64, Datatype.F32):
                src0_idx = vec_reg_count - (n_blocking * (k + 1)) + n
                main_idx = vec_reg_count - n_blocking + n
                acc_by_idx[main_idx] = libxsmm_x86_instruction_vec_compute_3reg(
                    generated_code,
                    micro_kernel_config.vadd_instruction,
                    vec_type.from_index(src0_idx),
                    vec_type.from_index(main_idx),
                    vec_type.from_index(main_idx),
                    acc_by_idx[src0_idx],
                    acc_by_idx[main_idx],
                    None,
                )
            else:
                raise NotImplementedError

    acc_vectors = tuple(
        acc_by_idx[vec_reg_count - n_blocking + n] for n in range(n_blocking)
    )

    return KLoopVals(
        a=a_val,
        b=b_val,
        c=vals.c,
        rbp=vals.rbp,
        rsp=vals.rsp,
        n_counter=vals.n_counter,
        m_counter=vals.m_counter,
        mask_k1=vals.mask_k1,
        acc_vectors=acc_vectors,
        k_counter=vals.k_counter,
    )


def libxsmm_generator_gemm_avx512_microkernel_nofsdbcst(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    m_blocking: int,
    n_blocking: int,
    vals: KLoopVals,
) -> KLoopVals:
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

    a_val = vals.a
    b_val = vals.b
    acc_vectors = list(vals.acc_vectors)

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

    # A vectors loaded upfront, indexed by m
    a_vec_vals: list[SSAValue[VectorRegT]] = []

    for m in range(m_blocking):
        # load column vectors of A upfront
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
        a_vec_type = _vec_reg_type(a_vname)

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
                    use_masking_a = micro_kernel_config.use_masking_a_c and (
                        m == (m_blocking - 1)
                    )
                    a_dest = a_vec_type.from_index(
                        m + vreg_ab_offset
                        if (is_Ai2_Bi8_gemm > 0)
                        else 1 + m + vreg_ab_offset
                    )
                    a_vec_vals.append(
                        libxsmm_x86_instruction_vec_move_ld(
                            generated_code,
                            micro_kernel_config.instruction_set,
                            a_vmove_instruction,
                            a_val,
                            None,
                            0,
                            (micro_kernel_config.datatype_size_in)
                            * (micro_kernel_config.vector_length)
                            * m
                            * k_pack_factor,
                            a_dest,
                            vals.mask_k1 if use_masking_a else None,
                            True,
                            False,
                        )
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

        if is_Ai4_Bi8_gemm:
            raise NotImplementedError
        elif is_Ai2_Bi8_gemm:
            raise NotImplementedError
        elif is_Ai1_Bi8_gemm:
            raise NotImplementedError
        else:
            # Process vreg A in case of f16/i8
            ...

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
            b_vec_type = _vec_reg_type(b_vname)
            b_vec_val = libxsmm_x86_instruction_vec_move_ld(
                generated_code,
                micro_kernel_config.instruction_set,
                b_vmove_instruction,
                b_val,
                x86.registers.UNALLOCATED_REG64,
                0,
                b_offset,
                b_vec_type.from_index(vreg_ab_offset),
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
                    b_val = generated_code.insert(
                        x86.ops.RI_AddOp(b_val, b_offset)
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
                        a_val = generated_code.insert(
                            x86.ops.RI_AddOp(
                                a_val,
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
                    vname = micro_kernel_config.vector_name
                    vec_type = _vec_reg_type(vname)
                    src0_idx = 1 + m + vreg_ab_offset + k * m_blocking
                    src1_idx = vreg_ab_offset
                    acc_idx = m + (m_blocking * n)
                    dst_idx = vec_reg_acc_start + acc_idx
                    acc_vectors[acc_idx] = libxsmm_x86_instruction_vec_compute_3reg(
                        generated_code,
                        micro_kernel_config.vmul_instruction,
                        vec_type.from_index(src0_idx),
                        vec_type.from_index(src1_idx),
                        vec_type.from_index(dst_idx),
                        a_vec_vals[m],
                        b_vec_val,
                        acc_vectors[acc_idx],
                    )

    return KLoopVals(
        a=a_val,
        b=b_val,
        c=vals.c,
        rbp=vals.rbp,
        rsp=vals.rsp,
        n_counter=vals.n_counter,
        m_counter=vals.m_counter,
        mask_k1=vals.mask_k1,
        acc_vectors=tuple(acc_vectors),
        k_counter=vals.k_counter,
    )
