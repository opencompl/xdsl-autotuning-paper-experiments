from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NamedTuple
from enum import IntEnum

from xdsl.dialects import x86
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.x86.registers import (
    GeneralRegisterType,
    UNALLOCATED_REG64,
    RDI,
    RSI,
    RDX,
)
from xdsl.ir import Block, Region
from xdsl.dialects.x86_func import FuncOp
from xdsl.rewriter import InsertPoint, Rewriter

LIBXSMM_X86_AVX512_MASK = 1  # this specifies k1
LIBXSMM_X86_VEC_REG_UNDEF = 255


def libxsmm_mmfunction_signature(module: ModuleOp, routine_name: str) -> FuncOp:
    operand_types = (RDI, RSI, RDX)

    func = FuncOp(
        routine_name,
        Region(Block(arg_types=operand_types)),
        (operand_types, ()),
        "public",
    )

    Rewriter.insert_op(func, InsertPoint.at_end(module.body.block))

    return func


@dataclass
class TileConfig:
    palette_id: int = 0
    tile0rowsb: int = 0
    tile0cols: int = 0
    tile1rowsb: int = 0
    tile1cols: int = 0
    tile2rowsb: int = 0
    tile2cols: int = 0
    tile3rowsb: int = 0
    tile3cols: int = 0
    tile4rowsb: int = 0
    tile4cols: int = 0
    tile5rowsb: int = 0
    tile5cols: int = 0
    tile6rowsb: int = 0
    tile6cols: int = 0
    tile7rowsb: int = 0
    tile7cols: int = 0


@dataclass
class LoopLabelTracker:
    """
    Structure for tracking local labels in assembly we do not allow overlapping loops
    libxsmm_loop_label_tracker

    We have to add more logic to track the SSA values corresponding to registers.
    """

    dest_blocks: list[Block] = field(default_factory=list)
    """
    Starting blocks of loops, and the index of the induction variable in the current arg
    list.
    """

    counter = 32

    @property
    def current_loop_number(self) -> int:
        # Instead of libxsmm's numbered labels and jumps forward/backwards, just allocate new labels
        # return len(self.dest_blocks) + 32
        self.counter += 1
        return self.counter


@dataclass
class BlockingInfo:
    tiles: int = 0
    sizes: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    blocking: int = 0
    block_size: int = 0


@dataclass
class MicroKernelConfig:
    """libxsmm_micro_kernel_config"""

    instruction_set: int = field(default=0)
    vector_reg_count: int = field(default=0)
    vector_length: int = field(default=0)
    datatype_size_in: int = field(default=0)
    datatype_size_in2: int = field(default=0)
    datatype_size_out: int = field(default=0)
    # TODO: add other instructions
    a_vmove_instruction: (
        type[
            x86.ops.DM_VmovapdOp
            | x86.ops.DM_VmovapsOp
            | x86.ops.DM_VmovupsOp
            | x86.ops.DM_VmovupdOp
        ]
        | None
    ) = field(default=None)
    b_vmove_instruction: (
        type[
            x86.ops.DM_VmovapdOp
            | x86.ops.DM_VmovapsOp
            | x86.ops.DM_VmovupsOp
            | x86.ops.DM_VmovupdOp
            | x86.ops.DM_VbroadcastsdOp
            | x86.ops.DM_VbroadcastssOp
        ]
        | None
    ) = field(default=None)
    b_shuff_instruction: int = field(default=0)
    c_vmove_ld_instruction: (
        type[
            x86.ops.DM_VmovapdOp
            | x86.ops.DM_VmovapsOp
            | x86.ops.DM_VmovupsOp
            | x86.ops.DM_VmovupdOp
        ]
        | None
    ) = field(default=None)
    c_vmove_st_instruction: (
        type[
            x86.ops.MS_VmovapdOp
            | x86.ops.MS_VmovupdOp
            | x86.ops.MS_VmovapsOp
            | x86.ops.MS_VmovupsOp
        ]
        | None
    ) = field(default=None)
    c_vmove_nts_instruction: (
        type[
            x86.ops.MS_VmovapdOp
            | x86.ops.MS_VmovupdOp
            | x86.ops.MS_VmovntpdOp
            | x86.ops.MS_VmovapsOp
            | x86.ops.MS_VmovupsOp
            | x86.ops.MS_VmovntpsOp
        ]
        | None
    ) = field(default=None)
    use_masking_a_c: bool = field(default=False)
    prefetch_instruction: int = field(default=0)
    vxor_instruction: type[x86.ops.DSS_VpxordOp] | None = field(default=None)
    vmul_instruction: (
        type[x86.ops.RSS_Vfmadd231pdOp | x86.ops.RSS_Vfmadd231psOp] | None
    ) = field(default=None)
    vadd_instruction: type[x86.ops.DSS_AddpdOp | x86.ops.DSS_AddpsOp] | None = field(
        default=None
    )
    alu_add_instruction: int = field(default=0)
    alu_sub_instruction: int = field(default=0)
    alu_cmp_instruction: int = field(default=0)
    alu_jmp_instruction: int = field(default=0)
    alu_mov_instruction: int = field(default=0)
    vector_name: Literal["x", "y", "z"] = field(default="z")

    # Auxiliary variables for GEMM fusion info
    fused_eltwise: int = field(default=0)
    m_loop_exists: int = field(default=0)
    n_loop_exists: int = field(default=0)
    fused_bcolbias: int = field(default=0)
    fused_hcolbias: int = field(default=0)
    fused_b8colbias: int = field(default=0)
    fused_h8colbias: int = field(default=0)
    fused_scolbias: int = field(default=0)
    fused_relu: int = field(default=0)
    fused_relu_nobitmask: int = field(default=0)
    fused_relu_bwd: int = field(default=0)
    fused_sigmoid: int = field(default=0)
    overwrite_C: int = field(default=0)
    vnni_format_C: int = field(default=0)
    sparsity_factor_A: int = field(default=0)
    decompress_A: int = field(default=0)
    vnni_cvt_output_ext_buf: int = field(default=0)
    norm_to_normT_B_ext_buf: int = field(default=0)
    has_colbias_act_fused: int = field(default=0)
    current_m: int = field(default=0)
    m_bitmask_advance: int = field(default=0)

    # Register names/logistics for fusion boo-keeping
    reserved_zmms: int = field(default=0)
    reserved_mask_regs: int = field(default=0)
    vnni_perm_reg: int = field(default=0)
    vnni_perm_reg2: int = field(default=0)
    zero_reg: int = field(default=0)
    scf_vreg: int = field(default=0)
    aux_vreg: int = field(default=0)
    vec_x2: int = field(default=0)
    vec_nom: int = field(default=0)
    vec_denom: int = field(default=0)
    vec_c0: int = field(default=0)
    vec_c1: int = field(default=0)
    vec_c2: int = field(default=0)
    vec_c3: int = field(default=0)
    vec_c1_d: int = field(default=0)
    vec_c2_d: int = field(default=0)
    vec_c3_d: int = field(default=0)
    vec_hi_bound: int = field(default=0)
    vec_lo_bound: int = field(default=0)
    vec_ones: int = field(default=0)
    vec_neg_ones: int = field(default=0)
    vec_halves: int = field(default=0)
    mask_hi: int = field(default=0)
    mask_lo: int = field(default=0)
    perm_table_vnni_lo: int = field(default=0)
    perm_table_vnni_hi: int = field(default=0)
    norm_to_normT_mask_reg_0: int = field(default=0)
    norm_to_normT_mask_reg_1: int = field(default=0)
    mask_m_fp32: int = field(default=0)
    mask_m_bf16: int = field(default=0)
    mask_m_lp_cvt: int = field(default=0)
    mask_k_lp_cvt: int = field(default=0)
    mask_m_lp_cvt_st: int = field(default=0)
    mask_k_lp_cvt_st: int = field(default=0)
    mask_lo_i4: int = field(default=0)
    mask_hi_i4: int = field(default=0)
    perm_table_zpt_bcast: int = field(default=0)
    luth_reg0: int = field(default=0)
    luth_reg1: int = field(default=0)
    lutl_reg0: int = field(default=0)
    lutl_reg1: int = field(default=0)
    sign_reg: int = field(default=0)
    blend_reg: int = field(default=0)
    tmp_reg0: int = field(default=0)
    tmp_reg1: int = field(default=0)

    # Auxiliary arrays for micro-kernel iteration space traversal
    use_paired_tilestores: int = field(default=0)
    m_tiles: int = field(default=0)
    n_tiles: int = field(default=0)
    _im: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    _in: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    _C_tile_id: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    _C_tile_mate_id: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    _im_offset_prefix_sums: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    _in_offset_prefix_sums: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    config_trans_a_tile: int = field(default=0)
    m_blocking_info: BlockingInfo = field(default_factory=lambda: BlockingInfo())

    tile_config: TileConfig = field(default_factory=TileConfig)
    gemm_scratch_ld: int = field(default=0)
    emulate_cvt2bf16fp32: int = field(default=0)
    emulate_cvt2bf16fp32_vperm: int = field(default=0)
    emulate_cvt2bf16fp32_vaux: int = field(default=0)
    emulate_cvt2bf16fp32_vaux0: int = field(default=0)
    emulate_cvt2bf16fp32_vaux1: int = field(default=0)
    mask_cvt_hi: int = field(default=0)
    mask_cvt_lo: int = field(default=0)
    io_loop_label_tracker: LoopLabelTracker = field(
        default_factory=lambda: LoopLabelTracker()
    )

    m_remainder: int = field(default=0)
    br_loop_index: int = field(default=0)
    cur_unroll_factor: int = field(default=0)
    is_peeled_br_loop: int = field(default=0)
    p_jump_label_tracker: LoopLabelTracker = field(
        default_factory=lambda: LoopLabelTracker()
    )
    loop_label_id: int = field(default=0)
    k_amx_microkernel: int = field(default=0)
    B_offs_trans: int = field(default=0)
    stride_b_trans: int = field(default=0)
    enforce_Mx1_amx_tile_blocking: int = field(default=0)

    use_custom_bf8_preproc: int = field(default=0)
    bf8_gemm_via_stack_alloc_tensors: int = field(default=0)
    hf8_gemm_via_stack_alloc_tensors: int = field(default=0)
    atrans_gemm_stack_alloc_tensors: int = field(default=0)
    avnni_gemm_stack_alloc_tensors: int = field(default=0)
    avnni_gemm_sw_pipeline: int = field(default=0)
    atvnni_gemm_stack_alloc_tensors: int = field(default=0)
    avnni_btrans_gemm_stack_alloc_tensors: int = field(default=0)
    atvnni_btrans_gemm_stack_alloc_tensors: int = field(default=0)
    bvnni_btrans_gemm_stack_alloc_tensors: int = field(default=0)


@dataclass
class GPRegMapping:
    """
    Structure for storing the current GP (General Purpose) register mapping.
    libxsmm_gp_reg_mapping
    """

    gp_reg_param_struct: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_a: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_a_base: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_b: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_b_base: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_c: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_a_prefetch: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_a_offset: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_b_prefetch: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_b_offset: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    # gp_reg_c_prefetch: int  # commented out in C code
    gp_reg_mloop: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_nloop: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_kloop: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_reduce_count: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_reduce_loop: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_a_ptrs: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_b_ptrs: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_lda: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_ldb: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_ldc: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_scf: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_zpt: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_help_0: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_help_1: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_help_2: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_help_3: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_help_4: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_help_5: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_help_6: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    # Auxiliary regs for sparsity in A support
    gp_reg_bitmap_a: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_decompressed_a: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_decompressed_elts: GeneralRegisterType = field(default=UNALLOCATED_REG64)
    gp_reg_popcnt: GeneralRegisterType = field(default=UNALLOCATED_REG64)


class Blocking(NamedTuple):
    range_1: int
    block_1: int
    range_2: int
    block_2: int
    ret: int


def libxsmm_compute_equalized_blocking(size: int, max_block: int) -> Blocking:
    size = max(size, 1)
    number_of_chunks = ((size - 1) // max_block) + 1
    modulo = size % number_of_chunks
    n2 = size // number_of_chunks
    n1 = n2 + 1
    N2 = 0
    N1 = 0
    chunk = 0
    ret = 0

    # ranges
    n1 = min(n1, max_block)
    for chunk in range(number_of_chunks):
        if chunk < modulo:
            N1 += n1
        else:
            N2 += n2

    # if we have perfect blocking, swap n2 and n1
    if modulo == 0:
        n1 = n2
        N1 = N2
        n2 = 0
        N2 = 0

    # some checks
    if (N1 % n1) != 0:
        ret = 1

    if n2 != 0:
        if N2 % n2 != 0:
            ret = 1

    return Blocking(N1, n1, N2, n2, ret)


class GEMMStackVar(IntEnum):
    """Auxiliary stack variables"""

    NONE = 0
    PFA_PTR = 1
    PFB_PTR = 2
    A_OFFS_BRGEMM_PTR = 3
    B_OFFS_BRGEMM_PTR = 4
    INT8_SCF = 5
    GEMM_SCRATCH_PTR = 6
    ELT_BIAS_PTR = 7
    ELT_OUTPUT_PTR = 8
    ARG_7 = 9
    ARG_8 = 10
    ARG_9 = 11
    ARG_10 = 12
    ELT_BUF1 = 13
    ELT_BUF2 = 14
    ELT_BITMAP_PTR = 15
    ELT_DECOMPRESS_BUF = 16
    TRANS_EXT_BUF_B = 17
    TRANS_EXT_BUF_C = 18
    ELT_RELU_BITMASK_PTR = 19
    BRCOUNT = 20
    TRANSPOSE_PTR = 21
    AVX2_MASK_PTR = 22
    SSE_AVX2_LP_HELPER_PTR = 23
    A_EMU_PTR = 24
    B_EMU_PTR = 25
    MELTW_STRUCT_PTR = 26
    A_SCRATCH_PTR = 27
    C_SCRATCH_PTR = 28
    C_OUTPUT_PTR = 29
    BIAS_SCRATCH_PTR = 30
    ZPT_PTR = 31
    AUX_VAR = 32
    MXSCALE_PTR = 33
    SCF_BRGEMM_PTR = 34
    ZPT_BRGEMM_PTR = 35
    BSCALE_PTR = 36
    BSCALE_BRGEMM_PTR = 37
    LDA_PTR = 38
    LDB_PTR = 39
    LDC_PTR = 40
