"""LIBXSMM's AMX GEMM kernel (generator_gemm_amx.c and
generator_gemm_amx_microkernel.c), for BF16 A packed in K pairs (VNNI_A), flat
BF16 B and F32 or BF16 C, without transposes, fusion or batch-reduce, on SPR.

A BF16 C cannot move between memory and the F32 accumulator tiles directly: its
columns are widened to F32 with vector instructions into the GEMM scratch on the
stack, which the tiles load from, and the tiles are stored to the scratch, from
where vector instructions round their columns back to BF16 into C.

The C generator names registers and leaves it to the code it prints to keep track
of what each holds. Here every instruction defines SSA values, so
`AMXGeneratedCode` keeps the SSA value each register currently holds: an
instruction reads its operands from it and writes its results back, and a loop
threads all of them through the arguments of its body block.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from xdsl.builder import Builder
from xdsl.dialects import x86
from xdsl.dialects.x86.registers import (
    R8,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
    RBP,
    RCX,
    RDI,
    RDX,
    RSI,
    RSP,
    TMM,
    YMM,
    ZMM,
    AMXTileRegisterType,
    AVX512MaskRegisterType,
    GeneralRegisterType,
    X86RegisterType,
    X86VectorRegisterType,
)
from xdsl.dialects.x86_func import FuncOp
from xdsl.ir import Block, Operation, SSAValue
from xdsl.rewriter import InsertPoint

from autotuner import amx
from autotuner.libxsmm_gemm.generator_common import (
    BlockingInfo,
    GEMMStackVar,
    GPRegMapping,
    LoopLabelTracker,
    MicroKernelConfig,
    TileConfig,
)
from autotuner.libxsmm_gemm.generator_common_x86 import (
    libxsmm_generator_initialize_avx512_mask,
)
from autotuner.libxsmm_gemm.generator_gemm_common import (
    GEMM_STACK_VAR_OFFSETS,
    libxsmm_generator_gemm_destroy_stack_frame,
    libxsmm_generator_gemm_init_micro_kernel_config,
    libxsmm_generator_gemm_setup_stack_frame,
)
from autotuner.libxsmm_gemm.generator_x86_instructions import (
    libxsmm_x86_instruction_open_stream_gemm,
    libxsmm_x86_instruction_register_jump_back_label,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode
from autotuner.libxsmm_gemm.libxsmm_main import GEMMDescriptor, GEMMFlag
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype

# LIBXSMM reads these knobs from the environment
# (LIBXSMM_X86_AMX_GEMM_ENFORCE_Mx1_TILE_BLOCKING,
# LIBXSMM_X86_AMX_GEMM_PANEL_SW_PIPELINE_GRANULARITY and the tile load and
# prefetch ones); the port takes their defaults, under which none of them changes
# the kernel.
LIBXSMM_CPUID_X86_AMX_GEMM_ENFORCE_MX1_TILE_BLOCKING = 0

# libxsmm_cpuid_dot_pack_factor(BF16): two K values share a 32-bit lane.
BF16_K_PACK_FACTOR = 2


@dataclass
class AMXGeneratedCode(GeneratedCode):
    """`GeneratedCode` that also knows which SSA value each register holds."""

    routine_name: str = ""
    registers: dict[X86RegisterType, SSAValue] = field(
        default_factory=dict[X86RegisterType, SSAValue]
    )

    def emit(self, op: Operation) -> Operation:
        """Insert `op`; the registers it writes now hold its results."""
        self.insert(op)
        for result in op.results:
            if isinstance(result.type, X86RegisterType):
                self.registers[result.type] = result
        return op

    def reg(self, register: X86RegisterType) -> SSAValue:
        """The value `register` holds, read from the register itself if no
        instruction emitted here wrote it."""
        if register not in self.registers:
            match register:
                case GeneralRegisterType():
                    get = x86.ops.GetRegisterOp(register)
                case AMXTileRegisterType():
                    get = x86.ops.GetTileRegisterOp(register)
                case AVX512MaskRegisterType():
                    get = x86.ops.GetMaskRegisterOp(register)
                case X86VectorRegisterType():
                    get = x86.ops.GetAVXRegisterOp(register)
                case _:
                    raise NotImplementedError(register)
            self.emit(get)
        return self.registers[register]

    def gpr(self, register: GeneralRegisterType) -> SSAValue[GeneralRegisterType]:
        return SSAValue.get(self.reg(register), type=GeneralRegisterType)

    def tile(self, register: AMXTileRegisterType) -> SSAValue[AMXTileRegisterType]:
        return SSAValue.get(self.reg(register), type=AMXTileRegisterType)

    def forget_tiles(self) -> None:
        """`ldtilecfg` zeroes every tile, so no tile holds a value any more."""
        for register in list(self.registers):
            if isinstance(register, AMXTileRegisterType):
                del self.registers[register]


def _alu_imm(
    code: AMXGeneratedCode,
    instr: type[x86.ops.RI_AddOp | x86.ops.RI_SubOp],
    gp_reg: GeneralRegisterType,
    imm: int,
) -> None:
    """libxsmm_x86_instruction_alu_imm for add and sub."""
    code.emit(instr(code.reg(gp_reg), imm, register_out=gp_reg))


def _alu_mov_imm(code: AMXGeneratedCode, gp_reg: GeneralRegisterType, imm: int):
    """libxsmm_x86_instruction_alu_imm for mov."""
    code.emit(x86.ops.DI_MovOp(imm, destination=gp_reg))


def _push_reg(code: AMXGeneratedCode, gp_reg: GeneralRegisterType) -> None:
    code.emit(x86.ops.S_PushOp(code.reg(RSP), code.reg(gp_reg)))


def _pop_reg(code: AMXGeneratedCode, gp_reg: GeneralRegisterType) -> None:
    code.emit(x86.ops.D_PopOp(code.reg(RSP), destination=gp_reg))


def _register_jump_back_label(
    code: AMXGeneratedCode, loop_label_tracker: LoopLabelTracker
) -> None:
    """libxsmm_x86_instruction_register_jump_back_label, which here also opens the
    loop body: a block taking every register as an argument."""
    current = code.current_block
    region = current.parent
    assert region is not None and current.next_block is None
    live = tuple(code.registers.values())
    body = Block(arg_types=tuple(value.type for value in live))
    region.insert_block_after(body, current)
    code.insert(x86.ops.FallthroughOp(live, body))
    code.builder.insertion_point = InsertPoint.at_end(body)
    code.registers = {}
    for arg in body.args:
        assert isinstance(arg.type, X86RegisterType)
        code.registers[arg.type] = arg
    loop_label_tracker.dest_blocks.append(body)
    libxsmm_x86_instruction_register_jump_back_label(code, loop_label_tracker)


def _jump_back_to_label(
    code: AMXGeneratedCode,
    cmp_op: Operation,
    loop_label_tracker: LoopLabelTracker,
) -> None:
    """libxsmm_x86_instruction_jump_back_to_label with jl, passing the loop body
    the registers it takes."""
    body = loop_label_tracker.dest_blocks.pop()
    current = code.current_block
    region = current.parent
    assert region is not None and current.next_block is None
    carried = tuple(
        code.registers[register]
        for register in body.arg_types
        if isinstance(register, X86RegisterType)
    )
    assert len(carried) == len(body.arg_types)
    fallthrough = Block(arg_types=body.arg_types)
    region.insert_block_after(fallthrough, current)
    code.insert(x86.ops.C_JlOp(cmp_op, carried, carried, body, fallthrough))
    code.builder.insertion_point = InsertPoint.at_end(fallthrough)
    for arg in fallthrough.args:
        assert isinstance(arg.type, X86RegisterType)
        code.registers[arg.type] = arg


def _cmp_imm(code: AMXGeneratedCode, gp_reg: GeneralRegisterType, imm: int):
    return code.emit(x86.ops.SI_CmpOp(code.reg(gp_reg), imm))


def _getval_stack_var(
    code: AMXGeneratedCode, stack_var: GEMMStackVar, gp_reg: GeneralRegisterType
) -> None:
    """libxsmm_generator_gemm_getval_stack_var."""
    code.emit(
        x86.ops.DM_MovOp(
            code.reg(RBP), GEMM_STACK_VAR_OFFSETS[stack_var], destination=gp_reg
        )
    )


def _vec_load(
    code: AMXGeneratedCode,
    instr: type[x86.ops.DM_VmovupsOp | x86.ops.DM_Vmovdqu16Op],
    masked_instr: type[x86.ops.DMK_VmovupsOp | x86.ops.DMK_Vmovdqu16Op],
    gp_reg_base: GeneralRegisterType,
    displacement: int,
    vreg: X86VectorRegisterType,
    mask_id: int,
) -> None:
    """libxsmm_x86_instruction_vec_move of a load, zero-masked by k`mask_id`
    unless it is 0."""
    if mask_id:
        code.emit(
            masked_instr(
                code.reg(gp_reg_base),
                code.reg(x86.registers.K[mask_id]),
                displacement,
                z=True,
                destination=vreg,
            )
        )
    else:
        code.emit(instr(code.reg(gp_reg_base), displacement, destination=vreg))


def _vec_store(
    code: AMXGeneratedCode,
    instr: type[x86.ops.MS_VmovupsOp | x86.ops.MS_Vmovdqu16Op],
    masked_instr: type[x86.ops.MSK_VmovupsOp | x86.ops.MSK_Vmovdqu16Op],
    gp_reg_base: GeneralRegisterType,
    displacement: int,
    vreg: X86VectorRegisterType,
    mask_id: int,
) -> None:
    """libxsmm_x86_instruction_vec_move of a store, merge-masked by k`mask_id`
    unless it is 0."""
    if mask_id:
        code.emit(
            masked_instr(
                code.reg(gp_reg_base),
                code.reg(vreg),
                code.reg(x86.registers.K[mask_id]),
                displacement,
            )
        )
    else:
        code.emit(instr(code.reg(gp_reg_base), code.reg(vreg), displacement))


def _c_dtype_size(desc: GEMMDescriptor) -> int:
    """The bytes of an element of C: F32 or BF16."""
    return 2 if desc.datatype.c == Datatype.BF16 else 4


def _tile_config_label(code: AMXGeneratedCode, config_id: int) -> str:
    return f".L{code.routine_name}_tileconfig{config_id}"


def _ldtilecfg(code: AMXGeneratedCode, config_id: int, tile_config: TileConfig):
    """libxsmm_x86_instruction_tile_control with LDTILECFG and a configuration.

    LIBXSMM's text emitter jumps over the 64 bytes, which it places in the code
    under `.data_tconf_<id>`; `amx.load_tile_config` keeps them in `.rodata`
    instead, under a label of this function numbered by the same id.
    """
    tiles = tuple(
        amx.TileShape(rows=cols, bytes_per_row=rowsb) if rowsb and cols else None
        for rowsb, cols in (
            (
                getattr(tile_config, f"tile{tile}rowsb"),
                getattr(tile_config, f"tile{tile}cols"),
            )
            for tile in range(amx.TILE_REGISTERS)
        )
    )
    assert tile_config.palette_id == 1
    amx.load_tile_config(
        code.emit, amx.TileConfig(tiles), _tile_config_label(code, config_id)
    )
    code.forget_tiles()


def libxsmm_generator_gemm_amx_kernel_wrapper(
    func_op: FuncOp, arch: Arch, desc: GEMMDescriptor
) -> None:
    loop_label_tracker = LoopLabelTracker()
    gp_reg_mapping = GPRegMapping()

    # define gp register mapping
    gp_reg_mapping.gp_reg_param_struct = RDI
    gp_reg_mapping.gp_reg_a = gp_reg_mapping.gp_reg_param_struct
    gp_reg_mapping.gp_reg_b = RSI
    gp_reg_mapping.gp_reg_c = RDX
    gp_reg_mapping.gp_reg_a_prefetch = RCX
    gp_reg_mapping.gp_reg_b_prefetch = R8
    # gp_reg_scf stays undefined: there is no scaling factor for BF16

    # If we are generating the batchreduce kernel, then we rename the registers
    if (
        GEMMFlag.BATCH_REDUCE_ADDRESS in desc.flags
        or GEMMFlag.BATCH_REDUCE_OFFSET in desc.flags
        or GEMMFlag.BATCH_REDUCE_STRIDE in desc.flags
    ):
        raise NotImplementedError
    gp_reg_mapping.gp_reg_mloop = R12
    gp_reg_mapping.gp_reg_nloop = R13
    gp_reg_mapping.gp_reg_help_0 = R12
    gp_reg_mapping.gp_reg_help_1 = R13
    gp_reg_mapping.gp_reg_kloop = R14
    gp_reg_mapping.gp_reg_lda = R11
    gp_reg_mapping.gp_reg_ldb = R15
    gp_reg_mapping.gp_reg_ldc = R14

    code = AMXGeneratedCode(
        Builder(InsertPoint.at_end(func_op.body.block)),
        arch,
        routine_name=func_op.sym_name.data,
    )
    for arg in func_op.body.block.args:
        assert isinstance(arg.type, X86RegisterType)
        code.registers[arg.type] = arg

    # open asm; the classic ABI takes A, B and C instead of a struct
    if GEMMFlag.USE_XGEMM_ABI in desc.flags or GEMMFlag.USE_XGEMM_EXT_ABI in desc.flags:
        raise NotImplementedError
    libxsmm_x86_instruction_open_stream_gemm(code, gp_reg_mapping, False, desc.prefetch)

    # Only kernels that set up the tile configuration themselves and restore the
    # caller's afterwards are ported.
    if (
        GEMMFlag.NO_RESET_TILECONFIG in desc.flags
        or GEMMFlag.NO_SETUP_TILECONFIG in desc.flags
    ):
        raise NotImplementedError

    # saving current tileconfig to the stack
    _alu_imm(code, x86.ops.RI_SubOp, RSP, 64)
    code.emit(x86.ops.M_SttilecfgOp(code.reg(RSP), 0))

    # call Intel AMX kernel
    libxsmm_generator_gemm_amx_kernel(code, loop_label_tracker, gp_reg_mapping, desc)

    # restoring current tileconfig to the stack
    code.emit(x86.ops.M_LdtilecfgOp(code.reg(RSP), 0))
    _alu_imm(code, x86.ops.RI_AddOp, RSP, 64)

    # In C, the stream is closed with the inline assembly register string, but we
    # don't need to do this


def libxsmm_generator_gemm_amx_kernel(
    code: AMXGeneratedCode,
    loop_label_tracker: LoopLabelTracker,
    gp_reg_mapping: GPRegMapping,
    desc: GEMMDescriptor,
) -> None:
    micro_kernel_config = MicroKernelConfig()
    m0 = 0
    m1 = 0
    l_enforce_Mx1_amx_tile_blocking = (
        1
        if LIBXSMM_CPUID_X86_AMX_GEMM_ENFORCE_MX1_TILE_BLOCKING > 0
        else 1
        if desc.n <= 16
        else 0
    )
    ldb_adjustment = 1

    # Of LIBXSMM's AMX kernels only BF16 A in K pairs with flat BF16 B and F32 or
    # BF16 C is ported: the others first convert A or B in a software pipeline
    # (flat or transposed A, transposed B, F32 or FP8 inputs), compute in other
    # precisions, store C in VNNI layout, or fuse eltwise operations into the
    # store of C.
    if not (
        desc.datatype.ab == Datatype.BF16
        and desc.datatype.comp == Datatype.F32
        and desc.datatype.c in (Datatype.F32, Datatype.BF16)
        and GEMMFlag.VNNI_A in desc.flags
        and not desc.flags & (GEMMFlag.TRANS_A | GEMMFlag.TRANS_B | GEMMFlag.VNNI_B)
        and GEMMFlag.VNNI_C not in desc.flags
        and GEMMFlag.DECOMPRESS_A_VIA_BITMASK not in desc.flags
    ):
        raise NotImplementedError

    # define the micro kernel code gen properties
    libxsmm_generator_gemm_init_micro_kernel_config(
        micro_kernel_config, code.arch, desc, False
    )
    micro_kernel_config.enforce_Mx1_amx_tile_blocking = l_enforce_Mx1_amx_tile_blocking

    # First stamp out a nice GEMM and the if need be take care of remainder handling
    if desc.m % 16 == 0 or desc.m <= 32 or l_enforce_Mx1_amx_tile_blocking > 0:
        if l_enforce_Mx1_amx_tile_blocking > 0 and desc.m > 32:
            if (
                desc.m % 16 == 0
                or desc.m % 32 == 0
                or desc.m % 48 == 0
                or desc.m % 64 == 0
                or desc.m <= 64
            ):
                n_gemm_code_blocks = 1
            else:
                m0 = desc.m - (desc.m % 64)
                m1 = desc.m - m0
                desc.m = m0
                n_gemm_code_blocks = 2
        else:
            n_gemm_code_blocks = 1
    else:
        # Need to stamp out a remainder handling gemm
        m0 = desc.m - (desc.m % 32)
        m1 = desc.m - m0
        desc.m = m0
        n_gemm_code_blocks = 2

    # Here compute the 2D blocking info based on the M and N values
    m_blocking_info = [BlockingInfo(), BlockingInfo()]
    n_blocking_info = [BlockingInfo(), BlockingInfo()]
    tile_config = TileConfig()
    libxsmm_generator_gemm_init_micro_kernel_config_tileblocking(
        desc, micro_kernel_config, m_blocking_info, n_blocking_info, tile_config
    )

    # Setup stack frame...
    micro_kernel_config.m_tiles = m_blocking_info[0].tiles
    micro_kernel_config.n_tiles = n_blocking_info[0].tiles
    micro_kernel_config.m_blocking_info = m_blocking_info[0]

    rbp_val, rsp_val = libxsmm_generator_gemm_setup_stack_frame(
        code,
        desc,
        gp_reg_mapping,
        micro_kernel_config,
        rsp_val=code.gpr(RSP),
        locals_size=240,
    )
    code.registers[rbp_val.type] = rbp_val
    code.registers[rsp_val.type] = rsp_val

    libxsmm_generator_gemm_amx_setup_fusion_infra(code, desc, micro_kernel_config)
    libxsmm_generator_gemm_amx_setup_masking_infra(code, micro_kernel_config)

    _ldtilecfg(code, 0, tile_config)

    # Set the LD registers for A and B matrices
    _alu_mov_imm(code, gp_reg_mapping.gp_reg_lda, (desc.lda * 4) // 4)
    _alu_mov_imm(
        code,
        gp_reg_mapping.gp_reg_ldb,
        (desc.ldb * ldb_adjustment * micro_kernel_config.datatype_size_in2) // 4,
    )
    _alu_mov_imm(code, gp_reg_mapping.gp_reg_ldc, (desc.ldc * 4) // 4)

    libxsmm_generator_gemm_amx_kernel_nloop(
        code,
        loop_label_tracker,
        gp_reg_mapping,
        micro_kernel_config,
        desc,
        n_blocking_info,
        m_blocking_info,
    )

    if n_gemm_code_blocks > 1:
        desc.m = m1
        micro_kernel_config.m_remainder = 0

        # Here compute the 2D blocking info based on the M and N values
        libxsmm_generator_gemm_init_micro_kernel_config_tileblocking(
            desc, micro_kernel_config, m_blocking_info, n_blocking_info, tile_config
        )
        micro_kernel_config.m_blocking_info = m_blocking_info[0]
        libxsmm_generator_gemm_amx_setup_masking_infra(code, micro_kernel_config)

        # LIBXSMM numbers this configuration 0 like the first one; the build
        # patches it to 1, so that the text emitter labels it apart.
        _ldtilecfg(code, 1, tile_config)

        if micro_kernel_config.n_loop_exists > 0:
            # We should adjust n advancements in C/B etc. Also advance by M since all
            # M adjustments have been revoked by the n loop footer
            libxsmm_generator_gemm_amx_adjust_n_advancement(
                code, desc, gp_reg_mapping, micro_kernel_config, desc.n * -1
            )
            libxsmm_generator_gemm_amx_adjust_m_advancement(
                code, desc, gp_reg_mapping, micro_kernel_config, m0
            )
        elif micro_kernel_config.m_loop_exists == 0:
            # We should advance by M since no advancements have been made
            libxsmm_generator_gemm_amx_adjust_m_advancement(
                code, desc, gp_reg_mapping, micro_kernel_config, m0
            )
        else:
            # Nothing should be done since the M loop exists and has made the proper
            # advancements in the M direction
            pass
        _alu_mov_imm(code, gp_reg_mapping.gp_reg_lda, (desc.lda * 4) // 4)
        _alu_mov_imm(
            code,
            gp_reg_mapping.gp_reg_ldb,
            (desc.ldb * ldb_adjustment * micro_kernel_config.datatype_size_in2) // 4,
        )
        _alu_mov_imm(code, gp_reg_mapping.gp_reg_ldc, (desc.ldc * 4) // 4)

        libxsmm_generator_gemm_amx_kernel_nloop(
            code,
            loop_label_tracker,
            gp_reg_mapping,
            micro_kernel_config,
            desc,
            n_blocking_info,
            m_blocking_info,
        )

    # Properly destroy stack frame...
    rbp_val, rsp_val = libxsmm_generator_gemm_destroy_stack_frame(
        code, desc, gp_reg_mapping, micro_kernel_config, code.gpr(x86.registers.RBP)
    )
    code.registers[rbp_val.type] = rbp_val
    code.registers[rsp_val.type] = rsp_val


def libxsmm_setup_tile(tile_id: int, n_rows: int, n_cols: int, tc: TileConfig):
    """Configure tile `tile_id` as `n_cols` rows of `n_rows` 32-bit values: LIBXSMM
    names the dimensions after the column-major matrices the tiles hold."""
    assert 0 <= tile_id < 8, "valid tile id"
    setattr(tc, f"tile{tile_id}rowsb", n_rows * 4)
    setattr(tc, f"tile{tile_id}cols", n_cols)


def libxsmm_generator_gemm_init_micro_kernel_config_tileblocking(
    desc: GEMMDescriptor,
    micro_kernel_config: MicroKernelConfig,
    m_blocking_info: list[BlockingInfo],
    n_blocking_info: list[BlockingInfo],
    tile_config: TileConfig,
) -> None:
    l_enforce_Mx1_amx_tile_blocking = micro_kernel_config.enforce_Mx1_amx_tile_blocking
    l_k_pack_factor = BF16_K_PACK_FACTOR
    micro_kernel_config.config_trans_a_tile = 0

    tile_config.palette_id = 1

    # Find K blocking
    k_blocking = 32
    while desc.k % k_blocking != 0:
        k_blocking -= l_k_pack_factor

    m_info = m_blocking_info[0]
    n_info = n_blocking_info[0]
    if l_enforce_Mx1_amx_tile_blocking > 0:
        micro_kernel_config.m_remainder = 0
        m_blocking = 64
        while desc.m % m_blocking != 0:
            m_blocking -= 1
        m_info.blocking = m_blocking
        m_info.block_size = desc.m
        if m_blocking <= 16:
            m_info.tiles = 1
            m_info.sizes[0] = m_blocking
        elif m_blocking <= 32:
            m_info.tiles = 2
            m_info.sizes[0] = 16
            m_info.sizes[1] = m_blocking - m_info.sizes[0]
        elif m_blocking <= 48:
            m_info.tiles = 3
            m_info.sizes[0] = 16
            m_info.sizes[1] = 16
            m_info.sizes[2] = m_blocking - 2 * 16
        else:
            m_info.tiles = 4
            m_info.sizes[0] = 16
            m_info.sizes[1] = 16
            m_info.sizes[2] = 16
            m_info.sizes[3] = m_blocking - 3 * 16
        micro_kernel_config.m_remainder = m_info.sizes[m_info.tiles - 1] % 16

        n_blocking = 16
        while desc.n % n_blocking != 0:
            n_blocking -= 1
        n_info.blocking = n_blocking
        n_info.block_size = desc.n
        n_info.tiles = 1
        n_info.sizes[0] = n_blocking
    else:
        micro_kernel_config.m_remainder = 0
        m_blocking = 32
        while desc.m % m_blocking != 0:
            m_blocking -= 1

        m_info.blocking = m_blocking
        m_info.block_size = desc.m
        if m_blocking <= 16:
            m_info.tiles = 1
            m_info.sizes[0] = m_blocking
            micro_kernel_config.m_remainder = m_info.sizes[0] % 16
        else:
            m_info.tiles = 2
            m_info.sizes[0] = 16
            m_info.sizes[1] = m_blocking - m_info.sizes[0]
            micro_kernel_config.m_remainder = m_info.sizes[1] % 16

        n_blocking = 32
        while desc.n % n_blocking != 0:
            n_blocking -= 1
        n_info.blocking = n_blocking
        n_info.block_size = desc.n
        if n_blocking <= 16:
            n_info.tiles = 1
            n_info.sizes[0] = n_blocking
        else:
            n_info.tiles = 2
            n_info.sizes[0] = (n_blocking + 1) // 2
            n_info.sizes[1] = n_blocking - n_info.sizes[0]

    # First init all tiles with default value 16
    for im in range(8):
        libxsmm_setup_tile(im, 16, 16, tile_config)

    # First configure the accumulator tiles 0-4
    m_tiles = m_info.tiles
    n_tiles = n_info.tiles
    ii = 0
    for im in range(m_tiles):
        for in_ in range(n_tiles):
            libxsmm_setup_tile(ii, m_info.sizes[im], n_info.sizes[in_], tile_config)
            ii += 1
            if n_tiles == 1 and l_enforce_Mx1_amx_tile_blocking == 0:
                ii += 1
    # Configure tiles for A
    libxsmm_setup_tile(4, m_info.sizes[0], k_blocking // l_k_pack_factor, tile_config)
    if m_tiles in (2, 3, 4):
        libxsmm_setup_tile(
            5, m_info.sizes[m_tiles - 1], k_blocking // l_k_pack_factor, tile_config
        )

    # Configure tiles for B
    libxsmm_setup_tile(6, k_blocking // l_k_pack_factor, n_info.sizes[0], tile_config)
    if n_tiles == 2:
        libxsmm_setup_tile(
            7, k_blocking // l_k_pack_factor, n_info.sizes[1], tile_config
        )
    micro_kernel_config.tile_config = tile_config


def libxsmm_generator_gemm_amx_setup_fusion_infra(
    code: AMXGeneratedCode, desc: GEMMDescriptor, micro_kernel_config: MicroKernelConfig
) -> None:
    """Without fusion this only reserves k0 (which no instruction can use as a
    write mask) and no vector registers."""
    reserved_zmms = 0
    reserved_mask_regs = 1
    micro_kernel_config.reserved_zmms = reserved_zmms
    micro_kernel_config.reserved_mask_regs = reserved_mask_regs


def libxsmm_generator_gemm_amx_setup_masking_infra(
    code: AMXGeneratedCode, micro_kernel_config: MicroKernelConfig
) -> None:
    """Masks for the last 16 rows of M if they are partial: in F32 for a column of
    a tile, and in BF16 for the column of a pair of tiles that a BF16 C stores
    at once.

    An F32 C of an AMX kernel moves only with tile loads and stores, so it never
    reads either mask; LIBXSMM sets them up all the same.
    """
    reserved_mask_regs = micro_kernel_config.reserved_mask_regs
    temp_reg = R10
    micro_kernel_config.mask_m_fp32 = 0
    micro_kernel_config.mask_m_bf16 = 0

    if micro_kernel_config.m_remainder > 0:
        reserved_mask_regs += 2
        micro_kernel_config.mask_m_fp32 = reserved_mask_regs - 1
        micro_kernel_config.mask_m_bf16 = reserved_mask_regs - 2
        for mask, datatype in (
            (micro_kernel_config.mask_m_fp32, Datatype.F32),
            (micro_kernel_config.mask_m_bf16, Datatype.BF16),
        ):
            mask_val = libxsmm_generator_initialize_avx512_mask(
                code,
                temp_reg,
                AVX512MaskRegisterType.from_index(mask),
                16 - micro_kernel_config.m_remainder,
                datatype,
            )
            code.registers[mask_val.type] = mask_val
    micro_kernel_config.reserved_mask_regs = reserved_mask_regs


def libxsmm_generator_gemm_amx_adjust_m_advancement(
    code: AMXGeneratedCode,
    desc: GEMMDescriptor,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    m_adjustment: int,
) -> None:
    # A holds the K pairs of a row of M in 4 bytes
    a_vnni_adjustment = 1
    l_a_packed_bytes = 4

    # Adjust C pointer
    _alu_imm(
        code,
        x86.ops.RI_AddOp,
        gp_reg_mapping.gp_reg_c,
        m_adjustment * _c_dtype_size(desc),
    )

    # advance A pointer
    _alu_imm(
        code,
        x86.ops.RI_AddOp,
        gp_reg_mapping.gp_reg_a,
        (m_adjustment * l_a_packed_bytes // a_vnni_adjustment)
        // micro_kernel_config.sparsity_factor_A,
    )


def libxsmm_generator_gemm_amx_adjust_n_advancement(
    code: AMXGeneratedCode,
    desc: GEMMDescriptor,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    n_adjustment: int,
) -> None:
    # Adjust C pointer
    l_c_dtype_size = _c_dtype_size(desc)
    _alu_imm(
        code,
        x86.ops.RI_AddOp,
        gp_reg_mapping.gp_reg_c,
        n_adjustment * desc.ldc * l_c_dtype_size,
    )

    # handle trans B
    l_b_src_bytes = micro_kernel_config.datatype_size_in2
    l_b_offset = n_adjustment * desc.ldb * l_b_src_bytes
    _alu_imm(code, x86.ops.RI_AddOp, gp_reg_mapping.gp_reg_b, l_b_offset)


def libxsmm_generator_gemm_header_nloop_amx(
    code: AMXGeneratedCode,
    loop_label_tracker: LoopLabelTracker,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    n_blocking: int,
) -> None:
    _alu_mov_imm(code, gp_reg_mapping.gp_reg_nloop, 0)
    _register_jump_back_label(code, loop_label_tracker)
    _alu_imm(code, x86.ops.RI_AddOp, gp_reg_mapping.gp_reg_nloop, n_blocking)


def libxsmm_generator_gemm_header_mloop_amx(
    code: AMXGeneratedCode,
    loop_label_tracker: LoopLabelTracker,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    m_blocking: int,
) -> None:
    _alu_mov_imm(code, gp_reg_mapping.gp_reg_mloop, 0)
    _register_jump_back_label(code, loop_label_tracker)
    _alu_imm(code, x86.ops.RI_AddOp, gp_reg_mapping.gp_reg_mloop, m_blocking)


def libxsmm_generator_gemm_footer_nloop_amx(
    code: AMXGeneratedCode,
    loop_label_tracker: LoopLabelTracker,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    n_blocking: int,
    n_done: int,
    m_loop_exists: int,
) -> None:
    a_vnni_adjustment = 1
    l_a_packed_bytes = 4
    l_c_dtype_size = _c_dtype_size(desc)
    vnni_c_adjustment = 1

    _alu_imm(
        code,
        x86.ops.RI_AddOp,
        gp_reg_mapping.gp_reg_c,
        (n_blocking * desc.ldc * l_c_dtype_size)
        - (desc.m * m_loop_exists * l_c_dtype_size * vnni_c_adjustment),
    )

    # handle trans B
    l_b_src_bytes = micro_kernel_config.datatype_size_in2
    l_b_offset = n_blocking * desc.ldb * l_b_src_bytes
    _alu_imm(code, x86.ops.RI_AddOp, gp_reg_mapping.gp_reg_b, l_b_offset)

    if m_loop_exists:
        _alu_imm(
            code,
            x86.ops.RI_SubOp,
            gp_reg_mapping.gp_reg_a,
            (desc.m * l_a_packed_bytes // a_vnni_adjustment)
            // micro_kernel_config.sparsity_factor_A,
        )
    cmp_op = _cmp_imm(code, gp_reg_mapping.gp_reg_nloop, n_done)
    _jump_back_to_label(code, cmp_op, loop_label_tracker)


def libxsmm_generator_gemm_footer_mloop_amx(
    code: AMXGeneratedCode,
    loop_label_tracker: LoopLabelTracker,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    m_blocking: int,
    m_done: int,
    k_unrolled: int,
) -> None:
    a_vnni_adjustment = 1
    l_a_packed_bytes = 4

    # advance C pointer
    _alu_imm(
        code,
        x86.ops.RI_AddOp,
        gp_reg_mapping.gp_reg_c,
        m_blocking * _c_dtype_size(desc),
    )

    if k_unrolled == 0:
        # advance A pointer
        _alu_imm(
            code,
            x86.ops.RI_SubOp,
            gp_reg_mapping.gp_reg_a,
            (desc.k * micro_kernel_config.datatype_size_in * desc.lda)
            - (m_blocking * l_a_packed_bytes // a_vnni_adjustment),
        )
    else:
        # advance A pointer
        _alu_imm(
            code,
            x86.ops.RI_AddOp,
            gp_reg_mapping.gp_reg_a,
            (m_blocking * l_a_packed_bytes // a_vnni_adjustment)
            // micro_kernel_config.sparsity_factor_A,
        )

    # loop handling
    cmp_op = _cmp_imm(code, gp_reg_mapping.gp_reg_mloop, m_done)
    _jump_back_to_label(code, cmp_op, loop_label_tracker)


def libxsmm_generator_gemm_load_C_amx(
    code: AMXGeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    n_blocking_info: BlockingInfo,
    m_blocking_info: BlockingInfo,
) -> None:
    acc_id = 0
    m_tiles = m_blocking_info.tiles
    n_tiles = n_blocking_info.tiles
    l_enforce_Mx1_amx_tile_blocking = micro_kernel_config.enforce_Mx1_amx_tile_blocking

    if GEMMFlag.BETA_0 not in desc.flags and desc.datatype.c == Datatype.BF16:
        # Beta=1
        gp_reg_gemm_scratch = (
            gp_reg_mapping.gp_reg_help_1
            if micro_kernel_config.n_loop_exists == 0
            else gp_reg_mapping.gp_reg_help_0
        )
        # Check if we have to save the tmp registers
        if (
            gp_reg_gemm_scratch == gp_reg_mapping.gp_reg_help_0
            and micro_kernel_config.m_loop_exists == 1
        ):
            _push_reg(code, gp_reg_mapping.gp_reg_help_0)
        _alu_mov_imm(
            code,
            gp_reg_mapping.gp_reg_ldc,
            (micro_kernel_config.gemm_scratch_ld * 4) // 4,
        )
        _getval_stack_var(code, GEMMStackVar.GEMM_SCRATCH_PTR, gp_reg_gemm_scratch)
        i_m_offset = 0
        for im in range(m_tiles):
            mask_id = micro_kernel_config.mask_m_fp32 if im == m_tiles - 1 else 0
            i_n_offset = 0
            for in_ in range(n_tiles):
                # Now for all the columns in the tile, upconvert them to F32 from
                # BF16
                for col in range(n_blocking_info.sizes[in_]):
                    # we do mod 4 as are otherwise running out ymms
                    zmm_reg = (col % 4) + micro_kernel_config.reserved_zmms

                    # load 16 bit values into ymm portion of the register
                    _vec_load(
                        code,
                        x86.ops.DM_Vmovdqu16Op,
                        x86.ops.DMK_Vmovdqu16Op,
                        gp_reg_mapping.gp_reg_c,
                        ((i_n_offset + col) * desc.ldc + i_m_offset)
                        * micro_kernel_config.datatype_size_out,
                        YMM[zmm_reg],
                        mask_id,
                    )
                    libxsmm_generator_cvtbf16ps_avx512(code, zmm_reg)
                    # Store upconverted column to GEMM scratch
                    _vec_store(
                        code,
                        x86.ops.MS_VmovupsOp,
                        x86.ops.MSK_VmovupsOp,
                        gp_reg_gemm_scratch,
                        (
                            (i_n_offset + col) * micro_kernel_config.gemm_scratch_ld
                            + i_m_offset
                        )
                        * 4,
                        ZMM[zmm_reg],
                        mask_id,
                    )
                # Move zmm registers stored in GEMM scratch to the proper tile
                code.emit(
                    x86.ops.DMX_TileloaddOp(
                        code.reg(gp_reg_gemm_scratch),
                        code.reg(gp_reg_mapping.gp_reg_ldc),
                        (i_n_offset * micro_kernel_config.gemm_scratch_ld + i_m_offset)
                        * 4,
                        scale=4,
                        destination=TMM[acc_id],
                    )
                )
                acc_id += 1
                if n_tiles == 1 and l_enforce_Mx1_amx_tile_blocking == 0:
                    acc_id += 1
                i_n_offset += n_blocking_info.sizes[in_]
            i_m_offset += m_blocking_info.sizes[im]
        # Check if we have to restore the tmp registers
        _alu_mov_imm(code, gp_reg_mapping.gp_reg_ldc, (desc.ldc * 4) // 4)
        if (
            gp_reg_gemm_scratch == gp_reg_mapping.gp_reg_help_0
            and micro_kernel_config.m_loop_exists == 1
        ):
            _pop_reg(code, gp_reg_mapping.gp_reg_help_0)
    elif GEMMFlag.BETA_0 not in desc.flags:
        # Beta=1
        i_m_offset = 0
        for im in range(m_tiles):
            i_n_offset = 0
            for in_ in range(n_tiles):
                l_tile_disp = (i_n_offset * desc.ldc + i_m_offset) * 4
                code.emit(
                    x86.ops.DMX_TileloaddOp(
                        code.reg(gp_reg_mapping.gp_reg_c),
                        code.reg(gp_reg_mapping.gp_reg_ldc),
                        l_tile_disp,
                        scale=4,
                        destination=TMM[acc_id],
                    )
                )

                acc_id += 1
                if n_tiles == 1 and l_enforce_Mx1_amx_tile_blocking == 0:
                    acc_id += 1
                i_n_offset += n_blocking_info.sizes[in_]
            i_m_offset += m_blocking_info.sizes[im]
    else:
        # Beta=0
        for _ in range(m_tiles):
            for _ in range(n_tiles):
                code.emit(x86.ops.D_TilezeroOp(destination=TMM[acc_id]))
                acc_id += 1
                if n_tiles == 1 and l_enforce_Mx1_amx_tile_blocking == 0:
                    acc_id += 1


def libxsmm_generator_gemm_store_C_amx(
    code: AMXGeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    n_blocking_info: BlockingInfo,
    m_blocking_info: BlockingInfo,
) -> None:
    m_tiles = m_blocking_info.tiles
    n_tiles = n_blocking_info.tiles
    _C_tile_done = [0] * 4

    for i in range(m_tiles * n_tiles):
        im = micro_kernel_config._im[i]
        in_ = micro_kernel_config._in[i]
        _C_tile_done[micro_kernel_config._C_tile_id[i]] = 1
        if micro_kernel_config.use_paired_tilestores == 1:
            # If mate C tile is also ready, then two paired tilestore
            mate_id = micro_kernel_config._C_tile_mate_id
            if _C_tile_done[mate_id[micro_kernel_config._C_tile_id[i]]] == 1:
                min_mate_C_id = min(
                    micro_kernel_config._C_tile_id[i],
                    mate_id[micro_kernel_config._C_tile_id[i]],
                )
                im_store = min_mate_C_id // n_tiles
                in_store = min_mate_C_id % n_tiles
                libxsmm_generator_gemm_amx_paired_tilestore(
                    code,
                    gp_reg_mapping,
                    micro_kernel_config,
                    desc,
                    min_mate_C_id,
                    mate_id[min_mate_C_id],
                    micro_kernel_config._im_offset_prefix_sums[im_store],
                    micro_kernel_config._in_offset_prefix_sums[in_store],
                    n_blocking_info.sizes[in_store],
                )
        else:
            libxsmm_generator_gemm_amx_single_tilestore(
                code,
                gp_reg_mapping,
                micro_kernel_config,
                desc,
                micro_kernel_config._C_tile_id[i],
                micro_kernel_config._im_offset_prefix_sums[im],
                micro_kernel_config._in_offset_prefix_sums[in_],
                n_blocking_info.sizes[in_],
            )


def libxsmm_is_tile_in_last_tilerow(
    micro_kernel_config: MicroKernelConfig, tile: int
) -> bool:
    max_im = max(0, *micro_kernel_config._im)
    pos = 0
    for i in range(4):
        if micro_kernel_config._C_tile_id[i] == tile:
            pos = i
    return micro_kernel_config._im[pos] == max_im


def libxsmm_generator_cvtbf16ps_avx512(code: AMXGeneratedCode, vreg: int) -> None:
    """libxsmm_generator_cvt_to_ps_avx512 from BF16, in place: the words in the
    lower half of zmm`vreg` become the upper halves of its doublewords."""
    # convert 16 bit values into 32 bit (integer convert)
    code.emit(x86.ops.DS_VpmovsxwdOp(code.reg(YMM[vreg]), destination=ZMM[vreg]))
    # shift 16 bits to the left to generate valid FP32 numbers
    code.emit(x86.ops.DSI_VpslldOp(code.reg(ZMM[vreg]), 16, destination=ZMM[vreg]))


def libxsmm_generator_cvt_from_ps_avx512(code: AMXGeneratedCode, vreg: int) -> None:
    """libxsmm_generator_cvt_from_ps_avx512 to BF16, in place: the F32 of zmm`vreg`
    round to the words of its lower half."""
    code.emit(x86.ops.DS_Vcvtneps2bf16Op(code.reg(ZMM[vreg]), destination=YMM[vreg]))


def libxsmm_load_vreg_from_amx_tile(
    code: AMXGeneratedCode,
    micro_kernel_config: MicroKernelConfig,
    gp_reg_gemm_scratch: GeneralRegisterType,
    use_mask: int,
    col_id: int,
    vreg: int,
) -> None:
    """Load column `col_id` of a tile stored to the GEMM scratch; SPR has no
    instruction to move a tile row to a vector register directly."""
    _vec_load(
        code,
        x86.ops.DM_VmovupsOp,
        x86.ops.DMK_VmovupsOp,
        gp_reg_gemm_scratch,
        col_id * micro_kernel_config.gemm_scratch_ld * 4,
        ZMM[vreg],
        use_mask,
    )


def _tilestore_to_gemm_scratch(
    code: AMXGeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    gp_reg_gemm_scratch: GeneralRegisterType,
    tiles: Sequence[tuple[int, int]],
) -> None:
    """Store FP32 tiles to scratch: each (tile, displacement), with the row
    stride of the scratch in ldc meanwhile."""
    _alu_mov_imm(
        code,
        gp_reg_mapping.gp_reg_ldc,
        (micro_kernel_config.gemm_scratch_ld * 4) // 4,
    )
    for tile, displacement in tiles:
        code.emit(
            x86.ops.MXS_TilestoredOp(
                code.reg(gp_reg_gemm_scratch),
                code.reg(gp_reg_mapping.gp_reg_ldc),
                code.reg(TMM[tile]),
                displacement,
                scale=4,
            )
        )
    _alu_mov_imm(code, gp_reg_mapping.gp_reg_ldc, (desc.ldc * 4) // 4)


def libxsmm_generator_gemm_amx_paired_tilestore(
    code: AMXGeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    tile0: int,
    tile1: int,
    im_offset: int,
    in_offset: int,
    n_cols: int,
) -> None:
    """Store two vertically adjacent F32 tiles to a BF16 C, a column of both at a
    time: vcvtne2ps2bf16 rounds the column of `tile0` into the lower half of a
    register and that of `tile1` into the upper one.

    Without fusion the column is stored as soon as it is converted ("eager
    result store"), and LIBXSMM pairs only tiles that both exist.
    """
    assert tile1 >= 0 and desc.datatype.c == Datatype.BF16
    gp_reg_gemm_scratch = (
        gp_reg_mapping.gp_reg_help_0
        if micro_kernel_config.m_loop_exists == 0
        else gp_reg_mapping.gp_reg_help_1
    )
    reserved_zmms = micro_kernel_config.reserved_zmms
    out_tsize = 2

    # Check if we have to save the tmp registers
    if (
        gp_reg_gemm_scratch == gp_reg_mapping.gp_reg_help_1
        and micro_kernel_config.n_loop_exists == 1
    ):
        _push_reg(code, gp_reg_mapping.gp_reg_help_1)

    # Load the gemm scratch/relu ptr
    _getval_stack_var(code, GEMMStackVar.GEMM_SCRATCH_PTR, gp_reg_gemm_scratch)

    # SPR has no tilemovrow, so the tiles go through the scratch
    _tilestore_to_gemm_scratch(
        code,
        gp_reg_mapping,
        micro_kernel_config,
        desc,
        gp_reg_gemm_scratch,
        ((tile0, 0), (tile1, micro_kernel_config.gemm_scratch_ld * n_cols * 4)),
    )

    # Fully unroll in N dimension
    for col in range(n_cols):
        if col + reserved_zmms < 16:
            reg_0 = col % (16 - reserved_zmms) + reserved_zmms
        else:
            reg_0 = reserved_zmms + col % (31 - reserved_zmms)

        libxsmm_load_vreg_from_amx_tile(
            code,
            micro_kernel_config,
            gp_reg_gemm_scratch,
            micro_kernel_config.mask_m_fp32,
            col + n_cols,
            reg_0,
        )
        code.emit(
            x86.ops.DSM_Vcvtne2ps2bf16Op(
                code.reg(ZMM[reg_0]),
                code.reg(gp_reg_gemm_scratch),
                col * micro_kernel_config.gemm_scratch_ld * 4,
                destination=ZMM[reg_0],
            )
        )
        _vec_store(
            code,
            x86.ops.MS_Vmovdqu16Op,
            x86.ops.MSK_Vmovdqu16Op,
            gp_reg_mapping.gp_reg_c,
            ((in_offset + col) * desc.ldc + im_offset) * out_tsize,
            ZMM[reg_0],
            micro_kernel_config.mask_m_bf16,
        )

    # Check if we have to restore the tmp registers
    if (
        gp_reg_gemm_scratch == gp_reg_mapping.gp_reg_help_1
        and micro_kernel_config.n_loop_exists == 1
    ):
        _pop_reg(code, gp_reg_mapping.gp_reg_help_1)


def libxsmm_generator_gemm_amx_single_tilestore(
    code: AMXGeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    tile: int,
    im_offset: int,
    in_offset: int,
    n_cols: int,
) -> None:
    if desc.datatype.c == Datatype.F32:
        # An F32 C stores straight from the tile.
        store_offset = (in_offset * desc.ldc + im_offset) * 4
        code.emit(
            x86.ops.MXS_TilestoredOp(
                code.reg(gp_reg_mapping.gp_reg_c),
                code.reg(gp_reg_mapping.gp_reg_ldc),
                code.reg(TMM[tile]),
                store_offset,
                scale=4,
            )
        )
        return

    # A BF16 C goes through the scratch, a column of the tile at a time.
    gp_reg_gemm_scratch = (
        gp_reg_mapping.gp_reg_help_0
        if micro_kernel_config.m_loop_exists == 0
        else gp_reg_mapping.gp_reg_help_1
    )
    reserved_zmms = micro_kernel_config.reserved_zmms
    maskid = (
        micro_kernel_config.mask_m_fp32
        if libxsmm_is_tile_in_last_tilerow(micro_kernel_config, tile)
        and micro_kernel_config.m_remainder > 0
        else 0
    )

    # Potentially push aux register
    if (
        gp_reg_gemm_scratch == gp_reg_mapping.gp_reg_help_1
        and micro_kernel_config.n_loop_exists == 1
    ):
        _push_reg(code, gp_reg_mapping.gp_reg_help_1)

    _getval_stack_var(code, GEMMStackVar.GEMM_SCRATCH_PTR, gp_reg_gemm_scratch)
    _tilestore_to_gemm_scratch(
        code,
        gp_reg_mapping,
        micro_kernel_config,
        desc,
        gp_reg_gemm_scratch,
        ((tile, 0),),
    )

    for col in range(n_cols):
        reg_0 = col % (16 - reserved_zmms) + reserved_zmms
        libxsmm_load_vreg_from_amx_tile(
            code, micro_kernel_config, gp_reg_gemm_scratch, maskid, col, reg_0
        )
        libxsmm_generator_cvt_from_ps_avx512(code, reg_0)
        _vec_store(
            code,
            x86.ops.MS_Vmovdqu16Op,
            x86.ops.MSK_Vmovdqu16Op,
            gp_reg_mapping.gp_reg_c,
            ((in_offset + col) * desc.ldc + im_offset)
            * micro_kernel_config.datatype_size_out,
            YMM[reg_0],
            maskid,
        )

    # Potentially pop aux register
    if (
        gp_reg_gemm_scratch == gp_reg_mapping.gp_reg_help_1
        and micro_kernel_config.n_loop_exists == 1
    ):
        _pop_reg(code, gp_reg_mapping.gp_reg_help_1)


def libxsmm_generator_gemm_amx_kernel_mloop(
    code: AMXGeneratedCode,
    loop_label_tracker: LoopLabelTracker,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    n_blocking_info: BlockingInfo,
    m_blocking_info: Sequence[BlockingInfo],
) -> None:
    l_m_done = 0
    l_m_count = 0
    l_m_blocking = m_blocking_info[0].blocking
    m_assembly_loop_exists = 0 if l_m_blocking == desc.m else 1
    fully_unroll_k = 1
    unroll_hint = 0

    micro_kernel_config.B_offs_trans = 0
    micro_kernel_config.loop_label_id = 4
    micro_kernel_config.is_peeled_br_loop = 0
    micro_kernel_config.cur_unroll_factor = unroll_hint

    # apply m_blocking
    while l_m_done != desc.m:
        l_m_blocking = m_blocking_info[l_m_count].blocking
        l_m_block_size = m_blocking_info[l_m_count].block_size
        l_m_done += l_m_block_size

        if m_assembly_loop_exists:
            libxsmm_generator_gemm_header_mloop_amx(
                code,
                loop_label_tracker,
                gp_reg_mapping,
                micro_kernel_config,
                l_m_blocking,
            )

        libxsmm_generator_gemm_load_C_amx(
            code,
            gp_reg_mapping,
            micro_kernel_config,
            desc,
            n_blocking_info,
            m_blocking_info[l_m_count],
        )

        # Here is the K loop along with the microkernel
        libxsmm_generator_gemm_amx_kernel_kloop(
            code,
            loop_label_tracker,
            gp_reg_mapping,
            micro_kernel_config,
            desc,
            n_blocking_info,
            m_blocking_info[l_m_count],
            0,
            0,
            0,
        )
        libxsmm_generator_gemm_store_C_amx(
            code,
            gp_reg_mapping,
            micro_kernel_config,
            desc,
            n_blocking_info,
            m_blocking_info[l_m_count],
        )

        if m_assembly_loop_exists:
            libxsmm_generator_gemm_footer_mloop_amx(
                code,
                loop_label_tracker,
                gp_reg_mapping,
                micro_kernel_config,
                desc,
                l_m_blocking,
                l_m_done,
                fully_unroll_k,
            )
        l_m_count += 1


def libxsmm_generator_gemm_amx_kernel_nloop(
    code: AMXGeneratedCode,
    loop_label_tracker: LoopLabelTracker,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    n_blocking_info: Sequence[BlockingInfo],
    m_blocking_info: Sequence[BlockingInfo],
) -> None:
    # initialize n-blocking
    l_n_count = 0  # array counter for blocking info arrays
    l_n_done = 0  # progress tracker
    m_assembly_loop_exists = 1 if m_blocking_info[0].blocking < desc.m else 0
    n_assembly_loop_exists = 1 if n_blocking_info[0].blocking < desc.n else 0

    micro_kernel_config.m_loop_exists = m_assembly_loop_exists
    micro_kernel_config.n_loop_exists = n_assembly_loop_exists

    # apply n_blocking
    while l_n_done != desc.n:
        l_n_blocking = n_blocking_info[l_n_count].blocking
        l_n_block_size = n_blocking_info[l_n_count].block_size
        # advance N
        l_n_done += l_n_block_size
        if l_n_blocking < desc.n:
            # Open N loop
            libxsmm_generator_gemm_header_nloop_amx(
                code,
                loop_label_tracker,
                gp_reg_mapping,
                micro_kernel_config,
                l_n_blocking,
            )

        # Generate M loop
        libxsmm_generator_gemm_amx_kernel_mloop(
            code,
            loop_label_tracker,
            gp_reg_mapping,
            micro_kernel_config,
            desc,
            n_blocking_info[l_n_count],
            m_blocking_info,
        )

        if l_n_blocking < desc.n:
            # Close N loop
            libxsmm_generator_gemm_footer_nloop_amx(
                code,
                loop_label_tracker,
                gp_reg_mapping,
                micro_kernel_config,
                desc,
                l_n_blocking,
                l_n_done,
                m_assembly_loop_exists,
            )
        l_n_count += 1


def libxsmm_generator_gemm_header_k_loop_amx(
    code: AMXGeneratedCode,
    loop_label_tracker: LoopLabelTracker,
    micro_kernel_config: MicroKernelConfig,
    cnt_reg: GeneralRegisterType,
) -> None:
    _alu_mov_imm(code, cnt_reg, 0)
    _register_jump_back_label(code, loop_label_tracker)


def libxsmm_generator_gemm_footer_k_loop_amx(
    code: AMXGeneratedCode,
    loop_label_tracker: LoopLabelTracker,
    micro_kernel_config: MicroKernelConfig,
    cnt_reg: GeneralRegisterType,
    n_iters: int,
) -> None:
    _alu_imm(code, x86.ops.RI_AddOp, cnt_reg, 1)
    cmp_op = _cmp_imm(code, cnt_reg, n_iters)
    _jump_back_to_label(code, cmp_op, loop_label_tracker)


def libxsmm_generator_gemm_amx_kernel_kloop(
    code: AMXGeneratedCode,
    loop_label_tracker: LoopLabelTracker,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    n_blocking_info: BlockingInfo,
    m_blocking_info: BlockingInfo,
    A_offs: int,
    B_offs: int,
    fully_unrolled_brloop: int,
) -> None:
    l_k_pack_factor = BF16_K_PACK_FACTOR
    i_brgemm_loop = -2
    l_lda = desc.lda
    l_a_dtype_size = micro_kernel_config.datatype_size_in

    l_k_blocking = 32
    while desc.k % l_k_blocking != 0:
        l_k_blocking -= l_k_pack_factor

    # Limit unrolling in case of large number of trips counts
    l_unroll_kernel_limit = 256
    l_brgemm_unroll_factor = 1
    l_k_trips = (desc.k + l_k_blocking - 1) // l_k_blocking
    if l_k_trips * l_brgemm_unroll_factor > l_unroll_kernel_limit:
        l_k_gp_reg = gp_reg_mapping.gp_reg_help_1
        l_k_unroll_limit = 4
        while (l_k_trips - 1) % l_k_unroll_limit != 0:
            l_k_unroll_limit -= 1
        # K loop here
        is_last_k = 0
        _push_reg(code, l_k_gp_reg)
        libxsmm_generator_gemm_header_k_loop_amx(
            code, loop_label_tracker, micro_kernel_config, l_k_gp_reg
        )
        for k in range(0, l_k_unroll_limit * l_k_blocking, l_k_blocking):
            micro_kernel_config.k_amx_microkernel = k
            offset_A = (
                k * l_lda * l_a_dtype_size
            ) // micro_kernel_config.sparsity_factor_A + A_offs
            offset_B = k * micro_kernel_config.datatype_size_in2 + B_offs
            libxsmm_generator_gemm_amx_microkernel(
                code,
                gp_reg_mapping,
                micro_kernel_config,
                desc,
                n_blocking_info,
                m_blocking_info,
                offset_A,
                offset_B,
                is_last_k,
                i_brgemm_loop,
                fully_unrolled_brloop,
            )
        _alu_imm(
            code,
            x86.ops.RI_AddOp,
            gp_reg_mapping.gp_reg_a,
            (l_lda * l_k_unroll_limit * l_k_blocking * l_a_dtype_size)
            // micro_kernel_config.sparsity_factor_A,
        )
        _alu_imm(
            code,
            x86.ops.RI_AddOp,
            gp_reg_mapping.gp_reg_b,
            l_k_unroll_limit * l_k_blocking * micro_kernel_config.datatype_size_in2,
        )
        libxsmm_generator_gemm_footer_k_loop_amx(
            code,
            loop_label_tracker,
            micro_kernel_config,
            l_k_gp_reg,
            (l_k_trips - 1) // l_k_unroll_limit,
        )
        _alu_imm(
            code,
            x86.ops.RI_SubOp,
            gp_reg_mapping.gp_reg_a,
            (l_lda * l_a_dtype_size * (l_k_trips - 1) * l_k_blocking)
            // micro_kernel_config.sparsity_factor_A,
        )
        _alu_imm(
            code,
            x86.ops.RI_SubOp,
            gp_reg_mapping.gp_reg_b,
            (l_k_trips - 1) * l_k_blocking * micro_kernel_config.datatype_size_in2,
        )
        _pop_reg(code, l_k_gp_reg)
        # Peel off last iteration
        is_last_k = 1
        k = (l_k_trips - 1) * l_k_blocking
        offset_A = (
            k * l_lda * l_a_dtype_size
        ) // micro_kernel_config.sparsity_factor_A + A_offs
        offset_B = k * micro_kernel_config.datatype_size_in2 + B_offs
        libxsmm_generator_gemm_amx_microkernel(
            code,
            gp_reg_mapping,
            micro_kernel_config,
            desc,
            n_blocking_info,
            m_blocking_info,
            offset_A,
            offset_B,
            is_last_k,
            i_brgemm_loop,
            fully_unrolled_brloop,
        )
    else:
        # For now fully unroll the k loop
        for k in range(0, desc.k, l_k_blocking):
            micro_kernel_config.k_amx_microkernel = k
            is_last_k = 1 if k + l_k_blocking >= desc.k else 0
            offset_A = (
                k * l_lda * l_a_dtype_size
            ) // micro_kernel_config.sparsity_factor_A + A_offs
            offset_B = k * micro_kernel_config.datatype_size_in2 + B_offs
            libxsmm_generator_gemm_amx_microkernel(
                code,
                gp_reg_mapping,
                micro_kernel_config,
                desc,
                n_blocking_info,
                m_blocking_info,
                offset_A,
                offset_B,
                is_last_k,
                i_brgemm_loop,
                fully_unrolled_brloop,
            )


def libxsmm_generator_gemm_amx_microkernel(
    code: AMXGeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    n_blocking_info: BlockingInfo,
    m_blocking_info: BlockingInfo,
    offset_A: int,
    offset_B: int,
    is_last_k: int,
    i_brgemm_loop: int,
    fully_unrolled_brloop: int,
) -> None:
    m_tiles = m_blocking_info.tiles
    n_tiles = n_blocking_info.tiles
    # Tiles are stored in the micro-kernel only at the end of a peeled batch-reduce
    # loop; otherwise store_C_amx stores them.
    assert i_brgemm_loop == -2 and fully_unrolled_brloop == 0
    l_enforce_Mx1_amx_tile_blocking = micro_kernel_config.enforce_Mx1_amx_tile_blocking
    use_paired_tilestores = (
        1
        if desc.datatype.c == Datatype.BF16
        and m_tiles % 2 == 0
        and micro_kernel_config.vnni_format_C == 0
        and l_enforce_Mx1_amx_tile_blocking == 0
        else 0
    )
    lda_adjustment = 1  # A is in VNNI layout
    ldb_adjustment = 1  # B is not transposed
    l_ldb = desc.ldb

    # Tiles in the kernel are organized as indicated below when we use 2x2 2D blocking
    #
    #       C            A        B
    #
    #    N0  N1          Ki     N0  N1
    #
    # M0 T0  T1      M0  T4     T6  T7  Ki
    #           +=           X
    # M1 T2  T3      M1  T5
    #

    # Arrays that indicate the tile traversals in the microkernel; -1 is a step
    # that loads no tile.
    _im = [0] * 4
    _in = [0] * 4
    _in_tileloads_B = [0] * 4
    _A_tile_id = [0] * 4
    _B_tile_id = [0] * 4
    _C_tile_id = [0] * 4
    _A_tile_id_load = [0] * 4
    _B_tile_id_load = [0] * 4
    _A_offsets = [0] * 4
    _B_offsets = [0] * 4
    _im_offset_prefix_sums = [0] * 4
    _in_offset_prefix_sums = [0] * 4
    # The C tile each C tile pairs its stores with, -1 for none
    _C_tile_mate_id = [-1] * 4

    for i in range(1, 4):
        _im_offset_prefix_sums[i] = (
            _im_offset_prefix_sums[i - 1] + m_blocking_info.sizes[i - 1]
        )
    for i in range(1, 4):
        _in_offset_prefix_sums[i] = (
            _in_offset_prefix_sums[i - 1] + n_blocking_info.sizes[i - 1]
        )

    match (m_tiles, n_tiles):
        case (2, 2):
            _C_tile_mate_id[:] = (2, 3, 0, 1)
        case (2, 1) if l_enforce_Mx1_amx_tile_blocking > 0:
            _C_tile_mate_id[:] = (1, 0, -1, -1)
        case (2, 1):
            _C_tile_mate_id[:] = (2, -1, 0, -1)
        case (4, 1):
            _C_tile_mate_id[:] = (1, 0, 3, 2)
        case _:
            # In this case we cannot do paired tilestores
            pass

    # A is always VNNI-formatted here.
    # The tile loads are plain tileloadd: the streaming and read-shared ones only
    # replace them under LIBXSMM's environment knobs.
    match (m_tiles, n_tiles):
        case (2, 2):
            # Encode loop iterations
            _im[:] = (0, 1, 0, 1)
            _in[:] = (0, 0, 1, 1)
            _A_tile_id[:] = (4, 5, 4, 5)
            _A_tile_id_load[:] = (4, 5, -1, -1)
            _B_tile_id[:] = (6, 6, 7, 7)
            _B_tile_id_load[:] = (6, -1, 7, -1)
            _in_tileloads_B[:] = (0, -1, 1, -1)
        case (1, 2):
            _im[:] = (0, 0, -1, -1)
            _in[:] = (0, 1, -1, -1)
            _A_tile_id[:] = (4, 4, -1, -1)
            _A_tile_id_load[:] = (4, -1, -1, -1)
            _B_tile_id[:] = (6, 7, -1, -1)
            _B_tile_id_load[:] = (6, 7, -1, -1)
            _in_tileloads_B[:] = (0, 1, -1, -1)
        case (2, 1):
            _im[:] = (0, 1, -1, -1)
            _in[:] = (0, 0, -1, -1)
            _A_tile_id[:] = (4, 5, -1, -1)
            _A_tile_id_load[:] = (4, 5, -1, -1)
            _B_tile_id[:] = (6, 6, -1, -1)
            _B_tile_id_load[:] = (6, -1, -1, -1)
            _in_tileloads_B[:] = (0, -1, -1, -1)
        case (3, 1):
            _im[:] = (0, 1, 2, -1)
            _in[:] = (0, 0, 0, -1)
            _A_tile_id[:] = (4, 4, 5, -1)
            _A_tile_id_load[:] = (4, 4, 5, -1)
            _B_tile_id[:] = (6, 6, 6, -1)
            _B_tile_id_load[:] = (6, -1, -1, -1)
            _in_tileloads_B[:] = (0, -1, -1, -1)
        case (4, 1):
            _im[:] = (0, 1, 2, 3)
            _in[:] = (0, 0, 0, 0)
            _A_tile_id[:] = (4, 4, 4, 5)
            _A_tile_id_load[:] = (4, 4, 4, 5)
            _B_tile_id[:] = (6, 6, 6, 6)
            _B_tile_id_load[:] = (6, -1, -1, -1)
            _in_tileloads_B[:] = (0, -1, -1, -1)
        case (1, 1):
            _im[:] = (0, -1, -1, -1)
            _in[:] = (0, -1, -1, -1)
            _A_tile_id[:] = (4, -1, -1, -1)
            _A_tile_id_load[:] = (4, -1, -1, -1)
            _B_tile_id[:] = (6, -1, -1, -1)
            _B_tile_id_load[:] = (6, -1, -1, -1)
            _in_tileloads_B[:] = (0, -1, -1, -1)
        case _:
            raise NotImplementedError(f"{m_tiles}x{n_tiles} tiles")

    # Fill in the accumulator IDs properly and the A/B offsets. The C tiles are
    # numbered as in a 2x2 grid, except when they are all in one column of M
    # (always in Mx1 blocking).
    for i in range(4):
        if m_tiles > 2 or (
            m_tiles == 2 and n_tiles == 1 and l_enforce_Mx1_amx_tile_blocking
        ):
            _C_tile_id[i] = _im[i]
        else:
            _C_tile_id[i] = _im[i] * 2 + _in[i]
        _A_offsets[i] = (
            offset_A
            + (_im_offset_prefix_sums[_im[i]] * 4)
            // micro_kernel_config.sparsity_factor_A
        )
        # For 3, 4 and 1 tiles LIBXSMM scales by datatype_size_in, for the others
        # by datatype_size_in2: both are the 2 bytes of BF16.
        _B_offsets[i] = (
            offset_B
            + _in_offset_prefix_sums[_in_tileloads_B[i]]
            * l_ldb
            * micro_kernel_config.datatype_size_in2
        )

    # Now copy over the GEMM traversal meta-data for later usage by the store
    # function if needed
    micro_kernel_config.use_paired_tilestores = use_paired_tilestores
    micro_kernel_config._im = list(_im)
    micro_kernel_config._in = list(_in)
    micro_kernel_config._C_tile_id = list(_C_tile_id)
    micro_kernel_config._C_tile_mate_id = list(_C_tile_mate_id)
    micro_kernel_config._im_offset_prefix_sums = list(_im_offset_prefix_sums)
    micro_kernel_config._in_offset_prefix_sums = list(_in_offset_prefix_sums)

    for i in range(m_tiles * n_tiles):
        if _A_tile_id_load[i] > 0:
            code.emit(
                x86.ops.DMX_TileloaddOp(
                    code.reg(gp_reg_mapping.gp_reg_a),
                    code.reg(gp_reg_mapping.gp_reg_lda),
                    _A_offsets[i] * micro_kernel_config.sparsity_factor_A,
                    scale=4 // lda_adjustment,
                    destination=TMM[_A_tile_id_load[i]],
                )
            )

        if _B_tile_id_load[i] > 0:
            code.emit(
                x86.ops.DMX_TileloaddOp(
                    code.reg(gp_reg_mapping.gp_reg_b),
                    code.reg(gp_reg_mapping.gp_reg_ldb),
                    _B_offsets[i],
                    scale=4 // ldb_adjustment,
                    destination=TMM[_B_tile_id_load[i]],
                )
            )

        # LIBXSMM's operands are A, B and C, in AT&T order; in Intel order the
        # first source has the rows of C, which are the columns of B.
        code.emit(
            x86.ops.RSS_Tdpbf16psOp(
                code.tile(TMM[_C_tile_id[i]]),
                code.reg(TMM[_B_tile_id[i]]),
                code.reg(TMM[_A_tile_id[i]]),
            )
        )
