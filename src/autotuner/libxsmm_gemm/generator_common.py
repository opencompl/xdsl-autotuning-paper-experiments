from dataclasses import dataclass, field

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.x86 import registers
from xdsl.ir import Block, Region
from xdsl.dialects.x86_func import FuncOp, RetOp
from xdsl.rewriter import InsertPoint, Rewriter


def libxsmm_mmfunction_signature(module: ModuleOp, routine_name: str) -> FuncOp:
    operand_types = (registers.RDI, registers.RSI, registers.RDX)

    func = FuncOp(
        routine_name,
        Region(Block((RetOp(),), arg_types=operand_types)),
        (operand_types, ()),
        "public",
    )

    Rewriter.insert_op(func, InsertPoint.at_end(module.body.block))

    return func


@dataclass
class MicroKernelConfig:
    """libxsmm_micro_kernel_config"""


@dataclass
class LoopLabelTracker:
    """
    Structure for tracking local labels in assembly we do not allow overlapping loops
    libxsmm_loop_label_tracker
    """

    label_address: list[int] = field(default_factory=list)


@dataclass
class GPRegMapping:
    """
    Structure for storing the current GP (General Purpose) register mapping.
    libxsmm_gp_reg_mapping
    """

    gp_reg_param_struct: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_a: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_a_base: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_b: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_b_base: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_c: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_a_prefetch: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_a_offset: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_b_prefetch: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_b_offset: registers.GeneralRegisterType | None = field(default=None)
    # gp_reg_c_prefetch: int  # commented out in C code
    gp_reg_mloop: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_nloop: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_kloop: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_reduce_count: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_reduce_loop: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_a_ptrs: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_b_ptrs: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_lda: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_ldb: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_ldc: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_scf: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_zpt: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_help_0: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_help_1: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_help_2: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_help_3: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_help_4: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_help_5: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_help_6: registers.GeneralRegisterType | None = field(default=None)
    # Auxiliary regs for sparsity in A support
    gp_reg_bitmap_a: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_decompressed_a: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_decompressed_elts: registers.GeneralRegisterType | None = field(default=None)
    gp_reg_popcnt: registers.GeneralRegisterType | None = field(default=None)
