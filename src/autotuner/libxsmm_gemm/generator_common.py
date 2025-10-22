from dataclasses import dataclass, field

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.x86.registers import (
    GeneralRegisterType,
    UNALLOCATED_GENERAL,
    RDI,
    RSI,
    RDX,
)
from xdsl.ir import Block, Region
from xdsl.dialects.x86_func import FuncOp, RetOp
from xdsl.rewriter import InsertPoint, Rewriter


def libxsmm_mmfunction_signature(module: ModuleOp, routine_name: str) -> FuncOp:
    operand_types = (RDI, RSI, RDX)

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

    datatype_size_in2: int


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

    @property
    def current_loop_number(self) -> int:
        # No idea why libxsmm does this
        return len(self.dest_blocks) + 32


@dataclass
class GPRegMapping:
    """
    Structure for storing the current GP (General Purpose) register mapping.
    libxsmm_gp_reg_mapping
    """

    gp_reg_param_struct: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_a: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_a_base: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_b: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_b_base: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_c: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_a_prefetch: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_a_offset: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_b_prefetch: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_b_offset: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    # gp_reg_c_prefetch: int  # commented out in C code
    gp_reg_mloop: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_nloop: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_kloop: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_reduce_count: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_reduce_loop: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_a_ptrs: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_b_ptrs: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_lda: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_ldb: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_ldc: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_scf: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_zpt: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_help_0: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_help_1: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_help_2: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_help_3: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_help_4: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_help_5: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_help_6: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    # Auxiliary regs for sparsity in A support
    gp_reg_bitmap_a: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_decompressed_a: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_decompressed_elts: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
    gp_reg_popcnt: GeneralRegisterType = field(default=UNALLOCATED_GENERAL)
