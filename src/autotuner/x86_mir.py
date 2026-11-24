"""Print CompXSMM's register-unallocated x86 kernels as LLVM Machine IR.

CompXSMM assigns registers itself and never spills.  To see how LLVM's
allocator fares on exactly the same instructions, the IR just before
`x86-allocate-registers`, with its loops lowered to blocks by
`convert-x86-scf-to-x86`, is printed as LLVM Machine IR (MIR) in SSA form, with
a virtual register for every value and a PHI for every block argument, and
`llc` is started at LiveVariables.
That skips every SSA-level machine optimisation; with the machine scheduler
also disabled, LLVM keeps our instruction order and only runs two-address
lowering, coalescing, and register allocation (greedy by default) before
emitting code.  LiveVariables has to run: our MIR carries no kill flags, and
without them two-address lowering thinks every FMA's accumulator outlives it
and unfolds the broadcast operand rather than copy the accumulator, which
costs up to 40% on small kernels.

The only liberties LLVM gets are those any allocator would:

- Values our schedule pins to a register (the pointer arguments and the
  `k1` mask) are virtual, so LLVM may pick other registers for them.
- A `get_avx_register` zeroed by `vpxord` becomes LLVM's zero idiom, which
  LLVM may rematerialise instead of keeping it live.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import IO

from xdsl.context import Context
from xdsl.dialects import x86, x86_func
from xdsl.dialects.builtin import IntegerAttr, ModuleOp
from xdsl.dialects.x86.registers import (
    AVX2RegisterType,
    AVX512MaskRegisterType,
    AVX512RegisterType,
    GeneralRegisterType,
    SSERegisterType,
    X86VectorRegisterType,
)
from xdsl.ir import Block, Operation, SSAValue
from xdsl.utils.target import Target

# Opcode suffix and register class of each vector width (EVEX, so all 32).
VECTOR_FORMS = {
    AVX512RegisterType: ("Z", "vr512"),
    AVX2RegisterType: ("Z256", "vr256x"),
    SSERegisterType: ("Z128", "vr128x"),
}
SET0 = {
    AVX512RegisterType: "AVX512_512_SET0",
    AVX2RegisterType: "AVX512_256_SET0",
    SSERegisterType: "AVX512_128_SET0",
}
# A `kmov` decides the width of the mask it defines; masked instructions take
# only k1-k7, hence the write-mask classes.
MASK_CLASSES = {"kmovb": "vk8wm", "kmovw": "vk16wm", "kmovd": "vk32wm"}
GPR_ARGUMENTS = ("rdi", "rsi", "rdx", "rcx", "r8", "r9")
# The two-operand forms of the GPR arithmetic the schedules emit.
GPR_RI = {"add": "ADD64ri32", "sub": "SUB64ri32", "imul": "IMUL64rri32"}
GPR_RR = {"add": "ADD64rr", "sub": "SUB64rr", "imul": "IMUL64rr"}
# The X86::CondCode of each conditional jump `convert-x86-scf-to-x86` emits.
CONDITIONS = {"jl": 12, "jge": 13}


def is_fp(mnemonic: str) -> bool:
    """Floating-point arithmetic reads MXCSR, which MIR spells out."""
    return mnemonic.startswith(("vfmadd", "vfmsub", "vfnmadd", "vfnmsub")) or (
        mnemonic[:4] in ("vadd", "vsub", "vmul", "vdiv", "vmax", "vmin")
        and mnemonic.endswith(("ps", "pd", "ss", "sd"))
    )


class MirPrinter:
    """Prints one `x86_func.func` with unallocated registers as SSA MIR."""

    def __init__(self, func: x86_func.FuncOp):
        self.func = func
        self.vregs: dict[SSAValue, str] = {}
        self.classes: list[str] = []

    # Values and operands

    def new_vreg(self, cls: str) -> str:
        self.classes.append(cls)
        return f"%{len(self.classes) - 1}"

    def register_class(self, value: SSAValue) -> str:
        match value.type:
            case GeneralRegisterType():
                return "gr64"
            case AVX512MaskRegisterType():
                assert isinstance(value.owner, x86.ops.X86Instruction)
                return MASK_CLASSES[value.owner.assembly_instruction_name()]
            case X86VectorRegisterType():
                return VECTOR_FORMS[type(value.type)][1]
            case t:
                raise NotImplementedError(f"no MIR register class for {t}")

    def define(self, value: SSAValue) -> str:
        """A fresh virtual register for a value the IR defines."""
        assert value not in self.vregs, value
        self.vregs[value] = vreg = self.new_vreg(self.register_class(value))
        return vreg

    def use(self, value: SSAValue) -> str:
        return self.vregs[value]

    def memory(self, base: SSAValue, offset: IntegerAttr) -> str:
        return f"{self.use(base)}, 1, $noreg, {offset.value.data}, $noreg"

    # Conversion

    def print(self, output: IO[str]) -> None:
        blocks = list(self.func.body.blocks)
        number = {block: index for index, block in enumerate(blocks)}
        liveins: list[str] = []
        entry = blocks[0]
        prologue: list[str] = []
        for arg, default in zip(entry.args, GPR_ARGUMENTS, strict=False):
            assert isinstance(arg.type, GeneralRegisterType)
            phys = arg.type.register_name.data or default
            liveins.append(f"${phys}")
            prologue.append(f"{self.define(arg)} = COPY ${phys}")
        # Other blocks' arguments are PHIs, which may read values defined later.
        for block in blocks[1:]:
            for arg in block.args:
                self.define(arg)
        bodies = [
            [line for op in block.ops for line in self.convert_op(op, number)]
            for block in blocks
        ]

        incoming: dict[Block, list[tuple[Block, Sequence[SSAValue]]]] = {
            block: [] for block in blocks
        }
        for block in blocks:
            match block.last_op:
                case x86.ops.ConditionalJumpOperation() as jump:
                    incoming[jump.then_block].append((block, jump.then_values))
                    incoming[jump.else_block].append((block, jump.else_values))
                case x86.ops.FallthroughOp() as jump:
                    incoming[jump.successor].append((block, jump.block_values))
                case _:
                    pass

        output.write("---\n")
        output.write(f"name: {self.func.sym_name.data}\n")
        output.write("tracksRegLiveness: true\n")
        output.write("registers:\n")
        for index, cls in enumerate(self.classes):
            output.write(f"  - {{ id: {index}, class: {cls} }}\n")
        output.write("liveins:\n")
        for reg in liveins:
            output.write(f"  - {{ reg: '{reg}' }}\n")
        output.write("body: |\n")
        for block, body in zip(blocks, bodies, strict=True):
            output.write(f"  bb.{number[block]}:\n")
            if (last := block.last_op) is not None and last.successors:
                succ = ", ".join(f"%bb.{number[s]}" for s in last.successors)
                output.write(f"    successors: {succ}\n")
            if block is entry:
                if liveins:
                    output.write(f"    liveins: {', '.join(liveins)}\n")
                body = prologue + body
            else:
                body = [
                    f"{self.use(arg)} = PHI "
                    + ", ".join(
                        f"{self.use(values[i])}, %bb.{number[pred]}"
                        for pred, values in incoming[block]
                    )
                    for i, arg in enumerate(block.args)
                ] + body
            for line in body:
                output.write(f"    {line}\n")
            output.write("\n")
        output.write("...\n")

    def convert_op(self, op: Operation, number: dict[Block, int]) -> list[str]:
        match op:
            case x86.ops.LabelOp() | x86.ops.FallthroughOp():
                return []
            case x86.ops.SI_CmpOp():
                value = op.immediate.value.data
                return [
                    f"CMP64ri32 {self.use(op.source)}, {value}, implicit-def $eflags"
                ]
            case x86.ops.SS_CmpOp():
                a, b = self.use(op.source1), self.use(op.source2)
                return [f"CMP64rr {a}, {b}, implicit-def $eflags"]
            case x86.ops.ConditionalJumpOperation():
                condition = CONDITIONS[op.assembly_instruction_name()]
                then, else_ = number[op.then_block], number[op.else_block]
                return [
                    f"JCC_1 %bb.{then}, {condition}, implicit $eflags",
                    f"JMP_1 %bb.{else_}",
                ]
            case x86_func.RetOp():
                return ["RET64"]
            case x86.ops.GetAVXRegisterOp() if all(
                isinstance(use.operation, x86.ops.DSS_VpxordOp)
                and use.operation.source1 == use.operation.source2
                for use in op.result.uses
            ):
                return []  # only zeroed: see `vpxord` below
            case x86.ops.DSS_VpxordOp() if op.source1 == op.source2 and isinstance(
                op.source1.owner, x86.ops.GetAVXRegisterOp
            ):
                opcode = SET0[type(op.destination.type)]
                return [f"{self.define(op.destination)} = {opcode}"]
            case _:
                return [self.instruction(op)]

    def instruction(self, op: Operation) -> str:
        if not isinstance(op, x86.ops.X86Instruction):
            raise NotImplementedError(f"cannot convert {op.name} to MIR")
        mnemonic = op.assembly_instruction_name()
        implicit = ", implicit $mxcsr" if is_fp(mnemonic) else ""

        def vector(value: SSAValue, form: str) -> str:
            return f"{mnemonic.upper()}{VECTOR_FORMS[type(value.type)][0]}{form}"

        match op:
            # Loads
            case x86.ops.DM_Operation() if isinstance(
                op.destination.type, GeneralRegisterType
            ):
                mem = self.memory(op.memory, op.memory_offset)
                return f"{self.define(op.destination)} = MOV64rm {mem}"
            case x86.ops.DM_Operation():
                mem = self.memory(op.memory, op.memory_offset)
                opcode = vector(op.destination, "rm")
                return f"{self.define(op.destination)} = {opcode} {mem}"
            case x86.ops.DMK_Operation() if op.z is not None:
                mem = self.memory(op.memory, op.memory_offset)
                opcode = vector(op.destination, "rmkz")
                mask = self.use(op.mask_reg)
                return f"{self.define(op.destination)} = {opcode} {mask}, {mem}"
            # Stores
            case x86.ops.MS_Operation() if isinstance(
                op.source.type, GeneralRegisterType
            ):
                mem = self.memory(op.memory, op.memory_offset)
                return f"MOV64mr {mem}, {self.use(op.source)}"
            case x86.ops.MS_Operation():
                mem = self.memory(op.memory, op.memory_offset)
                return f"{vector(op.source, 'mr')} {mem}, {self.use(op.source)}"
            case x86.ops.MSK_Operation() if op.z is None:
                mem = self.memory(op.memory, op.memory_offset)
                mask, source = self.use(op.mask_reg), self.use(op.source)
                return f"{vector(op.source, 'mrk')} {mem}, {mask}, {source}"
            # Accumulations, whose result is tied to `register_in`
            case x86.ops.RSS_Operation():
                acc, a, b = map(self.use, (op.register_in, op.source1, op.source2))
                opcode = vector(op.register_out, "r")
                dest = self.define(op.register_out)
                return f"{dest} = {opcode} {acc}, {a}, {b}{implicit}"
            case x86.ops.RSM_Operation() | x86.ops.RSMB_Operation():
                acc, a = self.use(op.register_in), self.use(op.source1)
                mem = self.memory(op.memory, op.memory_offset)
                broadcast = isinstance(op, x86.ops.RSMB_Operation) and op.broadcast
                form = "mb" if broadcast else "m"
                opcode = vector(op.register_out, form)
                dest = self.define(op.register_out)
                return f"{dest} = {opcode} {acc}, {a}, {mem}{implicit}"
            # Three-operand vector arithmetic (including xsmm.accumulator.*)
            case x86.ops.DSS_Operation() if isinstance(
                op.destination.type, X86VectorRegisterType
            ):
                a, b = self.use(op.source1), self.use(op.source2)
                opcode = vector(op.destination, "rr")
                dest = self.define(op.destination)
                return f"{dest} = {opcode} {a}, {b}{implicit}"
            # General-purpose arithmetic
            case x86.ops.DI_Operation() if mnemonic == "mov":
                value = op.immediate.value.data
                return f"{self.define(op.destination)} = MOV64ri32 {value}"
            case x86.ops.RI_Operation() if mnemonic in GPR_RI:
                value = op.immediate.value.data
                src = self.use(op.register_in)
                dest = self.define(op.register_out)
                return (
                    f"{dest} = {GPR_RI[mnemonic]} {src}, {value}, "
                    "implicit-def dead $eflags"
                )
            case x86.ops.RS_Operation() if mnemonic in GPR_RR:
                src, other = self.use(op.register_in), self.use(op.source)
                dest = self.define(op.register_out)
                return (
                    f"{dest} = {GPR_RR[mnemonic]} {src}, {other}, "
                    "implicit-def dead $eflags"
                )
            case x86.ops.KS_Operation() if mnemonic in MASK_CLASSES:
                low = self.new_vreg("gr32")
                dest = self.define(op.destination)
                width = mnemonic[-1].upper()
                return (
                    f"{low} = COPY {self.use(op.source)}.sub_32bit\n    "
                    f"{dest} = KMOV{width}kr {low}"
                )
            case _:
                raise NotImplementedError(f"cannot convert {op.name} to MIR")


def print_mir(module: ModuleOp, output: IO[str]) -> None:
    for op in module.ops:
        if not isinstance(op, x86_func.FuncOp):
            raise NotImplementedError(f"cannot convert {op.name} to MIR")
        MirPrinter(op).print(output)


@dataclass(frozen=True)
class X86MirTarget(Target):
    """`xdsl-opt -t x86-mir`: unallocated x86 IR as LLVM SSA Machine IR."""

    name = "x86-mir"

    def emit(self, ctx: Context, module: ModuleOp, output: IO[str]) -> None:
        print_mir(module, output)


# How llc compiles our MIR, here and in the `compxsmm_llvm` benchmark variant.
LLC_FLAGS = (
    "-O2",
    "-x",
    "mir",
    "-start-before=livevars",
    "-enable-misched=false",
)
# llc's comment on every stack-slot access its allocator inserted, such as
# `# 64-byte Spill` or `# 64-byte Folded Reload`; an extended regex, for grep -E
# as well as Python.  CompXSMM's kernels fit in the register file, so the build
# fails if llc's output ever contains one.
SPILL_ANNOTATION = "[0-9]+-byte (Folded )?(Spill|Reload)"
