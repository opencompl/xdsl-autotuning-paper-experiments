"""Hand CompXSMM's register-unallocated x86 kernels to LLVM's register allocator.

CompXSMM assigns registers itself and never spills.  To see how LLVM's
allocator fares on exactly the same instructions, the IR just before
`x86-allocate-registers` is printed as LLVM Machine IR (MIR) in SSA form, with
a virtual register for every value, and `llc` is started at PHI elimination.
That skips every SSA-level machine optimisation; with the machine scheduler
also disabled, LLVM keeps our instruction order and only runs two-address
lowering, coalescing, and register allocation (greedy by default) before
emitting code.  The allocator's own optimisation remarks report the spills,
reloads, and copies it inserted.

The only liberties LLVM gets are those any allocator would:

- Values our schedule pins to a register (the pointer arguments and the
  `k1` mask) are virtual, so LLVM may pick other registers for them.
- A `get_avx_register` zeroed by `vpxord` becomes LLVM's zero idiom, which
  LLVM may rematerialise instead of keeping it live.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import IO

import yaml
from xdsl.context import Context
from xdsl.dialects import x86, x86_func, x86_scf
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
JL = 12  # X86::COND_L
JGE = 13  # X86::COND_GE


def is_fp(mnemonic: str) -> bool:
    """Floating-point arithmetic reads MXCSR, which MIR spells out."""
    return mnemonic.startswith(("vfmadd", "vfmsub", "vfnmadd", "vfnmsub")) or (
        mnemonic[:4] in ("vadd", "vsub", "vmul", "vdiv", "vmax", "vmin")
        and mnemonic.endswith(("ps", "pd", "ss", "sd"))
    )


@dataclass
class MirBlock:
    number: int
    lines: list[str] = field(default_factory=list)
    successors: list[int] = field(default_factory=list)


class MirPrinter:
    """Prints one `x86_func.func` with unallocated registers as SSA MIR."""

    def __init__(self, func: x86_func.FuncOp):
        self.func = func
        self.vregs: dict[SSAValue, str] = {}
        self.classes: list[str] = []
        self.blocks: list[MirBlock] = []
        self.zeroed: set[SSAValue] = set()

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

    # Blocks

    def new_block(self) -> MirBlock:
        self.blocks.append(block := MirBlock(len(self.blocks)))
        return block

    def emit(self, block: MirBlock, line: str) -> None:
        block.lines.append(line)

    # Conversion

    def print(self, output: IO[str]) -> None:
        entry = self.new_block()
        body = self.func.body.block
        liveins: list[str] = []
        for arg, default in zip(body.args, GPR_ARGUMENTS, strict=False):
            assert isinstance(arg.type, GeneralRegisterType)
            phys = arg.type.register_name.data or default
            liveins.append(f"${phys}")
            self.emit(entry, f"{self.define(arg)} = COPY ${phys}")
        self.convert_block(body, entry)

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
        for block in self.blocks:
            output.write(f"  bb.{block.number}:\n")
            if block.successors:
                succ = ", ".join(f"%bb.{s}" for s in block.successors)
                output.write(f"    successors: {succ}\n")
            if block is entry and liveins:
                output.write(f"    liveins: {', '.join(liveins)}\n")
            for line in block.lines:
                output.write(f"    {line}\n")
            output.write("\n")
        output.write("...\n")

    def convert_block(self, block: Block, current: MirBlock) -> MirBlock:
        """Append `block`'s operations, returning the MIR block that ends it."""
        for op in block.ops:
            current = self.convert_op(op, current)
        return current

    def convert_op(self, op: Operation, current: MirBlock) -> MirBlock:
        match op:
            case x86_scf.ForOp():
                return self.convert_for(op, current)
            case x86_scf.YieldOp():
                return current  # handled by the enclosing loop
            case x86_func.RetOp():
                self.emit(current, "RET64")
                return current
            case x86.ops.GetAVXRegisterOp() if all(
                isinstance(use.operation, x86.ops.DSS_VpxordOp)
                and use.operation.source1 == use.operation.source2
                for use in op.result.uses
            ):
                return current  # only zeroed: see `vpxord` below
            case x86.ops.DSS_VpxordOp() if op.source1 == op.source2 and isinstance(
                op.source1.owner, x86.ops.GetAVXRegisterOp
            ):
                opcode = SET0[type(op.destination.type)]
                self.emit(current, f"{self.define(op.destination)} = {opcode}")
                return current
            case _:
                self.emit(current, self.instruction(op))
                return current

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

    def convert_for(self, op: x86_scf.ForOp, current: MirBlock) -> MirBlock:
        """Lower the loop the way `convert-x86-scf-to-x86` does.

        The loop runs as a do-while; unless the bounds show that it runs at
        least once, a guard skips it.  The induction variable is incremented
        at the top of the body when the body does not read it, as there.
        """
        body = op.body.block
        iv, *args = body.args
        loop = self.new_block()
        exit_ = self.new_block()
        # Loop exit values are PHIs in the exit block when the guard can skip
        # the loop; collect them per predecessor.
        exit_incoming: list[tuple[MirBlock, Sequence[str]]] = []

        inits = [self.use(op.lb), *map(self.use, op.iter_args)]
        lb_owner = op.lb.owner
        runs_once = (
            isinstance(lb_owner, x86.ops.DI_MovOp)
            and isinstance(op.ub, IntegerAttr)
            and lb_owner.immediate.value.data < op.ub.value.data
        )
        if not runs_once:
            self.emit(current, self.compare(op.lb, op.ub))
            self.emit(current, f"JCC_1 %bb.{exit_.number}, {JGE}, implicit $eflags")
            current.successors.append(exit_.number)
            exit_incoming.append((current, inits))
        self.emit(current, f"JMP_1 %bb.{loop.number}")
        current.successors.append(loop.number)

        # PHIs for the induction variable and the iter_args, filled in once
        # the latch is known.
        phis = [self.define(value) for value in (iv, *args)]
        phi_lines = len(loop.lines)
        for _ in phis:
            self.emit(loop, "")

        step = op.step
        iv_used = iv.first_use is not None
        next_iv = self.new_vreg("gr64")
        increment = self.increment(next_iv, phis[0], step)
        if not iv_used:
            self.emit(loop, increment)
        latch = self.convert_block(body, loop)
        if iv_used:
            self.emit(latch, increment)
        yield_op = body.last_op
        assert isinstance(yield_op, x86_scf.YieldOp)
        nexts = [next_iv, *map(self.use, yield_op.operands)]
        self.emit(latch, self.compare_vreg(next_iv, op.ub))
        self.emit(latch, f"JCC_1 %bb.{loop.number}, {JL}, implicit $eflags")
        self.emit(latch, f"JMP_1 %bb.{exit_.number}")
        latch.successors += [loop.number, exit_.number]
        exit_incoming.append((latch, nexts))

        for i, (phi, init, nxt) in enumerate(zip(phis, inits, nexts, strict=True)):
            loop.lines[phi_lines + i] = (
                f"{phi} = PHI {init}, %bb.{current.number}, {nxt}, %bb.{latch.number}"
            )

        for i, result in enumerate((op.lb_end, *op.res)):
            if len(exit_incoming) == 1:
                self.vregs[result] = exit_incoming[0][1][i]
            else:
                incoming = ", ".join(
                    f"{values[i]}, %bb.{block.number}"
                    for block, values in exit_incoming
                )
                self.emit(exit_, f"{self.define(result)} = PHI {incoming}")
        return exit_

    def increment(self, dest: str, iv: str, step: IntegerAttr | SSAValue) -> str:
        if isinstance(step, IntegerAttr):
            operand, opcode = str(step.value.data), "ADD64ri32"
        else:
            operand, opcode = self.use(step), "ADD64rr"
        return f"{dest} = {opcode} {iv}, {operand}, implicit-def dead $eflags"

    def compare(self, lhs: SSAValue, rhs: IntegerAttr | SSAValue) -> str:
        return self.compare_vreg(self.use(lhs), rhs)

    def compare_vreg(self, lhs: str, rhs: IntegerAttr | SSAValue) -> str:
        if isinstance(rhs, IntegerAttr):
            return f"CMP64ri32 {lhs}, {rhs.value.data}, implicit-def $eflags"
        return f"CMP64rr {lhs}, {self.use(rhs)}, implicit-def $eflags"


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


# Running LLVM


@dataclass(frozen=True)
class RegallocStats:
    """What LLVM's register allocator inserted.

    Spills and reloads are LLVM's own `N-byte Spill`/`Reload` annotations of
    stack-slot accesses in the emitted assembly, which every allocator gets;
    a folded reload is a spill slot read directly as a memory operand.
    `copies` are the register copies only the greedy allocator reports.
    """

    spills: int = 0
    folded_spills: int = 0
    reloads: int = 0
    folded_reloads: int = 0
    copies: int = 0

    @property
    def spilled(self) -> bool:
        return any((self.spills, self.folded_spills, self.reloads, self.folded_reloads))


# How llc compiles our MIR, here and in the `compxsmm_llvm` benchmark variant.
LLC_FLAGS = (
    "-O2",
    "-x",
    "mir",
    "-start-before=phi-node-elimination",
    "-enable-misched=false",
)

SPILL_ANNOTATION = re.compile(r"# \d+-byte (Folded )?(Spill|Reload)$", re.MULTILINE)


class RemarkLoader(yaml.SafeLoader):
    """Reads LLVM's remark documents, whose tags (`!Missed` ...) carry no data."""


RemarkLoader.add_multi_constructor(
    "!", lambda loader, _, node: loader.construct_mapping(node, deep=True)
)


def parse_output(asm: str, remarks: str) -> RegallocStats:
    counts = {"copies": 0}
    for folded, kind in SPILL_ANNOTATION.findall(asm):
        name = f"{'folded_' if folded else ''}{kind.lower()}s"
        counts[name] = counts.get(name, 0) + 1
    for remark in yaml.load_all(remarks, Loader=RemarkLoader):
        args = {k: v for arg in remark.get("Args", ()) for k, v in arg.items()}
        if remark["Name"] == "SpillReloadCopies":
            counts["copies"] += int(args.get("NumVRCopies", 0))
    return RegallocStats(**counts)


def llvm_allocate(
    mir: str, mcpu: str, *, regalloc: str = "greedy", asm: Path | None = None
) -> RegallocStats:
    """Allocate `mir` with llc and report what the allocator inserted."""
    with tempfile.TemporaryDirectory() as tmp:
        source, remarks = Path(tmp, "kernel.mir"), Path(tmp, "remarks.yaml")
        asm = asm or Path(tmp, "kernel.s")
        source.write_text(mir)
        subprocess.run(
            [
                "llc",
                *LLC_FLAGS,
                "-mtriple=x86_64-unknown-linux-gnu",
                f"-mcpu={mcpu}",
                "-verify-machineinstrs",
                f"-regalloc={regalloc}",
                f"-pass-remarks-output={remarks}",
                "-pass-remarks-filter=regalloc",
                "-o",
                str(asm),
                str(source),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return parse_output(asm.read_text(), remarks.read_text())


# Command line


def unallocated_passes(config: Path, pipeline: str, isa: str) -> str:
    """A CompXSMM pipeline up to, not including, its register allocation."""
    passes = yaml.safe_load(config.read_text())[pipeline][isa]
    end = next(
        i for i, p in enumerate(passes) if p.startswith("x86-allocate-registers")
    )
    return ",".join(passes[:end])


def kernel_mir(path: Path, passes: str) -> str:
    from io import StringIO

    from xdsl.parser import Parser
    from xdsl.passes import PassPipeline
    from xdsl.universe import Universe

    multiverse = Universe.get_multiverse()
    ctx = Context()
    for name, dialect in multiverse.all_dialects.items():
        ctx.register_dialect(name, dialect)
    module = Parser(ctx, path.read_text(), str(path)).parse_module()
    PassPipeline.parse_spec(multiverse.all_passes, passes).apply(ctx, module)
    out = StringIO()
    print_mir(module, out)
    return out.getvalue()


@dataclass(frozen=True)
class Job:
    path: Path
    passes: str
    mcpu: str
    regalloc: str
    keep: Path | None


def run_job(job: Job) -> tuple[Path, RegallocStats | str]:
    try:
        mir = kernel_mir(job.path, job.passes)
        asm = None
        if job.keep is not None:
            stem = job.keep / job.path.parent.name / job.path.name.split(".")[1]
            stem.parent.mkdir(parents=True, exist_ok=True)
            stem.with_suffix(".mir").write_text(mir)
            asm = stem.with_suffix(".s")
        return job.path, llvm_allocate(mir, job.mcpu, regalloc=job.regalloc, asm=asm)
    except (subprocess.CalledProcessError, NotImplementedError, KeyError) as e:
        detail = e.stderr if isinstance(e, subprocess.CalledProcessError) else e
        return job.path, f"{type(e).__name__}: {detail}".strip()


def kernel_files(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(path.glob("**/compxsmm.*.compxsmm.mlir"))
        else:
            yield path


def main(argv: Sequence[str] | None = None) -> None:
    from autotuner.machines import MACHINES

    parser = argparse.ArgumentParser(
        description="Allocate CompXSMM kernels' registers with LLVM and report spills."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="compxsmm.*.compxsmm.mlir generator outputs, or directories of them",
    )
    parser.add_argument("--machine", default=os.environ.get("MACHINE"))
    parser.add_argument("--config", type=Path, default=Path("default.yaml"))
    parser.add_argument(
        "--pipeline",
        default="compxsmm-gemm-passes",
        help="the pipeline in --config whose allocation LLVM replaces",
    )
    parser.add_argument(
        "--regalloc", default="greedy", choices=["greedy", "basic", "pbqp", "fast"]
    )
    parser.add_argument(
        "--keep", type=Path, help="write each kernel's .mir and allocated .s here"
    )
    parser.add_argument("--csv", type=Path, help="write per-kernel results here")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args(argv)

    if not args.machine:
        parser.error("no machine given; pass --machine or set MACHINE")
    machine = MACHINES[args.machine]
    passes = unallocated_passes(args.config, args.pipeline, machine.isa)
    jobs = [
        Job(path, passes, machine.march, args.regalloc, args.keep)
        for path in kernel_files(args.inputs)
    ]

    names = [f.name for f in fields(RegallocStats)]
    rows: list[tuple[dict[str, object], RegallocStats]] = []
    failures = 0
    with ProcessPoolExecutor(args.jobs) as pool:
        for path, result in pool.map(run_job, jobs, chunksize=4):
            kernel = f"{path.parent.name}/{path.name.split('.')[1]}"
            if isinstance(result, str):
                failures += 1
                print(f"{kernel}: {result}", file=sys.stderr)
                continue
            row = {"kernel": kernel, **{n: getattr(result, n) for n in names}}
            rows.append((row, result))

    if args.csv:
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, ["kernel", *names])
            writer.writeheader()
            writer.writerows(row for row, _ in rows)

    spilled = [row for row, stats in rows if stats.spilled]
    for row in spilled:
        detail = ", ".join(f"{n}={row[n]}" for n in names if row[n])
        print(f"{row['kernel']}: {detail}")
    print(
        f"{len(spilled)} of {len(rows)} kernels spill under LLVM's {args.regalloc} "
        f"allocator ({failures} failed to convert)"
    )
    if failures:
        raise SystemExit(1)
