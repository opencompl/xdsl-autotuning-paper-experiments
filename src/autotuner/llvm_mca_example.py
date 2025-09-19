from typing import cast
from xdsl.dialects.x86.registers import X86VectorRegisterType
from xdsl.ir import Block, BlockArgument, Region
from xdsl.dialects import x86_func, x86
from analyzers import LLVM_MCA


def build_basic_block():
    input_types = [x86.registers.YMM0, x86.registers.YMM1, x86.registers.YMM2]

    new_block = Block(arg_types=input_types)
    block_args = cast(
        tuple[
            BlockArgument[X86VectorRegisterType],
            BlockArgument[X86VectorRegisterType],
            BlockArgument[X86VectorRegisterType],
        ],
        new_block.args,
    )
    new_region = Region(new_block)
    new_func = x86_func.FuncOp(
        "f",
        new_region,
        (input_types, []),
        visibility="public",
    )

    vfma = x86.ops.RSS_Vfmadd231pdOp(block_args[0], block_args[1], block_args[2])
    new_block.add_op(vfma)

    return new_func


def main():
    bb = build_basic_block()
    analyzer = LLVM_MCA(arch="x86-64", microarch="skylake")
    cost = analyzer.consistently_evaluate(bb)
    print(cost)


if __name__ == "__main__":
    main()
