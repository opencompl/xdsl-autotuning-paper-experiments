from xdsl.ir import Block, Region
from xdsl.dialects import x86_func, x86
from analyzers import LLVM_MCA


def build_basic_block():
    input_types = [x86.register.YMM0, x86.register.YMM1, x86.register.YMM2]

    new_block = Block()
    for i, a in enumerate(input_types):
        new_block.insert_arg(a, i)
    new_region = Region(new_block)
    new_func = x86_func.FuncOp(
        "f",
        new_region,
        (input_types, []),
        visibility="public",
    )

    vfma = x86.ops.RRR_Vfmadd231pdOp(
        new_block.args[0], new_block.args[1], new_block.args[2], result=input_types[0]
    )
    new_block.add_op(vfma)

    return new_func


def main():
    bb = build_basic_block()
    analyzer = LLVM_MCA(arch="x86-64", microarch="skylake")
    cost = analyzer.consistently_evaluate(bb)
    print(cost)


if __name__ == "__main__":
    main()
