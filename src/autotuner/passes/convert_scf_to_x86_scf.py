from typing import Iterable
from xdsl.context import Context
from xdsl.dialects import builtin, scf, x86
from xdsl.dialects.x86.register import X86RegisterType
from xdsl.ir import Block, Operation, SSAValue, Attribute
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from autotuner import x86_scf
from xdsl.backend.x86.lowering.helpers import (
    cast_operands_to_regs,
    scalar_type_to_register_type,
)


def cast_block_args_to_regs(block: Block, rewriter: PatternRewriter):
    """
    Change the type of the block arguments to registers and add cast operations just after
    the block entry.
    """

    for arg in block.args:
        rewriter.insert_op(
            cast_op := builtin.UnrealizedConversionCastOp(
                operands=[arg], result_types=[arg.type]
            ),
            InsertPoint.at_start(block),
        )
        new_val = cast_op.results[0]

        new_type = scalar_type_to_register_type(arg.type).unallocated()
        arg.replace_by_if(new_val, lambda use: use.operation != cast_op)
        rewriter.replace_value_with_new_type(arg, new_type)


def cast_matched_op_results(rewriter: PatternRewriter) -> list[SSAValue]:
    """
    Add cast operations just after the matched operation, to preserve the type validity of
    arguments of uses of results.
    """

    results = [
        builtin.UnrealizedConversionCastOp.get((val,), (val.type,))
        for val in rewriter.current_operation.results
    ]

    for res, result in zip(rewriter.current_operation.results, results):
        for use in set(res.uses):
            # avoid recursion on the casts we just inserted
            if use.operation != result:
                use.operation.operands[use.index] = result.results[0]

    rewriter.insert_op_after_matched_op(results)
    return [result.results[0] for result in results]


def move_to_unallocated_regs(
    values: Iterable[SSAValue],
    value_types: Iterable[Attribute],
) -> tuple[list[Operation], list[SSAValue]]:
    """
    Return move operations to unallocated registers.
    """

    new_ops = list[Operation]()
    new_values = list[SSAValue]()

    for value, value_type in zip(values, value_types, strict=True):
        register_type = scalar_type_to_register_type(value.type)
        assert isinstance(register_type, X86RegisterType)
        move_op = x86.ops.DS_MovOp(value, destination=register_type.unallocated())
        new_ops.append(move_op)
        new_values.append(move_op.destination)

    return new_ops, new_values


class ScfForLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter) -> None:
        lb, ub, step, *args = cast_operands_to_regs(rewriter)
        new_region = rewriter.move_region_contents_to_new_regions(op.body)
        cast_block_args_to_regs(new_region.block, rewriter)
        mv_ops, values = move_to_unallocated_regs(args, op.iter_args.types)
        rewriter.insert_op_before_matched_op(mv_ops)
        cast_matched_op_results(rewriter)
        new_op = x86_scf.ForOp(lb, ub, step, values, new_region)
        mv_res_ops, res_values = move_to_unallocated_regs(
            new_op.results, op.iter_args.types
        )

        rewriter.replace_matched_op((new_op, *mv_res_ops), res_values)


class ScfYieldLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.YieldOp, rewriter: PatternRewriter) -> None:
        rewriter.replace_matched_op(x86_scf.YieldOp(*cast_operands_to_regs(rewriter)))


class ConvertScfToX86ScfPass(ModulePass):
    name = "convert-scf-to-x86-scf"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ScfYieldLowering(),
                    ScfForLowering(),
                ]
            )
        ).rewrite_module(op)
