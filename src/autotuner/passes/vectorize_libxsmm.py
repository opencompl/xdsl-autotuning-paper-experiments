from itertools import product
from dataclasses import dataclass
from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, linalg, vector, scf, ptr
from xdsl.ir import Block, Region, SSAValue, field
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import DiagnosticException, PassFailedException
from xdsl.utils.hints import isa

_index_type = builtin.IndexType()


@dataclass
class VectorizeLibxsmmPattern(RewritePattern):
    vector_size: int = field(default=4)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.MatmulOp, rewriter: PatternRewriter, /):
        # C += A * B
        # C: M x N, A: M x K, B: K x N
        a, b = op.inputs
        c = op.outputs[0]
        a_type = a.type
        b_type = b.type
        c_type = c.type
        # Only handle matmul on memrefs for now
        if (
            not isa(a_type, builtin.MemRefType)
            or not isa(b_type, builtin.MemRefType)
            or not isa(c_type, builtin.MemRefType)
        ):
            raise DiagnosticException(
                "Vectorizing matmul on tensors not yet implemented."
            )
        M, K = a_type.get_shape()
        _K, N = b_type.get_shape()
        _M, _N = c_type.get_shape()

        assert M == _M
        assert N == _N
        assert K == _K

        assert M != -1
        assert N != -1
        assert K != -1

        if N % self.vector_size:
            raise PassFailedException(
                f"Can only tile matrices where the result row length is divisible by {self.vector_size}"
            )

        element_type = a_type.element_type
        vector_type = builtin.VectorType(element_type, (self.vector_size,))

        # All operations created inside this block are inserted before the matched op
        with ImplicitBuilder(rewriter):
            # Insert all the integer constants we'll need to index into the matrices
            constants = tuple(
                arith.ConstantOp(builtin.IntegerAttr(i, _index_type)).result
                for i in range(max(M, N) + 1)
            )
            for i, constant in enumerate(constants):
                constant.name_hint = f"c{i}"
            # Zero for convenience
            c0 = constants[0]
            c_k = arith.ConstantOp(builtin.IntegerAttr(K, _index_type)).result

            a_ptr = ptr.ToPtrOp(a).res
            b_ptr = ptr.ToPtrOp(b).res
            element_size = ptr.TypeOffsetOp(element_type, _index_type).offset
            a_leading = arith.MuliOp(element_size, c_k).result

            a_row_ptrs = [a_ptr]
            for i in range(1, M):
                a_row_ptrs.append(ptr.PtrAddOp(a_row_ptrs[-1], a_leading).result)

            # Load the rows of C as vectors, potentially multiple vectors per row
            c_vectors = [
                vector.LoadOp(c, (constants[m], constants[n]), vector_type).result
                for m in range(M)
                for n in range(0, N, self.vector_size)
            ]

            for_loop = scf.ForOp(
                c0,
                c_k,
                constants[1],
                a_row_ptrs + [b_ptr] + c_vectors,
                Region(
                    Block(
                        arg_types=(
                            _index_type,
                            ptr.PtrType(),
                        )
                        + (ptr.PtrType(),) * M
                        + (vector_type,) * (M * N // self.vector_size)
                    )
                ),
            )

            with ImplicitBuilder(for_loop.body) as (k, *acc):
                a_col_ptrs = acc[:M]
                b_vector_ptr = acc[M]
                acc = acc[M + 1 :]

                a_col = tuple(
                    ptr.LoadOp(a_col_ptr, element_type).res for a_col_ptr in a_col_ptrs
                )
                # Broadcast the mth column of A to vectors
                a_col_vectors = tuple(
                    vector.BroadcastOp(a_col[m], vector_type) for m in range(M)
                )
                fma_results: list[SSAValue] = []
                for n in range(N // self.vector_size):
                    b_vector = ptr.LoadOp(b_vector_ptr, vector_type).res

                    for m in range(M):
                        fma_results.append(
                            vector.FMAOp(
                                a_col_vectors[m],
                                b_vector,
                                acc[m * N // self.vector_size + n],
                            ).res
                        )

                    # Next vector in B
                    b_vector_ptr = ptr.PtrAddOp(
                        b_vector_ptr, constants[self.vector_size]
                    ).result

                new_a_col_ptrs = tuple(
                    ptr.PtrAddOp(prev_ptr, element_size).result
                    for m, prev_ptr in enumerate(a_col_ptrs)
                )

                scf.YieldOp(*new_a_col_ptrs, b_vector_ptr, *fma_results)

            for i, (m, n) in enumerate(
                product(range(M), range(0, N // self.vector_size))
            ):
                vector.StoreOp(
                    for_loop.results[i + M + 1],
                    c,
                    (constants[m], constants[n * self.vector_size]),
                )

        rewriter.erase_matched_op()


@dataclass(frozen=True)
class VectorizeLibxsmmPass(ModulePass):
    """
    A test pass vectorizing linalg.matmul with a specific vectorization strategy.
    """

    name = "vectorize-libxsmm"

    vector_size: int = 4

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            VectorizeLibxsmmPattern(self.vector_size), apply_recursively=False
        ).rewrite_module(op)
