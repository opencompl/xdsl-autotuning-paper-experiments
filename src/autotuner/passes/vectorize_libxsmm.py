from dataclasses import dataclass
from itertools import product

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, linalg, ptr, scf, vector
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
    """
    Vectorizes a matmul with a specific pattern:

    Given:
    ```
    C += A * B
    C: M x N, A: M x K, B: K x N
    ```

    Where:
    1. All sizes are static
    2. `N` divides the vector size
    3. `M + M * N // vector size < #registers`

    It will:

    1. Load all of C into vector registers (one vector size * M tile at a time)
    2. For k in K (for loop)
        1. Load the kth column of A and broadcast to vector registers (M registers)
        2. for n in N // vector size (unrolled)
            1. load the vector of elements of B corresponding to the nth column
            2. for m in M (unrolled)
                1. fma the columns of A with the vector of B, accumulating into the element of C
    """

    vector_size: int = field(default=4)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.ops.MatmulOp, rewriter: PatternRewriter, /):
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
                f"Can only tile matrices where the result row length {N} is divisible "
                f"by {self.vector_size}."
            )

        a_strides = a_type.get_strides()
        assert a_strides is not None
        a_leading = a_strides[0]
        assert a_leading is not None

        b_strides = b_type.get_strides()
        assert b_strides is not None
        b_leading = b_strides[0]
        assert b_leading is not None

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
            c_k.name_hint = "K"
            c_n = arith.ConstantOp(builtin.IntegerAttr(N, _index_type)).result
            c_n.name_hint = "N"
            c_a_leading_stride = arith.ConstantOp(
                builtin.IntegerAttr(a_leading, _index_type)
            ).result
            c_a_leading_stride.name_hint = "a_leading_stride"
            c_b_leading_stride = arith.ConstantOp(
                builtin.IntegerAttr(b_leading, _index_type)
            ).result
            c_b_leading_stride.name_hint = "b_leading_stride"

            c_vector_size = arith.ConstantOp(
                builtin.IntegerAttr(self.vector_size, _index_type)
            ).result
            c_vector_size.name_hint = "vec_size"

            a_ptr = ptr.ToPtrOp(a).res
            a_ptr.name_hint = "a_ptr"
            b_ptr = ptr.ToPtrOp(b).res
            b_ptr.name_hint = "b_ptr"
            element_bytes = ptr.TypeOffsetOp(element_type, _index_type).offset
            element_bytes.name_hint = "element_bytes"
            a_leading = arith.MuliOp(element_bytes, c_a_leading_stride).result
            a_leading.name_hint = "a_leading"
            c_vector_bytes = arith.MuliOp(element_bytes, c_vector_size).result
            c_vector_bytes.name_hint = "c_vector_bytes"
            c_row_bytes = arith.MuliOp(element_bytes, c_n).result
            c_row_bytes.name_hint = "c_row_bytes"
            b_leading_bytes = arith.MuliOp(element_bytes, c_b_leading_stride).result
            b_leading_bytes.name_hint = "b_leading_bytes"
            b_increment = arith.SubiOp(b_leading_bytes, c_row_bytes).result
            b_increment.name_hint = "b_increment"

            # Load the rows of C as vectors, potentially multiple vectors per row
            c_vectors = [
                vector.LoadOp(c, (constants[m], constants[n]), vector_type).result
                for n in range(0, N, self.vector_size)
                for m in range(M)
            ]

            for i, (n, m) in enumerate(
                product(range(0, N, self.vector_size), range(M))
            ):
                c_vectors[i].name_hint = f"c_{m}_{n}_init"

            for_loop = scf.ForOp(
                c0,
                c_k,
                constants[1],
                [a_ptr, b_ptr] + c_vectors,
                Region(
                    Block(
                        arg_types=(
                            _index_type,
                            ptr.PtrType(),
                            ptr.PtrType(),
                        )
                        + (vector_type,) * (M * N // self.vector_size)
                    )
                ),
            )

            for_loop.results[0].name_hint = "a_ptr_out"
            for_loop.results[1].name_hint = "b_ptr_out"

            for i, (n, m) in enumerate(
                product(range(0, N, self.vector_size), range(M))
            ):
                for_loop.results[2 + i].name_hint = f"c_{m}_{n}_res"

            with ImplicitBuilder(for_loop.body) as (
                k,
                a_0_k_ptr,
                b_vector_ptr,
                *c_rows,
            ):
                k.name_hint = "k"
                a_0_k_ptr.name_hint = "a_0_k_ptr"
                b_vector_ptr.name_hint = "b_k_0_ptr"

                for i, (n, m) in enumerate(
                    product(range(0, N, self.vector_size), range(M))
                ):
                    c_rows[i].name_hint = f"c_{m}_{n}_in"

                a_m_k_ptrs: list[SSAValue] = [a_0_k_ptr]
                for m in range(1, M):
                    a_m_k_ptr = ptr.PtrAddOp(a_m_k_ptrs[-1], a_leading).result
                    a_m_k_ptrs.append(a_m_k_ptr)
                    a_m_k_ptr.name_hint = f"a_{m}_k_ptr"

                a_m_ks = tuple(
                    ptr.LoadOp(a_col_ptr, element_type).res for a_col_ptr in a_m_k_ptrs
                )
                for m, a_m_k in enumerate(a_m_ks):
                    a_m_k.name_hint = f"a_{m}_k"
                # Broadcast the mth column of A to vectors
                a_col_vectors = tuple(
                    vector.BroadcastOp(a_m_ks[m], vector_type).vector for m in range(M)
                )
                for a_col_vector in a_col_vectors:
                    a_col_vector.name_hint = "a_col_vector"
                fma_results: list[SSAValue] = []
                for n in range(N // self.vector_size):
                    b_vector = ptr.LoadOp(b_vector_ptr, vector_type).res
                    b_vector.name_hint = "b_vector"

                    for m in range(M):
                        fma_results.append(
                            vector.FMAOp(
                                a_col_vectors[m],
                                b_vector,
                                c_rows[n * M + m],
                            ).res
                        )

                    # Next vector in B
                    b_vector_ptr = ptr.PtrAddOp(b_vector_ptr, c_vector_bytes).result
                    b_vector_ptr.name_hint = "b_vector_ptr"

                # The pointer has advanced past the last element of B in the row, but
                # the next row of B is potentially further away due to tiling, so it
                # must be incremented by the difference
                b_vector_ptr = ptr.PtrAddOp(b_vector_ptr, b_increment).result
                b_vector_ptr.name_hint = "b_vector_ptr"

                a_0_k_plus_one_ptr = ptr.PtrAddOp(a_0_k_ptr, element_bytes).result
                a_0_k_plus_one_ptr.name_hint = "a_0_k_plus_one_ptr"

                for i, (n, m) in enumerate(
                    product(range(0, N, self.vector_size), range(M))
                ):
                    fma_results[i].name_hint = f"c_{m}_{n}_out"

                scf.YieldOp(a_0_k_plus_one_ptr, b_vector_ptr, *fma_results)

            for i, (n, m) in enumerate(
                product(range(0, N, self.vector_size), range(M))
            ):
                vector.StoreOp(
                    for_loop.results[2 + i],
                    c,
                    (constants[m], constants[n]),
                )

        rewriter.erase(op)


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
