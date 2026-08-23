from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, x86
from xdsl.dialects.x86.registers import AVX512MaskRegisterType, GeneralRegisterType
from xdsl.ir import SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulKOp, MatmulMOp
from autotuner.libxsmm_gemm.generator_common import GPRegMapping, MicroKernelConfig
from autotuner.libxsmm_gemm.generator_gemm_common import (
    libxsmm_generator_gemm_init_micro_kernel_config,
    libxsmm_generator_gemm_load_C,
    libxsmm_generator_gemm_store_C,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import ARCH_BY_CODE, Arch
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode
from autotuner.libxsmm_gemm.libxsmm_main import (
    DescDatatype,
    GEMMDescriptor,
    GEMMFlag,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


@dataclass
class ConvertMatmulMToKPattern(RewritePattern):
    """Expose the K body and preserve the pointer semantics of matmul_m."""

    arch: Arch

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulMOp, rewriter: PatternRewriter, /) -> None:
        if isinstance(op.datatype, builtin.Float32Type):
            datatype = Datatype.F32
        elif isinstance(op.datatype, builtin.Float64Type):
            datatype = Datatype.F64
        else:
            raise PassFailedException(
                f"unsupported xsmm.matmul_m datatype {op.datatype}"
            )

        m_blocking = op.m_blocking.value.data
        n_blocking = op.n_blocking.value.data
        k = op.k.value.data
        lda = op.lda.value.data
        flags = GEMMFlag.NONE
        if op.aligned_a.value.data:
            flags |= GEMMFlag.ALIGN_A
        if op.aligned_c.value.data:
            flags |= GEMMFlag.ALIGN_C

        desc = GEMMDescriptor(
            m=m_blocking,
            n=n_blocking,
            k=k,
            lda=lda,
            ldb=op.ldb.value.data,
            ldc=op.ldc.value.data,
            datatype=DescDatatype(datatype, datatype, datatype, datatype),
            flags=flags,
            prefetch=GEMMPrefetchType.NONE,
        )
        micro_kernel_config = libxsmm_generator_gemm_init_micro_kernel_config(
            MicroKernelConfig(),
            self.arch,
            desc,
            use_masking_a_c=op.mask is not None,
        )
        generated_code = GeneratedCode(rewriter, self.arch)
        gp_reg_mapping = GPRegMapping()

        mask = (
            None
            if op.mask is None
            else SSAValue.get(op.mask, type=AVX512MaskRegisterType)
        )
        c = SSAValue.get(op.c, type=GeneralRegisterType)
        accumulators = libxsmm_generator_gemm_load_C(
            generated_code,
            gp_reg_mapping,
            micro_kernel_config,
            desc,
            m_blocking,
            n_blocking,
            c_val=c,
            mask_k1=mask,
        )
        matmul_k = generated_code.insert(
            MatmulKOp(
                op.a,
                op.b,
                op.c,
                op.rbp,
                op.rsp,
                op.mask,
                accumulators,
                m_blocking=m_blocking,
                n_blocking=n_blocking,
                k_blocking=k,
                lda=lda,
                ldb=op.ldb.value.data,
                datatype=op.datatype,
                aligned_a=bool(op.aligned_a),
            )
        )

        element_size = op.datatype.bitwidth // 8
        b_out = generated_code.insert(
            x86.ops.RI_SubOp(
                matmul_k.b_out,
                k * element_size,
                register_out=op.b_out.type,
            )
        ).register_out

        libxsmm_generator_gemm_store_C(
            generated_code,
            gp_reg_mapping,
            micro_kernel_config,
            desc,
            m_blocking,
            n_blocking,
            c_val=SSAValue.get(matmul_k.c_out, type=GeneralRegisterType),
            acc_vectors=tuple(matmul_k.accumulator_outs),
            mask_k1=(
                None
                if matmul_k.mask_out is None
                else SSAValue.get(matmul_k.mask_out, type=AVX512MaskRegisterType)
            ),
        )

        c_out = generated_code.insert(
            x86.ops.RI_AddOp(
                matmul_k.c_out,
                m_blocking * element_size,
                register_out=op.c_out.type,
            )
        ).register_out
        a_out = generated_code.insert(
            x86.ops.RI_SubOp(
                matmul_k.a_out,
                (k * lda - m_blocking) * element_size,
                register_out=op.a_out.type,
            )
        ).register_out

        results = (
            a_out,
            b_out,
            c_out,
            matmul_k.rbp_out,
            matmul_k.rsp_out,
            *((matmul_k.mask_out,) if matmul_k.mask_out is not None else ()),
        )
        rewriter.replace(op, [], results)


@dataclass(frozen=True)
class XsmmMatmulMToKPass(ModulePass):
    """Lower matmul_m to C accesses, matmul_k, and pointer adjustments.

    The generated adjustments preserve the M operation's pointer semantics:
    A and C advance by the M block, while B remains unchanged. These results do
    not depend on whether a later transformation tiles the generated matmul_k.
    """

    name = "xsmm-matmul-m-to-k"

    arch: str = "skx"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        try:
            arch = ARCH_BY_CODE[self.arch]
        except KeyError as error:
            raise PassFailedException(
                f"unknown architecture code '{self.arch}'"
            ) from error
        if not (Arch.LIBXSMM_X86_AVX512_SKX <= arch <= Arch.LIBXSMM_X86_ALLFEAT):
            raise PassFailedException(
                "xsmm-matmul-m-to-k currently supports AVX-512 architectures only"
            )

        PatternRewriteWalker(
            ConvertMatmulMToKPattern(arch),
            apply_recursively=False,
        ).rewrite_module(op)
