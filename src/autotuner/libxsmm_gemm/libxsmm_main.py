from enum import IntEnum, IntFlag, unique
from typing import NamedTuple

from autotuner.libxsmm_gemm.libxsmm_typedefs import DataType


class DescDataType(NamedTuple):
    a: DataType
    b: DataType
    c: DataType
    comp: DataType

    def ab(self) -> DataType | None:
        return self.a if self.a == self.b else None


class GemmFlag(IntFlag):
    NONE = 0
    # Transpose matrix A
    TRANS_A = 1
    # Transpose matrix B
    TRANS_B = 2
    # Transpose matrix A and B
    TRANS_AB = TRANS_A | TRANS_B
    # Beta=0|1
    BETA_0 = 4
    # Generate aligned load instructions
    ALIGN_A = 8
    # Aligned load/store instructions
    ALIGN_C = 16
    # Aligned C matrix, but using NTS Hint when storing
    ALIGN_C_NTS_HINT = 32 | ALIGN_C
    # AMX hint to avoid tileconfig/release, it's negated bits, so that 0 is default "on"
    NO_RESET_TILECONFIG = 64
    NO_SETUP_TILECONFIG = 128
    # in case of integer GEMM, if A is unsigned
    A_UNSIGNED = 256
    # in case of integer GEMM, if B is unsigned
    B_UNSIGNED = 512
    # in case of integer GEMM, if C is unsigned
    C_UNSIGNED = 1024
    # in case of integer GEMM, if A and B are unsigned
    AB_UNSIGNED = A_UNSIGNED | B_UNSIGNED
    # for low precision we also require up-front packed formats "VNNI" for best performance, this flag indicates A
    VNNI_A = 2048
    # for low precision we also require up-front packed formats "VNNI" for best performance, this flag indicates B
    VNNI_B = 4096
    # for low precision we also require post packed formats "VNNI" for best performance, this flag indicated C
    VNNI_C = 8192
    # use GEMM ABI
    USE_XGEMM_ABI = 16384
    # use XGEMM_EXT ABI
    USE_XGEMM_EXT_ABI = 32768

    # Pseudo-flag denoting big descriptor
    DESC_ISBIG = 65536
    # Batch-reduce Ai * Bi.
    BATCH_REDUCE_ADDRESS = 65536
    # Batch-reduce Ai * Bi.
    BATCH_REDUCE_OFFSET = 131072
    # Batch-reduce Ai * Bi.
    BATCH_REDUCE_STRIDE = 262144
    USE_COL_VEC_SCF = 524288
    USE_COL_VEC_ZPT = 1048576
    INTERPRETE_A_AS_INT4_VNNI2 = 2097152
    INTERPRETE_A_AS_INT4_VNNI8_INTLV = 4194304
    DECOMPRESS_A_VIA_BITMASK = 8388608
    INTERPRETE_A_AS_MXFP4_VNNI2 = 16777216
    USE_MxK_ZPT = 33554432
    USE_MxK_SCF = 67108864
    INTERPRETE_A_AS_MXFP4_VNNI8_INTLV = 134217728
    INTERPRETE_A_AS_INT2_VNNI4_INTLV = 268435456
    INTERPRETE_A_AS_INT1_VNNI4 = 536870912
    # combined types
    ALIGN_C_NTS_HINT_BETA_0 = BETA_0 | ALIGN_C_NTS_HINT
    ALIGN_C_NTS_HINT_BATCH_REDUCE_ADDRESS = BATCH_REDUCE_ADDRESS | ALIGN_C_NTS_HINT
    ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_ADDRESS = (
        BETA_0 | ALIGN_C_NTS_HINT | BATCH_REDUCE_ADDRESS
    )
    ALIGN_C_NTS_HINT_BATCH_REDUCE_OFFSET = BATCH_REDUCE_OFFSET | ALIGN_C_NTS_HINT
    ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_OFFSET = (
        BETA_0 | ALIGN_C_NTS_HINT | BATCH_REDUCE_OFFSET
    )
    ALIGN_C_NTS_HINT_BATCH_REDUCE_STRIDE = BATCH_REDUCE_STRIDE | ALIGN_C_NTS_HINT
    ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_STRIDE = (
        BETA_0 | ALIGN_C_NTS_HINT | BATCH_REDUCE_STRIDE
    )
    ALIGN_C_NTS_HINT_BETA_0_A_UNSIGNED = BETA_0 | ALIGN_C_NTS_HINT | A_UNSIGNED
    ALIGN_C_NTS_HINT_BATCH_REDUCE_ADDRESS_A_UNSIGNED = (
        BATCH_REDUCE_ADDRESS | ALIGN_C_NTS_HINT | A_UNSIGNED
    )
    ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_ADDRESS_A_UNSIGNED = (
        BETA_0 | ALIGN_C_NTS_HINT | BATCH_REDUCE_ADDRESS | A_UNSIGNED
    )
    ALIGN_C_NTS_HINT_BATCH_REDUCE_OFFSET_A_UNSIGNED = (
        BATCH_REDUCE_OFFSET | ALIGN_C_NTS_HINT | A_UNSIGNED
    )
    ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_OFFSET_A_UNSIGNED = (
        BETA_0 | ALIGN_C_NTS_HINT | BATCH_REDUCE_OFFSET | A_UNSIGNED
    )
    ALIGN_C_NTS_HINT_BATCH_REDUCE_STRIDE_A_UNSIGNED = (
        BATCH_REDUCE_STRIDE | ALIGN_C_NTS_HINT | A_UNSIGNED
    )
    ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_STRIDE_A_UNSIGNED = (
        BETA_0 | ALIGN_C_NTS_HINT | BATCH_REDUCE_STRIDE | A_UNSIGNED
    )
    ALIGN_C_NTS_HINT_BETA_0_B_UNSIGNED = BETA_0 | ALIGN_C_NTS_HINT | B_UNSIGNED
    ALIGN_C_NTS_HINT_BATCH_REDUCE_ADDRESS_B_UNSIGNED = (
        BATCH_REDUCE_ADDRESS | ALIGN_C_NTS_HINT | B_UNSIGNED
    )
    ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_ADDRESS_B_UNSIGNED = (
        BETA_0 | ALIGN_C_NTS_HINT | BATCH_REDUCE_ADDRESS | B_UNSIGNED
    )
    ALIGN_C_NTS_HINT_BATCH_REDUCE_OFFSET_B_UNSIGNED = (
        BATCH_REDUCE_OFFSET | ALIGN_C_NTS_HINT | B_UNSIGNED
    )
    ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_OFFSET_B_UNSIGNED = (
        BETA_0 | ALIGN_C_NTS_HINT | BATCH_REDUCE_OFFSET | B_UNSIGNED
    )
    ALIGN_C_NTS_HINT_BATCH_REDUCE_STRIDE_B_UNSIGNED = (
        BATCH_REDUCE_STRIDE | ALIGN_C_NTS_HINT | B_UNSIGNED
    )
    ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_STRIDE_B_UNSIGNED = (
        BETA_0 | ALIGN_C_NTS_HINT | BATCH_REDUCE_STRIDE | B_UNSIGNED
    )
    ALIGN_C_NTS_HINT_BETA_0_AB_UNSIGNED = BETA_0 | ALIGN_C_NTS_HINT | AB_UNSIGNED
    ALIGN_C_NTS_HINT_BATCH_REDUCE_ADDRESS_AB_UNSIGNED = (
        BATCH_REDUCE_ADDRESS | ALIGN_C_NTS_HINT | AB_UNSIGNED
    )
    ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_ADDRESS_AB_UNSIGNED = (
        BETA_0 | ALIGN_C_NTS_HINT | BATCH_REDUCE_ADDRESS | AB_UNSIGNED
    )
    ALIGN_C_NTS_HINT_BATCH_REDUCE_OFFSET_AB_UNSIGNED = (
        BATCH_REDUCE_OFFSET | ALIGN_C_NTS_HINT | AB_UNSIGNED
    )
    ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_OFFSET_AB_UNSIGNED = (
        BETA_0 | ALIGN_C_NTS_HINT | BATCH_REDUCE_OFFSET | AB_UNSIGNED
    )
    ALIGN_C_NTS_HINT_BATCH_REDUCE_STRIDE_AB_UNSIGNED = (
        BATCH_REDUCE_STRIDE | ALIGN_C_NTS_HINT | AB_UNSIGNED
    )
    ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_STRIDE_AB_UNSIGNED = (
        BETA_0 | ALIGN_C_NTS_HINT | BATCH_REDUCE_STRIDE | AB_UNSIGNED
    )
    # Marker flag; do not use
    INVALID = 1073741824


@unique
class GEMMPrefetchType(IntEnum):
    """Enumeration of the available prefetch strategies."""

    # No data-prefetch.
    NONE = 0  # Equivalent to LIBXSMM_PREFETCH_NONE

    # Prefetch PA using accesses to A.
    AL2 = 1

    # Prefetch PA using accesses to B.
    BL2 = 2

    # The following options are currently not used, but included for completeness.
    # Prefetch PA using accesses to C.
    # CL1 = 4
    # AL2BL2 = 3
    # AL2BL2CL1 = 7


class GEMMDescriptor(NamedTuple):
    m: int
    n: int
    k: int
    lda: int
    ldb: int
    ldc: int
    datatype: DescDataType
    flags: GemmFlag
    prefetch: GEMMPrefetchType

    def is_Amxfp4_Bi8_gemm(self) -> bool:
        return (
            GemmFlag.INTERPRETE_A_AS_MXFP4_VNNI8_INTLV in self.flags
            and self.datatype.a == DataType.I8
            and self.datatype.b == DataType.I8
            and self.datatype.c in (DataType.BF16, DataType.F32)
        )

    def is_Amxfp4_Bfp32_gemm(self) -> bool:
        return (
            GemmFlag.INTERPRETE_A_AS_MXFP4_VNNI2 in self.flags
            and self.datatype.a == DataType.I8
            and self.datatype.b == DataType.F32
            and self.datatype.c == DataType.F32
        )

    def is_Amxfp4_Bbf16_gemm(self) -> bool:
        return (
            GemmFlag.INTERPRETE_A_AS_MXFP4_VNNI2 in self.flags
            and self.datatype.a == DataType.I8
            and self.datatype.b == DataType.BF16
            and self.datatype.c in (DataType.BF16, DataType.F32)
        )

    def is_Ai4_Bi8_gemm(self) -> bool:
        return (
            GemmFlag.INTERPRETE_A_AS_INT4_VNNI8_INTLV in self.flags
            and self.datatype.a == DataType.I8
            and self.datatype.b == DataType.I8
            and self.datatype.c == DataType.I32
        )

    def is_Ai2_Bi8_gemm(self) -> bool:
        return (
            GemmFlag.INTERPRETE_A_AS_INT2_VNNI4_INTLV in self.flags
            and self.datatype.a == DataType.I8
            and self.datatype.b == DataType.I8
            and self.datatype.c == DataType.I32
        )

    def is_Ai1_Bi8_gemm(self) -> bool:
        return (
            GemmFlag.INTERPRETE_A_AS_INT1_VNNI4 in self.flags
            and self.datatype.a == DataType.I8
            and self.datatype.b == DataType.I8
            and self.datatype.c == DataType.I32
        )

    def is_Abf8_Bbf16_gemm(self) -> bool:
        return (
            self.datatype.a == DataType.BF8
            and self.datatype.b == DataType.BF16
            and self.datatype.c in (DataType.BF16, DataType.F32)
        )

    def is_Abf8_Bf16_gemm(self) -> bool:
        return (
            self.datatype.a == DataType.BF8
            and self.datatype.b == DataType.F16
            and self.datatype.c in (DataType.F16, DataType.F32)
        )

    def is_Ahf8_Bbf16_gemm(self) -> bool:
        return (
            self.datatype.a == DataType.HF8
            and self.datatype.b == DataType.BF16
            and self.datatype.c in (DataType.BF16, DataType.F32)
        )
