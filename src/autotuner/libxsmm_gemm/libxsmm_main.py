from typing import NamedTuple

from autotuner.libxsmm_gemm.libxsmm_typedefs import DataType


class DescDataType(NamedTuple):
    a: DataType
    b: DataType
    c: DataType
    comp: DataType


class GEMMDescriptor(NamedTuple):
    m: int
    n: int
    k: int
    lda: int
    ldb: int
    ldc: int
    datatype: DescDataType
