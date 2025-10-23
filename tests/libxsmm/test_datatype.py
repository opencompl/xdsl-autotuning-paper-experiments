from autotuner.libxsmm_gemm.libxsmm_main import DescDatatype
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


def test_ab():
    F64 = Datatype.F64
    F32 = Datatype.F32
    assert DescDatatype(F64, F64, F64, F64).ab == F64
    assert DescDatatype(F32, F32, F32, F32).ab == F32
    assert DescDatatype(F64, F32, F64, F64).ab is None
