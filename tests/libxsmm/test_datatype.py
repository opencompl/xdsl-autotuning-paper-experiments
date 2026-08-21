import pytest
from xdsl.dialects.builtin import f32, f64

from autotuner.libxsmm_gemm.libxsmm_main import DescDatatype
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


def test_ab():
    F64 = Datatype.F64
    F32 = Datatype.F32
    assert DescDatatype(F64, F64, F64, F64).ab == F64
    assert DescDatatype(F32, F32, F32, F32).ab == F32
    assert DescDatatype(F64, F32, F64, F64).ab is None


def test_abc():
    F64 = Datatype.F64
    F32 = Datatype.F32
    assert DescDatatype(F64, F64, F64, F64).abc == F64
    assert DescDatatype(F32, F32, F32, F32).abc == F32
    assert DescDatatype(F64, F32, F64, F64).abc is None
    assert DescDatatype(F64, F64, F32, F64).abc is None
    assert DescDatatype(F64, F64, F64, F32).abc == F64


def test_builtin_type():
    assert Datatype.F32.builtin_type == f32
    assert Datatype.F64.builtin_type == f64

    with pytest.raises(NotImplementedError):
        Datatype.I8.builtin_type
