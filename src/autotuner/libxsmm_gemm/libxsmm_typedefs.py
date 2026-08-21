from enum import StrEnum, auto

from xdsl.dialects.builtin import Float32Type, Float64Type, f32, f64


class Datatype(StrEnum):
    # Order preserved from libxsmm
    F64 = auto()
    F32 = auto()
    BF16 = auto()
    F16 = auto()
    BF8 = auto()
    HF8 = auto()
    I64 = auto()
    U64 = auto()
    I32 = auto()
    U32 = auto()
    I16 = auto()
    U16 = auto()
    I8 = auto()
    U8 = auto()
    IMPLICIT = auto()
    UNSUPPORTED = auto()

    @property
    def builtin_type(self) -> Float32Type | Float64Type:
        match self:
            case Datatype.F32:
                return f32
            case Datatype.F64:
                return f64
            case _:
                raise NotImplementedError(
                    f"Unsupported builtin floating-point type for {self}"
                )
