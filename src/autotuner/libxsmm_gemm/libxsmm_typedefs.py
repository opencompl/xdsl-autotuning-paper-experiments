from enum import StrEnum, auto


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
