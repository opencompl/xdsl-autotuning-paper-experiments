"""Instructions needed by the exp port but absent from our pinned xDSL.

Keep these local until they can be upstreamed; the xDSL dependency is unchanged.
Like xDSL's x86 ops, operands and results carry explicit register assignments.
"""

from xdsl.backend.assembly_printer import reg
from xdsl.backend.register_allocatable import RegisterConstraints
from xdsl.dialects.builtin import IntegerAttr, StringAttr, UnitAttr, i64
from xdsl.dialects.x86.assembly import AssemblyInstructionArg, masked_source_str
from xdsl.dialects.x86.ops import (
    DS_Operation,
    DSI8_Operation,
    DSS_Operation,
    DSSI_Operation,
    RSS_Operation,
    X86AsmOperation,
    X86Instruction,
)
from xdsl.dialects.x86.registers import AVX512MaskRegisterType, AVX512RegisterType
from xdsl.ir import Dialect, SSAValue
from xdsl.irdl import (
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
    traits_def,
)
from xdsl.traits import MemoryReadEffect


@irdl_op_definition
class DS_Vmovdqu64Op(DS_Operation[AVX512RegisterType, AVX512RegisterType]):
    name = "x86_ext.ds.vmovdqu64"


@irdl_op_definition
class DS_Vcvtps2dqOp(DS_Operation[AVX512RegisterType, AVX512RegisterType]):
    name = "x86_ext.ds.vcvtps2dq"


@irdl_op_definition
class DSI_VrndscalepsOp(DSI8_Operation[AVX512RegisterType, AVX512RegisterType]):
    name = "x86_ext.dsi.vrndscaleps"
    sae = opt_attr_def(UnitAttr)

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return (
            reg(self.destination),
            reg(self.source),
            "{sae}" if self.sae is not None else None,
            self.immediate,
        )


@irdl_op_definition
class DSI_VpslldOp(DSI8_Operation[AVX512RegisterType, AVX512RegisterType]):
    name = "x86_ext.dsi.vpslld"


@irdl_op_definition
class DSS_VsubpsOp(
    DSS_Operation[AVX512RegisterType, AVX512RegisterType, AVX512RegisterType]
):
    name = "x86_ext.dss.vsubps"


@irdl_op_definition
class DSS_VmulpsOp(
    DSS_Operation[AVX512RegisterType, AVX512RegisterType, AVX512RegisterType]
):
    name = "x86_ext.dss.vmulps"


@irdl_op_definition
class DSS_VscalefpsOp(
    DSS_Operation[AVX512RegisterType, AVX512RegisterType, AVX512RegisterType]
):
    name = "x86_ext.dss.vscalefps"


@irdl_op_definition
class DSS_VminpsOp(
    DSS_Operation[AVX512RegisterType, AVX512RegisterType, AVX512RegisterType]
):
    name = "x86_ext.dss.vminps"


@irdl_op_definition
class DSS_VmaxpsOp(
    DSS_Operation[AVX512RegisterType, AVX512RegisterType, AVX512RegisterType]
):
    name = "x86_ext.dss.vmaxps"


@irdl_op_definition
class DSS_VpadddOp(
    DSS_Operation[AVX512RegisterType, AVX512RegisterType, AVX512RegisterType]
):
    name = "x86_ext.dss.vpaddd"


@irdl_op_definition
class RSS_Vfmadd213psOp(
    RSS_Operation[AVX512RegisterType, AVX512RegisterType, AVX512RegisterType]
):
    name = "x86_ext.rss.vfmadd213ps"


@irdl_op_definition
class RSS_Vfnmadd231psOp(
    RSS_Operation[AVX512RegisterType, AVX512RegisterType, AVX512RegisterType]
):
    name = "x86_ext.rss.vfnmadd231ps"


@irdl_op_definition
class DSSI_VcmppsOp(
    DSSI_Operation[AVX512MaskRegisterType, AVX512RegisterType, AVX512RegisterType]
):
    name = "x86_ext.dssi.vcmpps"


@irdl_op_definition
class DSSK_VblendmpsOp(X86Instruction):
    name = "x86_ext.dssk.vblendmps"
    source1 = operand_def(AVX512RegisterType)
    source2 = operand_def(AVX512RegisterType)
    mask = operand_def(AVX512MaskRegisterType)
    destination = result_def(AVX512RegisterType)
    assembly_format = (
        "$source1 `,` $source2 `,` $mask attr-dict `:` "
        "functional-type(operands, results)"
    )

    def __init__(
        self,
        source1: SSAValue,
        source2: SSAValue,
        mask: SSAValue,
        *,
        destination: AVX512RegisterType,
    ):
        super().__init__(operands=[source1, source2, mask], result_types=[destination])

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return (
            masked_source_str(self.destination, self.mask, None),
            reg(self.source1),
            reg(self.source2),
        )

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints(
            (self.source1, self.source2, self.mask), (self.destination,), ()
        )


@irdl_op_definition
class InlineConstantOp(X86AsmOperation):
    """A 64-byte splat embedded in text, with a jump over its payload.

    This is data emission, not a control-flow edge: execution continues past
    the payload. LIBXSMM deliberately pays this jump once per constant/load.
    """

    name = "x86_ext.inline_constant"
    label = attr_def(StringAttr)
    bits = attr_def(IntegerAttr)
    assembly_format = "$label `,` $bits attr-dict"

    def __init__(self, label: str, bits: int):
        super().__init__(
            attributes={"label": StringAttr(label), "bits": IntegerAttr(bits, i64)}
        )

    def assembly_line(self) -> str:
        label = self.label.data
        values = ", ".join([f"0x{self.bits.value.data:08x}"] * 16)
        return f"    jmp {label}_end\n{label}:\n    .long {values}\n{label}_end:"


@irdl_op_definition
class DM_VmovupsRipOp(X86Instruction):
    """Load a full constant vector from a RIP-relative assembler label."""

    name = "x86_ext.dm.vmovups_rip"
    label = attr_def(StringAttr)
    destination = result_def(AVX512RegisterType)
    assembly_format = "$label attr-dict `:` type($destination)"
    traits = traits_def(MemoryReadEffect())

    def __init__(self, label: str, *, destination: AVX512RegisterType):
        super().__init__(
            attributes={"label": StringAttr(label)}, result_types=[destination]
        )

    def assembly_instruction_name(self) -> str:
        return "vmovups"

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return reg(self.destination), f"[rip + {self.label.data}]"

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints((), (self.destination,), ())


X86Ext = Dialect(
    "x86_ext",
    [
        DS_Vmovdqu64Op,
        DS_Vcvtps2dqOp,
        DSI_VrndscalepsOp,
        DSI_VpslldOp,
        DSS_VsubpsOp,
        DSS_VmulpsOp,
        DSS_VscalefpsOp,
        DSS_VminpsOp,
        DSS_VmaxpsOp,
        DSS_VpadddOp,
        RSS_Vfmadd213psOp,
        RSS_Vfnmadd231psOp,
        DSSI_VcmppsOp,
        DSSK_VblendmpsOp,
        InlineConstantOp,
        DM_VmovupsRipOp,
    ],
)
