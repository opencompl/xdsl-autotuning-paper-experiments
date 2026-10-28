from collections.abc import Mapping
from dataclasses import dataclass

from autotuner.cpx_bf16_nano_kernel import CpxBf16NanoKernel
from autotuner.cpx_bf16_vnni_nano_kernel import CpxBf16VnniNanoKernel
from autotuner.nano_kernel import ISAInfo, NanoKernel
from autotuner.skx_nano_kernel import SKX_NANO_KERNELS, AVX512Info
from autotuner.spr_amx_bf16_nano_kernel import AMXInfo, SprAmxBf16NanoKernel


@dataclass(frozen=True)
class XsmmStrategy:
    """A named policy shared by XSMM scheduling and lowering passes."""

    isa_info: ISAInfo
    nano_kernel: NanoKernel


XSMM_STRATEGIES: Mapping[str, XsmmStrategy] = {
    **{
        name: XsmmStrategy(AVX512Info(), nano_kernel)
        for name, nano_kernel in (
            *SKX_NANO_KERNELS.items(),
            (CpxBf16NanoKernel().name, CpxBf16NanoKernel()),
            (CpxBf16VnniNanoKernel().name, CpxBf16VnniNanoKernel()),
        )
    },
    SprAmxBf16NanoKernel().name: XsmmStrategy(AMXInfo(), SprAmxBf16NanoKernel()),
}


def get_xsmm_strategy(name: str) -> XsmmStrategy:
    """Return the named XSMM lowering strategy."""
    try:
        return XSMM_STRATEGIES[name]
    except KeyError as error:
        choices = ", ".join(sorted(XSMM_STRATEGIES))
        raise ValueError(
            f"unknown XSMM strategy '{name}'; expected one of: {choices}"
        ) from error
