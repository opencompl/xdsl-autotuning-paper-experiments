from collections.abc import Mapping
from dataclasses import dataclass

from autotuner.avx512_kdot_nano_kernel import KdotWithSkxFallbackNanoKernel
from autotuner.nano_kernel import GemmDescriptor, ISAInfo, NanoKernel, TileSizes
from autotuner.skx_nano_kernel import SKX_NANO_KERNELS, AVX512Info


@dataclass(frozen=True)
class KTilePolicy:
    """Select the register-kernel K extent for one scheduled tile."""

    blocking: int = 4
    threshold: int = 23

    def tile_size(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
        nano_kernel: NanoKernel,
    ) -> int | None:
        if tile.k <= self.threshold:
            return None
        return self.blocking


@dataclass(frozen=True)
class KdotKTilePolicy(KTilePolicy):
    """Keep the complete K reduction when the K-dot nano-kernel can consume it."""

    def tile_size(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        isa_info: ISAInfo,
        nano_kernel: NanoKernel,
    ) -> int | None:
        if isinstance(nano_kernel, KdotWithSkxFallbackNanoKernel) and (
            nano_kernel.uses_kdot(descriptor, tile, isa_info)
        ):
            return None
        return super().tile_size(descriptor, tile, isa_info, nano_kernel)


@dataclass(frozen=True)
class XsmmStrategy:
    """A named policy shared by XSMM scheduling and lowering passes."""

    isa_info: ISAInfo
    nano_kernel: NanoKernel
    k_tiling: KTilePolicy = KTilePolicy()


XSMM_STRATEGIES: Mapping[str, XsmmStrategy] = {
    **{
        name: XsmmStrategy(AVX512Info(), nano_kernel)
        for name, nano_kernel in SKX_NANO_KERNELS.items()
    },
    "zen5-kdot": XsmmStrategy(
        AVX512Info(),
        KdotWithSkxFallbackNanoKernel(SKX_NANO_KERNELS["libxsmm-skx"]),
        KdotKTilePolicy(),
    ),
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
