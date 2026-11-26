from xdsl.universe import Universe

from autotuner.dialects.fft import FFT
from autotuner.dialects.x86ext import X86EXT
from autotuner.dialects.xsmm import XSMM
from autotuner.passes.test_xsmm_tiling import TestXsmmTilingPass
from autotuner.passes.vectorize_libxsmm import VectorizeLibxsmmPass
from autotuner.passes.xsmm_apply_schedule import XsmmApplySchedulePass
from autotuner.x86_mir import X86MirTarget

AUTOTUNER_UNIVERSE = Universe(
    all_dialects={
        "xsmm": lambda: XSMM,
        "fft": lambda: FFT,
        "x86ext": lambda: X86EXT,
    },
    all_passes={
        "test-xsmm-tiling": lambda: TestXsmmTilingPass,
        "vectorize-libxsmm": lambda: VectorizeLibxsmmPass,
        "xsmm-apply-schedule": lambda: XsmmApplySchedulePass,
    },
    all_targets={
        "x86-mir": lambda: X86MirTarget,
    },
)
