from xdsl.universe import Universe

from autotuner.dialects.xsmm import XSMM
from autotuner.llvm_regalloc import X86MirTarget
from autotuner.passes.test_xsmm_tiling import TestXsmmTilingPass
from autotuner.passes.vectorize_libxsmm import VectorizeLibxsmmPass
from autotuner.passes.xsmm_apply_schedule import XsmmApplySchedulePass

AUTOTUNER_UNIVERSE = Universe(
    all_dialects={
        "xsmm": lambda: XSMM,
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
