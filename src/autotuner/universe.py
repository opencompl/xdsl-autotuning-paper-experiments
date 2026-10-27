from xdsl.universe import Universe

from autotuner.dialects.x86_ext import X86Ext
from autotuner.dialects.xsmm import XSMM
from autotuner.passes.test_xsmm_tiling import TestXsmmTilingPass
from autotuner.passes.vectorize_libxsmm import VectorizeLibxsmmPass
from autotuner.passes.x86_allocate_registers_with_masks import (
    X86AllocateRegistersWithMasksPass,
)
from autotuner.passes.xsmm_apply_exp_schedule import XsmmApplyExpSchedulePass
from autotuner.passes.xsmm_apply_schedule import XsmmApplySchedulePass
from autotuner.passes.xsmm_expand_horner import XsmmExpandHornerPass

AUTOTUNER_UNIVERSE = Universe(
    all_dialects={
        "x86_ext": lambda: X86Ext,
        "xsmm": lambda: XSMM,
    },
    all_passes={
        "test-xsmm-tiling": lambda: TestXsmmTilingPass,
        "vectorize-libxsmm": lambda: VectorizeLibxsmmPass,
        "x86-allocate-registers-with-masks": lambda: X86AllocateRegistersWithMasksPass,
        "xsmm-apply-exp-schedule": lambda: XsmmApplyExpSchedulePass,
        "xsmm-apply-schedule": lambda: XsmmApplySchedulePass,
        "xsmm-expand-horner": lambda: XsmmExpandHornerPass,
    },
)
