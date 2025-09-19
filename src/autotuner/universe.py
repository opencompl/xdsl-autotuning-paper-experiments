from xdsl.universe import Universe

from autotuner.x86_scf import X86_Scf

AUTOTUNER_UNIVERSE = Universe(
    all_dialects={
        "x86_scf": lambda: X86_Scf,
    },
    all_passes={},
)
