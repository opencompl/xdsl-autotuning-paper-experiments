from xdsl.universe import Universe

from autotuner.passes.convert_scf_to_x86_scf import ConvertScfToX86ScfPass
from autotuner.passes.convert_x86_scf_to_x86 import ConvertX86ScfToX86Pass
from autotuner.passes.vectorize_libxsmm import VectorizeLibxsmmPass
from autotuner.x86_scf import X86_Scf
from autotuner.passes.reconcile_moves import ReconcileMovesPass

AUTOTUNER_UNIVERSE = Universe(
    all_dialects={
        "x86_scf": lambda: X86_Scf,
    },
    all_passes={
        "convert-x86-scf-to-x86": lambda: ConvertX86ScfToX86Pass,
        "convert-scf-to-x86-scf": lambda: ConvertScfToX86ScfPass,
        "vectorize-libxsmm": lambda: VectorizeLibxsmmPass,
        "reconcile-moves": lambda: ReconcileMovesPass,
    },
)
