from xdsl.universe import Universe

from autotuner.passes.vectorize_libxsmm import VectorizeLibxsmmPass

AUTOTUNER_UNIVERSE = Universe(
    all_passes={
        "vectorize-libxsmm": lambda: VectorizeLibxsmmPass,
    },
)
