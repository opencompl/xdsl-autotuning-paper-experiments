from xdsl.universe import Universe

from autotuner.dialects.xsmm import XSMM
from autotuner.passes.convert_xsmm_to_x86 import ConvertXsmmToX86Pass
from autotuner.passes.vectorize_libxsmm import VectorizeLibxsmmPass
from autotuner.passes.xsmm_matmul_m_to_k import XsmmMatmulMToKPass
from autotuner.passes.xsmm_tile_k import XsmmTileKPass

AUTOTUNER_UNIVERSE = Universe(
    all_dialects={
        "xsmm": lambda: XSMM,
    },
    all_passes={
        "convert-xsmm-to-x86": lambda: ConvertXsmmToX86Pass,
        "vectorize-libxsmm": lambda: VectorizeLibxsmmPass,
        "xsmm-matmul-m-to-k": lambda: XsmmMatmulMToKPass,
        "xsmm-tile-k": lambda: XsmmTileKPass,
    },
)
