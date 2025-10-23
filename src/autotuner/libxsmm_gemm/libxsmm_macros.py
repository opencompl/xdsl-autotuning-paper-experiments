from autotuner.libxsmm_gemm.libxsmm_main import GEMMFlag


def gemm_flags(transa: str, transb: str) -> GEMMFlag:
    """
    Consolidate BLAS-transpose indicators into a set of flags as in C macro LIBXSMM_GEMM_FLAGS.

    Args:
        transa: Transpose argument for A. Should be a single character (e.g. 'N', 'n', 'T', 't', etc).
        transb: Transpose argument for B. Should be a single character.

    Returns:
        Integer value representing the combination of GEMMFlag.TRANS_A and GEMMFlag.TRANS_B as needed.
        (Assumes GEMMFlag enum is imported.)
    """
    flags = GEMMFlag.NONE

    if transa not in "nN":
        flags |= GEMMFlag.TRANS_A

    if transb not in "nN":
        flags |= GEMMFlag.TRANS_B

    return flags
