"""AMX blocking, tile configurations, and AMX kernels run in an emulator.

There is no AMX hardware to test on, so the generated kernels run in
`amx_emulator`, which interprets their disassembly and faults on tile shapes
the hardware would fault on.
"""

import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from amx_emulator import EmulationError, convert_fp32_to_bfloat16, run_gemm

from autotuner.amx import TileConfig, TileShape
from autotuner.spr_amx_bf16_nano_kernel import (
    AmxCodeBlock,
    AmxTileLayout,
    amx_code_blocks,
    k_blocking,
)

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / ".venv" / "bin"


def test_tile_config_bytes() -> None:
    config = TileConfig((TileShape(9, 64), None, TileShape(1, 4))).to_bytes()
    assert len(config) == 64
    assert config[0] == 1
    assert config[16:22] == bytes((64, 0, 0, 0, 4, 0))
    assert config[48:51] == bytes((9, 0, 1))
    assert not any(config[1:16]) and not any(config[22:48]) and not any(config[51:])


def test_tile_shape_bounds() -> None:
    with pytest.raises(ValueError):
        TileShape(17, 64)
    with pytest.raises(ValueError):
        TileShape(16, 65)


def test_k_blocking() -> None:
    assert [k_blocking(k) for k in (2, 6, 32, 34, 64, 96)] == [2, 6, 32, 2, 32, 32]


@pytest.mark.parametrize(
    ("shape", "blocks"),
    (
        # One block of 2x2 tiles.
        ((64, 64, 64), (AmxCodeBlock(64, 32, 32, 32),)),
        # 34 has no good M divisor: a 32-row block and a 2-row remainder.
        ((34, 34, 34), (AmxCodeBlock(32, 32, 17, 2), AmxCodeBlock(2, 2, 17, 2))),
        # N of one tile: up to four tiles of M ("Mx1").
        ((40, 4, 32), (AmxCodeBlock(40, 40, 4, 32),)),
        ((70, 16, 8), (AmxCodeBlock(64, 64, 16, 8), AmxCodeBlock(6, 6, 16, 8))),
        ((48, 48, 96), (AmxCodeBlock(48, 24, 24, 32),)),
    ),
)
def test_code_blocks_follow_libxsmm(
    shape: tuple[int, int, int], blocks: tuple[AmxCodeBlock, ...]
) -> None:
    assert amx_code_blocks(*shape) == blocks


def test_tile_configs_match_libxsmm() -> None:
    # The two configurations LIBXSMM's generator loads for 34x34x34, as bytes
    # 16-31 (bytes per row of each tile) and 48-55 (rows of each tile).
    first, second = (
        block.layout().tile_config() for block in amx_code_blocks(34, 34, 34)
    )
    assert first.to_bytes()[16:32] == bytes([64, 0] * 6 + [4, 0] * 2)
    assert first.to_bytes()[48:56] == bytes((9, 8, 9, 8, 1, 1, 9, 8))
    # The remainder block has one M tile of two rows: its accumulators and its
    # A tile are 8 bytes wide, and the tiles it leaves unused keep 16 x 64.
    assert second.to_bytes()[16:32] == bytes(
        [8, 0] * 2 + [64, 0] * 2 + [8, 0, 64, 0] + [4, 0] * 2
    )
    assert second.to_bytes()[48:56] == bytes((9, 8, 16, 16, 1, 16, 9, 8))


def test_layout_rejects_blocks_without_room() -> None:
    with pytest.raises(ValueError):
        AmxTileLayout.for_block(33, 17, 32)
    with pytest.raises(ValueError):
        AmxTileLayout.for_block(65, 16, 32)
    with pytest.raises(ValueError):
        AmxTileLayout.for_block(16, 16, 3)


needs_tools = pytest.mark.skipif(
    platform.system() != "Linux"
    or platform.machine() != "x86_64"
    or not all(shutil.which(tool) for tool in ("clang", "objdump"))
    or not (BIN / "xdsl-opt").exists(),
    reason="links x86-64 ELF kernels with clang and disassembles them with objdump",
)

PIPELINES = {
    "xdsl_libxsmm": ((), "x86-prologue-epilogue-insertion"),
    "compxsmm": (
        ("--disable-regalloc",),
        (
            "xsmm-apply-schedule{strategy=libxsmm-spr-bf16-amx disable-regalloc=true},"
            "x86-regalloc-verify-liveness,x86-allocate-registers{arch=avx512},"
            "convert-x86-scf-to-x86,x86-prologue-epilogue-insertion"
        ),
    ),
    "compxsmm_manual": (
        (),
        (
            "xsmm-apply-schedule{strategy=libxsmm-spr-bf16-amx},"
            "x86-regalloc-verify-liveness,convert-x86-scf-to-x86,"
            "x86-prologue-epilogue-insertion"
        ),
    ),
}


def _build(
    variant: str,
    m: int,
    n: int,
    k: int,
    directory: Path,
    precision: str = "BF16F32_VNNI_A",
) -> Path:
    extra, pipeline = PIPELINES[variant]
    mlir, asm, library = (
        directory / f"kernel{suffix}" for suffix in (".mlir", ".s", ".so")
    )
    subprocess.run(
        (
            BIN / ("libxsmm-gemm" if variant == "xdsl_libxsmm" else "compxsmm-gemm"),
            "dense",
            mlir,
            "matmul",
            *map(str, (m, n, k, m, k, m)),
            *("1", "1", "1", "1", "spr", "nopf", precision),
            *extra,
        ),
        check=True,
    )
    subprocess.run(
        (BIN / "xdsl-opt", mlir, "-p", pipeline, "-t", "x86-asm", "-o", asm), check=True
    )
    subprocess.run(("clang", "-fPIC", "-shared", asm, "-o", library), check=True)
    return library


@needs_tools
@pytest.mark.parametrize("variant", PIPELINES)
@pytest.mark.parametrize(
    "shape",
    ((32, 32, 32), (64, 64, 64), (34, 34, 34), (40, 4, 32), (70, 16, 8), (34, 50, 18)),
)
def test_amx_kernel(variant: str, shape: tuple[int, int, int], tmp_path: Path) -> None:
    library = _build(variant, *shape, tmp_path)
    result, reference, emulator = run_gemm(library, *shape)
    np.testing.assert_allclose(result, reference, rtol=1e-5, atol=1e-5)
    m, n, k = shape
    assert emulator.mnemonics["tdpbf16ps"] >= (m // 16) * (n // 16) * (k // 32)


@needs_tools
@pytest.mark.parametrize("variant", PIPELINES)
@pytest.mark.parametrize(
    "shape",
    (
        # Paired stores of two tiles of M, and N in two tiles.
        (32, 32, 32),
        (64, 64, 64),
        # A remainder block of one partial tile, with N loops.
        (34, 34, 34),
        # Mx1: single stores, three or four tiles of M, a partial last one.
        (40, 4, 32),
        (70, 16, 8),
        (18, 2, 2),
        # Paired stores with a partial lower tile, N of one column per block.
        (17, 37, 4),
        (34, 50, 18),
    ),
)
def test_amx_kernel_bf16_c(
    variant: str, shape: tuple[int, int, int], tmp_path: Path
) -> None:
    library = _build(variant, *shape, tmp_path, "BF16_VNNI_A")
    result, reference, emulator = run_gemm(library, *shape, bf16_c=True)
    # The F32 sum, rounded to BF16 once: within half a BF16 unit in the last
    # place, 2^-8 of the value.
    np.testing.assert_allclose(result, reference, rtol=2**-8, atol=1e-6)
    m, n, k = shape
    assert emulator.mnemonics["tdpbf16ps"] >= (m // 16) * (n // 16) * (k // 32)
    # C moves through the scratch: each column of each tile is widened in once
    # and rounded out once, two tiles at a time when their stores pair.
    tile_columns = n * sum(
        len(block.layout().m_tiles) * (block.m // block.m_blocking)
        for block in amx_code_blocks(m, n, k)
    )
    assert emulator.mnemonics["vpmovsxwd"] == tile_columns
    rounded = emulator.mnemonics.get("vcvtneps2bf16", 0) + 2 * emulator.mnemonics.get(
        "vcvtne2ps2bf16", 0
    )
    assert rounded == tile_columns


def test_bf16_rounding_matches_the_sdm() -> None:
    values = np.array(
        [1.0, 1 + 2**-8, 1 + 3 * 2**-8, -(1 + 2**-8 + 2**-20), 1e-40, -1e-40],
        np.float32,
    )
    # Ties go to even; denormals become zeros of their sign.
    expected = np.array([0x3F80, 0x3F80, 0x3F82, 0xBF81, 0x0000, 0x8000], np.uint16)
    np.testing.assert_array_equal(convert_fp32_to_bfloat16(values), expected)
    nan = np.array([0x7F800001], np.uint32).view(np.float32)
    assert convert_fp32_to_bfloat16(nan)[0] == 0x7FC0


@needs_tools
def test_emulator_faults_on_mismatched_tiles(tmp_path: Path) -> None:
    _build("compxsmm_manual", 32, 32, 32, tmp_path)
    asm = (tmp_path / "kernel.s").read_text()
    # Halve the K of the A tiles alone: their rows no longer match B's.
    broken = asm.replace(
        ".byte 0x10, 0x10, 0x10, 0x10\n.byte 0x10, 0x10, 0x10, 0x10",
        ".byte 0x10, 0x10, 0x10, 0x10\n.byte 0x08, 0x08, 0x10, 0x10",
    )
    assert broken != asm
    (tmp_path / "broken.s").write_text(broken)
    subprocess.run(
        (
            "clang",
            "-fPIC",
            "-shared",
            tmp_path / "broken.s",
            "-o",
            tmp_path / "broken.so",
        ),
        check=True,
    )
    with pytest.raises(EmulationError, match="tdpbf16ps"):
        run_gemm(tmp_path / "broken.so", 32, 32, 32)
