"""AMX tile configurations and the code that loads them.

`ldtilecfg` takes a 64-byte operand that fixes, for each of the eight tile
registers, how many rows it has and how many bytes each row holds. LIBXSMM's
AMX kernels describe their tiles once per block of code and load the result
before the block; both the LIBXSMM port and the AMX nano-kernel do the same
through `load_tile_config`.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from xdsl.dialects import x86
from xdsl.ir import Operation

TILE_REGISTERS = 8
# The widest row a tile has in palette 1, in bytes, and the most rows.
MAX_TILE_BYTES_PER_ROW = 64
MAX_TILE_ROWS = 16


@dataclass(frozen=True)
class TileShape:
    """The rows of one tile register and the bytes of each of them."""

    rows: int
    bytes_per_row: int

    def __post_init__(self) -> None:
        if not 0 < self.rows <= MAX_TILE_ROWS:
            raise ValueError(f"a tile has 1 to {MAX_TILE_ROWS} rows, not {self.rows}")
        if not 0 < self.bytes_per_row <= MAX_TILE_BYTES_PER_ROW:
            raise ValueError(
                f"a tile row has 1 to {MAX_TILE_BYTES_PER_ROW} bytes, "
                f"not {self.bytes_per_row}"
            )


@dataclass(frozen=True)
class TileConfig:
    """A palette-1 tile configuration, by tile register index.

    Registers mapped to None are left unconfigured, so any use of them faults.
    """

    tiles: tuple[TileShape | None, ...]

    def __post_init__(self) -> None:
        if len(self.tiles) > TILE_REGISTERS:
            raise ValueError(
                f"there are {TILE_REGISTERS} tile registers, not {len(self.tiles)}"
            )

    def to_bytes(self) -> bytes:
        """The 64 bytes `ldtilecfg` loads: the palette, then each register's
        bytes per row as little-endian 16-bit words from byte 16, then its rows
        from byte 48."""
        data = bytearray(64)
        data[0] = 1
        for index, shape in enumerate(self.tiles):
            if shape is None:
                continue
            data[16 + 2 * index : 18 + 2 * index] = shape.bytes_per_row.to_bytes(
                2, "little"
            )
            data[48 + index] = shape.rows
        return bytes(data)


def tile_config_ops(config: TileConfig, label: str) -> Sequence[Operation]:
    """The operations that load `config`: its bytes, under `label` in the
    read-only data section, then an `ldtilecfg` of them.

    LIBXSMM's text emitter instead jumps over the bytes, placed in the code
    itself; keeping them out of the instruction stream loads the same bytes
    without the jump.
    """
    data = config.to_bytes()
    return (
        x86.ops.DirectiveOp(".pushsection", ".rodata"),
        x86.ops.DirectiveOp(".p2align", "6"),
        x86.ops.LabelOp(label),
        *(
            x86.ops.DirectiveOp(
                ".byte", ", ".join(f"0x{byte:02x}" for byte in data[i : i + 4])
            )
            for i in range(0, len(data), 4)
        ),
        x86.ops.DirectiveOp(".popsection", None),
        x86.ops.L_LdtilecfgOp(label),
    )


def load_tile_config(
    insert: Callable[[Operation], object], config: TileConfig, label: str
) -> None:
    """Insert, one by one through `insert`, the operations that load `config`."""
    for op in tile_config_ops(config, label):
        insert(op)
