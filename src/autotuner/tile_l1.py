from itertools import product
from typing import NamedTuple
from xdsl.transforms.memref_stream_interleave import factors as _factors

from functools import lru_cache

factors = lru_cache(maxsize=128, typed=True)(_factors)


class Tile(NamedTuple):
    rows: int
    cols: int

    def iter_subtiles(self):
        yield from (
            TileOne(
                Tile(self.rows // tile_rows, self.cols // tile_cols),
                Tile(tile_rows, tile_cols),
            )
            for tile_rows, tile_cols in product(factors(self.rows), factors(self.cols))
        )


class TileOne(NamedTuple):
    outer: Tile
    inner: Tile

    def rows(self) -> int:
        return self.outer.rows * self.inner.rows

    def cols(self) -> int:
        return self.outer.cols * self.inner.cols
