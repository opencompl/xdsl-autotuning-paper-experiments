from autotuner.tile_l1 import Tile, TileOne


def test_subtiles():
    assert tuple(Tile(5, 5).iter_subtiles()) == (
        TileOne(Tile(5, 5), Tile(1, 1)),
        TileOne(Tile(5, 1), Tile(1, 5)),
        TileOne(Tile(1, 5), Tile(5, 1)),
        TileOne(Tile(1, 1), Tile(5, 5)),
    )

    assert tuple(Tile(6, 6).iter_subtiles()) == (
        TileOne(Tile(6, 6), Tile(1, 1)),
        TileOne(Tile(6, 3), Tile(1, 2)),
        TileOne(Tile(6, 2), Tile(1, 3)),
        TileOne(Tile(6, 1), Tile(1, 6)),
        TileOne(Tile(3, 6), Tile(2, 1)),
        TileOne(Tile(3, 3), Tile(2, 2)),
        TileOne(Tile(3, 2), Tile(2, 3)),
        TileOne(Tile(3, 1), Tile(2, 6)),
        TileOne(Tile(2, 6), Tile(3, 1)),
        TileOne(Tile(2, 3), Tile(3, 2)),
        TileOne(Tile(2, 2), Tile(3, 3)),
        TileOne(Tile(2, 1), Tile(3, 6)),
        TileOne(Tile(1, 6), Tile(6, 1)),
        TileOne(Tile(1, 3), Tile(6, 2)),
        TileOne(Tile(1, 2), Tile(6, 3)),
        TileOne(Tile(1, 1), Tile(6, 6)),
    )
