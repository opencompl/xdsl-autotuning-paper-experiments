"""Where lanes go through the x86 shuffles the FFT expansions use.

Each function takes vectors as lists of lane labels and returns the labels of
the result, so the expansions can work out lane orders by simulation instead
of by hand. The same simulation checks the transposes in the tests.
"""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

T = TypeVar("T", bound=Hashable)

CHUNK_BITS = 128


def _chunks(vector: Sequence[T], element_bits: int) -> list[list[T]]:
    per_chunk = CHUNK_BITS // element_bits
    return [
        list(vector[start : start + per_chunk])
        for start in range(0, len(vector), per_chunk)
    ]


def unpack_low(a: Sequence[T], b: Sequence[T], element_bits: int) -> list[T]:
    """``vunpcklpd`` (64-bit) or ``vunpcklps`` (32-bit)."""
    result: list[T] = []
    for ca, cb in zip(_chunks(a, element_bits), _chunks(b, element_bits), strict=True):
        if element_bits == 64:
            result += [ca[0], cb[0]]
        else:
            result += [ca[0], cb[0], ca[1], cb[1]]
    return result


def unpack_high(a: Sequence[T], b: Sequence[T], element_bits: int) -> list[T]:
    """``vunpckhpd`` (64-bit) or ``vunpckhps`` (32-bit)."""
    result: list[T] = []
    for ca, cb in zip(_chunks(a, element_bits), _chunks(b, element_bits), strict=True):
        if element_bits == 64:
            result += [ca[1], cb[1]]
        else:
            result += [ca[2], cb[2], ca[3], cb[3]]
    return result


def shuffle_ps(a: Sequence[T], b: Sequence[T], immediate: int) -> list[T]:
    """``vshufps``: per 128-bit chunk, two lanes of ``a`` then two of ``b``."""
    result: list[T] = []
    for ca, cb in zip(_chunks(a, 32), _chunks(b, 32), strict=True):
        result += [
            ca[immediate & 3],
            ca[(immediate >> 2) & 3],
            cb[(immediate >> 4) & 3],
            cb[(immediate >> 6) & 3],
        ]
    return result


def shuffle_128(
    a: Sequence[T], b: Sequence[T], immediate: int, element_bits: int
) -> list[T]:
    """``vshuff64x2``: the lower half from 128-bit chunks of ``a``, the upper of ``b``."""
    ca, cb = _chunks(a, element_bits), _chunks(b, element_bits)
    if len(ca) == 4:
        picked = [
            ca[immediate & 3],
            ca[(immediate >> 2) & 3],
            cb[(immediate >> 4) & 3],
            cb[(immediate >> 6) & 3],
        ]
    elif len(ca) == 2:
        picked = [ca[immediate & 1], cb[(immediate >> 1) & 1]]
    else:
        raise ValueError("vshuff64x2 needs a ymm or zmm register")
    return [lane for chunk in picked for lane in chunk]


ShuffleKind = Literal["unpack_low", "unpack_high", "shuffle_ps", "shuffle_128"]


@dataclass(frozen=True)
class Shuffle:
    """One two-source shuffle: ``kind`` with ``immediate`` where it takes one."""

    kind: ShuffleKind
    immediate: int = 0

    def apply(
        self,
        a: Sequence[T],
        b: Sequence[T],
        element_bits: int,
        lane_bits: int | None = None,
    ) -> list[T]:
        """Shuffle ``element_bits``-wide elements of vectors whose labels are
        ``lane_bits`` wide (by default, as wide as the elements)."""
        lane_bits = element_bits if lane_bits is None else lane_bits
        if lane_bits > element_bits:
            # Split each label into elements, and check none is torn apart.
            parts = lane_bits // element_bits
            moved = self.apply(
                [(lane, part) for lane in a for part in range(parts)],
                [(lane, part) for lane in b for part in range(parts)],
                element_bits,
            )
            result: list[T] = []
            for start in range(0, len(moved), parts):
                lane = moved[start][0]
                if moved[start : start + parts] != [(lane, p) for p in range(parts)]:
                    raise ValueError(f"{self} splits {lane_bits}-bit lanes")
                result.append(lane)
            return result
        group = element_bits // lane_bits
        if group > 1:
            grouped = self.apply(
                [tuple(a[i : i + group]) for i in range(0, len(a), group)],
                [tuple(b[i : i + group]) for i in range(0, len(b), group)],
                element_bits,
            )
            return [lane for element in grouped for lane in element]
        match self.kind:
            case "unpack_low":
                return unpack_low(a, b, element_bits)
            case "unpack_high":
                return unpack_high(a, b, element_bits)
            case "shuffle_ps":
                return shuffle_ps(a, b, self.immediate)
            case "shuffle_128":
                return shuffle_128(a, b, self.immediate, element_bits)


@dataclass(frozen=True)
class ShuffleStep:
    """``registers[destination] = shuffle(registers[a], registers[b])``."""

    destination: int
    a: int
    b: int
    shuffle: Shuffle


def _stage_shuffles(
    block_bits: int, register_bits: int
) -> tuple[Shuffle, Shuffle, int]:
    """The shuffles that interleave blocks of ``block_bits`` from two registers,
    and the width in bits of the unpack they perform on each element."""
    match block_bits:
        case 32:
            return Shuffle("unpack_low"), Shuffle("unpack_high"), 32
        case 64:
            return Shuffle("unpack_low"), Shuffle("unpack_high"), 64
        case 128 if register_bits == 256:
            return Shuffle("shuffle_128", 0b00), Shuffle("shuffle_128", 0b11), 64
        case 128 | 256 if register_bits == 512:
            # Even then odd 128-bit chunks of both sources. Applied twice, this
            # interleaves first 128-bit and then 256-bit blocks.
            return Shuffle("shuffle_128", 0x88), Shuffle("shuffle_128", 0xDD), 64
        case _:
            raise ValueError(f"no shuffle for {block_bits}-bit blocks")


@dataclass(frozen=True)
class TransposeNetwork:
    """A shuffle network that transposes ``lanes`` registers up to lane order.

    Running ``steps`` on registers holding operand ``source_of_slot[s]`` in
    slot ``s`` leaves, in slot ``t``, the natural transpose's result
    ``result_of_slot[t]``.
    """

    lanes: int
    element_bits: int
    steps: tuple[tuple[ShuffleStep, int], ...]
    """Each step with the width of elements its unpack interleaves."""
    source_of_slot: tuple[int, ...]
    result_of_slot: tuple[int, ...]

    def run(self, registers: Sequence[Sequence[T]]) -> list[list[T]]:
        """Simulate the network on registers in slot order."""
        current = [list(register) for register in registers]
        for stage_start in range(0, len(self.steps), self.lanes):
            previous = [list(register) for register in current]
            for step, bits in self.steps[stage_start : stage_start + self.lanes]:
                current[step.destination] = step.shuffle.apply(
                    previous[step.a], previous[step.b], bits, self.element_bits
                )
        return current


def transpose_network(lanes: int, element_bits: int) -> TransposeNetwork:
    """Build and verify the network that transposes ``lanes`` vectors of
    ``lanes`` lanes of ``element_bits``-bit elements."""
    register_bits = lanes * element_bits
    if lanes == 1:
        return TransposeNetwork(1, element_bits, (), (0,), (0,))
    if register_bits not in (128, 256, 512):
        raise ValueError(f"cannot transpose {lanes} lanes of {element_bits} bits")

    steps: list[tuple[ShuffleStep, int]] = []
    block = 1
    while block < lanes:
        low, high, bits = _stage_shuffles(block * element_bits, register_bits)
        for i in range(lanes):
            if (i // block) % 2 == 0:
                steps.append((ShuffleStep(i, i, i + block, low), bits))
                steps.append((ShuffleStep(i + block, i, i + block, high), bits))
        block *= 2

    network = TransposeNetwork(lanes, element_bits, tuple(steps), (), ())
    result = network.run([[(s, lane) for lane in range(lanes)] for s in range(lanes)])

    # Slot t of the result must hold one input lane, tau(t), of every input
    # slot, and lane i must come from the same input slot, sigma(i), in every
    # result slot: then feeding operand i into slot sigma(i) leaves operand
    # i's lane tau(t) in lane i of slot t, which is the natural transpose's
    # result tau(t).
    tau = [result[t][0][1] for t in range(lanes)]
    sigma = [result[0][i][0] for i in range(lanes)]
    for t in range(lanes):
        for i in range(lanes):
            if result[t][i] != (sigma[i], tau[t]):
                raise AssertionError(
                    f"{lanes}x{lanes} network is not a permuted transpose"
                )
    if sorted(sigma) != list(range(lanes)) or sorted(tau) != list(range(lanes)):
        raise AssertionError(f"{lanes}x{lanes} network drops lanes")

    source_of_slot = [0] * lanes
    for operand, slot in enumerate(sigma):
        source_of_slot[slot] = operand
    return TransposeNetwork(
        lanes,
        element_bits,
        tuple(steps),
        tuple(source_of_slot),
        tuple(tau),
    )


def deinterleave_shuffles(element_bits: int) -> tuple[Shuffle, Shuffle]:
    """The shuffles taking two vectors of interleaved complex numbers to the
    vector of their real parts and the vector of their imaginary parts."""
    if element_bits == 64:
        return Shuffle("unpack_low"), Shuffle("unpack_high")
    return Shuffle("shuffle_ps", 0x88), Shuffle("shuffle_ps", 0xDD)


def interleave_shuffles(element_bits: int) -> tuple[Shuffle, Shuffle]:
    """The shuffles taking a vector of real parts and one of imaginary parts to
    the two vectors of interleaved complex numbers they make up."""
    return Shuffle("unpack_low"), Shuffle("unpack_high")


def complex_lane_order(lanes: int, element_bits: int) -> tuple[int, ...]:
    """The order of complex numbers in the lanes of a deinterleaved load.

    Loading ``lanes`` consecutive complex numbers and deinterleaving them with
    ``deinterleave_shuffles`` puts number ``order[i]`` in lane ``i``. Storing
    through ``interleave_shuffles`` takes the same order back to consecutive
    numbers, so a value loaded this way can be stored without permuting it.
    """
    if lanes == 1:
        return (0,)
    memory = [(part, index) for index in range(lanes) for part in ("re", "im")]
    first, second = memory[:lanes], memory[lanes:]
    low, high = deinterleave_shuffles(element_bits)
    real = low.apply(first, second, element_bits)
    imaginary = high.apply(first, second, element_bits)
    if [part for part, _ in real] != ["re"] * lanes or [
        index for _, index in imaginary
    ] != [index for _, index in real]:
        raise AssertionError("deinterleave does not split real and imaginary parts")
    order = tuple(index for _, index in real)

    low, high = interleave_shuffles(element_bits)
    real_lanes = [("re", lane) for lane in range(lanes)]
    imaginary_lanes = [("im", lane) for lane in range(lanes)]
    stored = low.apply(real_lanes, imaginary_lanes, element_bits) + high.apply(
        real_lanes, imaginary_lanes, element_bits
    )
    stored_order = [0] * lanes
    for position in range(lanes):
        (re_part, re_lane), (im_part, im_lane) = stored[2 * position : 2 * position + 2]
        assert (re_part, im_part) == ("re", "im") and re_lane == im_lane
        stored_order[re_lane] = position
    if tuple(stored_order) != order:
        raise AssertionError("interleave does not invert deinterleave")
    return order
