"""Base-case DFT kernels over complex values in registers.

Every kernel maps ``n`` complex points to their DFT with sign ``sign``, lane by
lane, using a lazy complex algebra: ``Emitter`` for real and imaginary parts
in separate registers, ``InterleavedAlgebra`` for interleaved ones. In both,
multiplications by ``±i`` and ``±1`` are free, those by ``(±1 ± i) / sqrt(2)``
fold their scale into the next fused multiply-add, and products by other
constants become fused multiply-adds directly.
"""

import math
from collections.abc import Sequence
from typing import Protocol, TypeVar

C = TypeVar("C")

BASE_RADICES = (2, 3, 4, 5, 7, 8, 16)


class ComplexAlgebra(Protocol[C]):
    """Lazy complex arithmetic on values of type ``C``."""

    def cadd(self, a: C, b: C, /) -> C: ...
    def csub(self, a: C, b: C, /) -> C: ...
    def cscale(self, a: C, factor: float, /) -> C: ...
    def csum(self, values: Sequence[C], /) -> C: ...
    def times_i(self, a: C, sign: int, /) -> C: ...
    def times_root(self, a: C, exponent: int, n: int, /) -> C: ...


def _radix2(e: ComplexAlgebra[C], x: Sequence[C]) -> list[C]:
    return [e.cadd(x[0], x[1]), e.csub(x[0], x[1])]


def _radix4(e: ComplexAlgebra[C], x: Sequence[C], sign: int) -> list[C]:
    a = e.cadd(x[0], x[2])
    b = e.csub(x[0], x[2])
    c = e.cadd(x[1], x[3])
    d = e.times_i(e.csub(x[1], x[3]), sign)
    return [e.cadd(a, c), e.cadd(b, d), e.csub(a, c), e.csub(b, d)]


def _split(e: ComplexAlgebra[C], x: Sequence[C], sign: int, r: int, m: int) -> list[C]:
    """Decimation in time, ``n = r * m``: ``r`` DFTs of ``m`` points over the
    residues of the index modulo ``r``, twiddles, then ``m`` DFTs of ``r``."""
    n = r * m
    inner = [dft(e, [x[j2 + r * j1] for j1 in range(m)], sign) for j2 in range(r)]
    result: list[C] = [x[0]] * n
    for k1 in range(m):
        column = [e.times_root(inner[j2][k1], sign * j2 * k1, n) for j2 in range(r)]
        for k2, value in enumerate(dft(e, column, sign)):
            result[k1 + m * k2] = value
    return result


def _odd_prime(e: ComplexAlgebra[C], x: Sequence[C], sign: int) -> list[C]:
    """The symmetric form of a DFT of odd length, pairing ``j`` with ``n - j``."""
    n = len(x)
    half = (n - 1) // 2
    sums = [e.cadd(x[j], x[n - j]) for j in range(1, half + 1)]
    differences = [e.csub(x[j], x[n - j]) for j in range(1, half + 1)]

    result: list[C] = [x[0]] * n
    result[0] = e.csum([x[0], *sums])
    for k in range(1, half + 1):
        cosines = [math.cos(2 * math.pi * j * k / n) for j in range(1, half + 1)]
        sines = [math.sin(2 * math.pi * j * k / n) for j in range(1, half + 1)]
        # a = x0 + sum cos * s, b = sum sin * d; X[k] = a + sign*i*b.
        a = e.csum([x[0], *(e.cscale(s, c) for s, c in zip(sums, cosines))])
        b = e.csum([e.cscale(d, s) for d, s in zip(differences, sines)])
        rotated = e.times_i(b, sign)
        result[k] = e.cadd(a, rotated)
        result[n - k] = e.csub(a, rotated)
    return result


def dft(e: ComplexAlgebra[C], x: Sequence[C], sign: int) -> list[C]:
    """The DFT of ``x`` with lazily signed and scaled results."""
    n = len(x)
    match n:
        case 1:
            return list(x)
        case 2:
            return _radix2(e, x)
        case 4:
            return _radix4(e, x, sign)
        case 8:
            return _split(e, x, sign, 2, 4)
        case 16:
            return _split(e, x, sign, 4, 4)
        case 3 | 5 | 7:
            return _odd_prime(e, x, sign)
        case _:
            raise ValueError(f"no base-case kernel for {n} points")
