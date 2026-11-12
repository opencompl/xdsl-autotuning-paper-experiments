"""Base-case DFT kernels over complex values in registers.

Every kernel maps ``n`` complex points to their DFT with sign ``sign``, lane by
lane, using the lazy algebra of ``Emitter``: multiplications by ``±i`` and
``±1`` are free, those by ``(±1 ± i) / sqrt(2)`` cost one addition per
component and fold their scale into the next fused multiply-add, and products
by other constants become fused multiply-adds directly.
"""

import math
from collections.abc import Sequence

from autotuner.fft.values import Complex, Emitter

BASE_RADICES = (2, 3, 4, 5, 7, 8)


def _radix2(e: Emitter, x: Sequence[Complex]) -> list[Complex]:
    return [e.cadd(x[0], x[1]), e.csub(x[0], x[1])]


def _radix4(e: Emitter, x: Sequence[Complex], sign: int) -> list[Complex]:
    a = e.cadd(x[0], x[2])
    b = e.csub(x[0], x[2])
    c = e.cadd(x[1], x[3])
    d = e.times_i(e.csub(x[1], x[3]), sign)
    return [e.cadd(a, c), e.cadd(b, d), e.csub(a, c), e.csub(b, d)]


def _split(
    e: Emitter, x: Sequence[Complex], sign: int, r: int, m: int
) -> list[Complex]:
    """Decimation in time, ``n = r * m``: ``r`` DFTs of ``m`` points over the
    residues of the index modulo ``r``, twiddles, then ``m`` DFTs of ``r``."""
    n = r * m
    inner = [dft(e, [x[j2 + r * j1] for j1 in range(m)], sign) for j2 in range(r)]
    result: list[Complex] = [x[0]] * n
    for k1 in range(m):
        column = [e.times_root(inner[j2][k1], sign * j2 * k1, n) for j2 in range(r)]
        for k2, value in enumerate(dft(e, column, sign)):
            result[k1 + m * k2] = value
    return result


def _odd_prime(e: Emitter, x: Sequence[Complex], sign: int) -> list[Complex]:
    """The symmetric form of a DFT of odd length, pairing ``j`` with ``n - j``."""
    n = len(x)
    half = (n - 1) // 2
    sums = [e.cadd(x[j], x[n - j]) for j in range(1, half + 1)]
    differences = [e.csub(x[j], x[n - j]) for j in range(1, half + 1)]

    result: list[Complex] = [x[0]] * n
    result[0] = (
        e.sum([x[0][0], *(s[0] for s in sums)]),
        e.sum([x[0][1], *(s[1] for s in sums)]),
    )
    for k in range(1, half + 1):
        cosines = [math.cos(2 * math.pi * j * k / n) for j in range(1, half + 1)]
        sines = [math.sin(2 * math.pi * j * k / n) for j in range(1, half + 1)]
        # a = x0 + sum cos * s, b = sum sin * d; X[k] = a + sign*i*b.
        a = tuple(
            e.sum([x[0][part], *(s[part].scaled(c) for s, c in zip(sums, cosines))])
            for part in (0, 1)
        )
        b = tuple(
            e.sum([d[part].scaled(s) for d, s in zip(differences, sines)])
            for part in (0, 1)
        )
        rotated = e.times_i((b[0], b[1]), sign)
        result[k] = e.cadd((a[0], a[1]), rotated)
        result[n - k] = e.csub((a[0], a[1]), rotated)
    return result


def dft(e: Emitter, x: Sequence[Complex], sign: int) -> list[Complex]:
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
        case 3 | 5 | 7:
            return _odd_prime(e, x, sign)
        case _:
            raise ValueError(f"no base-case kernel for {n} points")
