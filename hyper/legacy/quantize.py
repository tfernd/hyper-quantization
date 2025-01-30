from __future__ import annotations

from torch import Tensor

from ..quantize import quantize_tensor, dequantize_tensor
from .packing import pack_q8_0, unpack_q8_0
from .packing import pack_q5_0, unpack_q5_0, pack_q5_1, unpack_q5_1
from .packing import pack_q4_0, unpack_q4_0, pack_q4_1, unpack_q4_1


def quantize_q8_0(x: Tensor, /) -> Tensor:
    q, scale, xmin = quantize_tensor(x, num_bits=8)
    assert xmin is None
    Q = pack_q8_0(q, scale)

    return Q


def dequatize_q8_0(Q: Tensor, /) -> Tensor:
    q, scale = unpack_q8_0(Q)
    y = dequantize_tensor(q, scale).flatten(1, 2)

    return y


def quantize_q4_0(x: Tensor, /) -> Tensor:
    q, scale, xmin = quantize_tensor(x, num_bits=4)
    assert xmin is None
    Q = pack_q4_0(q, scale)

    return Q


def dequatize_q4_0(Q: Tensor, /) -> Tensor:
    q, scale = unpack_q4_0(Q)
    y = dequantize_tensor(q, scale).flatten(1, 2)

    return y


def quantize_q4_1(x: Tensor, /) -> Tensor:
    q, scale, xmin = quantize_tensor(x, num_bits=4, symmetric=False)
    assert xmin is not None
    Q = pack_q4_1(q, scale, xmin)

    return Q


def dequatize_q4_1(Q: Tensor, /) -> Tensor:
    q, scale, xmin = unpack_q4_1(Q)
    y = dequantize_tensor(q, scale, xmin).flatten(1, 2)

    return y


def quantize_q5_0(x: Tensor, /) -> Tensor:
    q, scale, xmin = quantize_tensor(x, num_bits=5)
    assert xmin is None
    Q = pack_q5_0(q, scale)

    return Q


def dequatize_q5_0(Q: Tensor, /) -> Tensor:
    q, scale = unpack_q5_0(Q)
    y = dequantize_tensor(q, scale).flatten(1, 2)

    return y


def quantize_q5_1(x: Tensor, /) -> Tensor:
    q, scale, xmin = quantize_tensor(x, num_bits=5, symmetric=False)
    assert xmin is not None
    Q = pack_q5_1(q, scale, xmin)

    return Q


def dequatize_q5_1(Q: Tensor, /) -> Tensor:
    q, scale, xmin = unpack_q5_1(Q)
    y = dequantize_tensor(q, scale, xmin).flatten(1, 2)

    return y
