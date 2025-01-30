from __future__ import annotations

import torch
from torch import Tensor

from ..bit_ops import bit_shift, bit_unshift
from .bit_packing import pack1bit, unpack1bit, pack4bits, unpack4bits, pack_half, unpack_half


############## 8-bit block (un)packing ##############


def pack_q8_0(q: Tensor, scale: Tensor, /, block_size: int = 32) -> Tensor:
    assert q.ndim == 3
    assert scale.ndim == 3
    assert q.size(2) == block_size
    assert scale.size(2) == 1
    assert q.dtype == torch.int8

    q = bit_shift(q, num_bits=8)
    scale = pack_half(scale)

    Q = torch.cat([scale, q], dim=2)

    return Q.flatten(1, 2)


def unpack_q8_0(Q: Tensor, /, block_size: int = 32) -> tuple[Tensor, Tensor]:
    assert Q.ndim == 2
    assert Q.dtype == torch.uint8

    Q = Q.unflatten(dim=1, sizes=(-1, 2 + block_size))
    scale, q = Q.split((2, block_size), dim=2)

    q = bit_unshift(q, num_bits=8)
    scale = unpack_half(scale)

    return q, scale


############## 4-bit block (un)packing ##############


def pack_q4_0(q: Tensor, scale: Tensor, /, block_size: int = 32) -> Tensor:
    assert q.ndim == 3
    assert scale.ndim == 3
    assert q.size(2) == block_size
    assert scale.size(2) == 1
    assert q.dtype == torch.int8

    q = bit_shift(q, num_bits=4)
    q = pack4bits(q)
    scale = pack_half(scale)

    Q = torch.cat([scale, q], dim=2)

    return Q.flatten(1, 2)


def unpack_q4_0(Q: Tensor, /, block_size: int = 32) -> tuple[Tensor, Tensor]:
    assert Q.ndim == 2
    assert Q.dtype == torch.uint8

    qsize = (2, block_size // 2)
    Q = Q.unflatten(dim=1, sizes=(-1, sum(qsize)))
    scale, q = Q.split(qsize, dim=2)

    q = unpack4bits(q)
    q = bit_unshift(q, num_bits=4)
    scale = unpack_half(scale)

    return q, scale


def pack_q4_1(q: Tensor, scale: Tensor, xmin: Tensor, /, block_size: int = 32) -> Tensor:
    assert q.ndim == 3
    assert scale.ndim == 3
    assert xmin.ndim == 3
    assert q.size(2) == block_size
    assert scale.size(2) == 1
    assert xmin.size(2) == 1
    assert q.dtype == torch.uint8

    q = pack4bits(q)
    scale = pack_half(scale)
    xmin = pack_half(xmin)

    Q = torch.cat([scale, xmin, q], dim=2)

    return Q.flatten(1, 2)


def unpack_q4_1(Q: Tensor, /, block_size: int = 32) -> tuple[Tensor, Tensor, Tensor]:
    assert Q.ndim == 2
    assert Q.dtype == torch.uint8

    qsize = (2, 2, block_size // 2)
    Q = Q.unflatten(dim=1, sizes=(-1, sum(qsize)))
    scale, xmin, q = Q.split(qsize, dim=2)

    q = unpack4bits(q)
    scale = unpack_half(scale)
    xmin = unpack_half(xmin)

    return q, scale, xmin


############## 5-bit block (un)packing ##############


def pack_q5_0(q: Tensor, scale: Tensor, /, block_size: int = 32) -> Tensor:
    assert q.ndim == 3
    assert scale.ndim == 3
    assert q.size(2) == block_size
    assert scale.size(2) == 1
    assert q.dtype == torch.int8

    q = bit_shift(q, num_bits=5)
    qh = pack1bit(q >> 4)
    ql = pack4bits(q)
    scale = pack_half(scale)

    Q = torch.cat([scale, qh, ql], dim=2)

    return Q.flatten(1, 2)


def unpack_q5_0(Q: Tensor, /, block_size: int = 32) -> tuple[Tensor, Tensor]:
    assert Q.ndim == 2
    assert Q.dtype == torch.uint8

    # ?
    qsize = (2, block_size // 8, block_size // 2)
    Q = Q.unflatten(dim=1, sizes=(-1, sum(qsize)))
    scale, qh, ql = Q.split(qsize, dim=2)

    qh = unpack1bit(qh)
    ql = unpack4bits(ql)
    q = ql | (qh << 4)
    q = bit_unshift(q, num_bits=5)
    scale = unpack_half(scale)

    return q, scale


def pack_q5_1(q: Tensor, scale: Tensor, xmin: Tensor, /, block_size: int = 32) -> Tensor:
    assert q.ndim == 3
    assert scale.ndim == 3
    assert xmin.ndim == 3
    assert q.size(2) == block_size
    assert scale.size(2) == 1
    assert xmin.size(2) == 1
    assert q.dtype == torch.uint8

    qh = pack1bit(q >> 4)
    ql = pack4bits(q)
    scale = pack_half(scale)
    xmin = pack_half(xmin)

    Q = torch.cat([scale, xmin, qh, ql], dim=2)

    return Q.flatten(1, 2)


def unpack_q5_1(Q: Tensor, /, block_size: int = 32) -> tuple[Tensor, Tensor, Tensor]:
    assert Q.ndim == 2
    assert Q.dtype == torch.uint8

    # ?
    qsize = (2, 2, block_size // 8, block_size // 2)
    Q = Q.unflatten(dim=1, sizes=(-1, sum(qsize)))
    scale, xmin, qh, ql = Q.split(qsize, dim=2)

    qh = unpack1bit(qh)
    ql = unpack4bits(ql)
    q = ql | (qh << 4)
    scale = unpack_half(scale)
    xmin = unpack_half(xmin)

    return q, scale, xmin
