from __future__ import annotations
from typing import Literal

import torch
from torch import Tensor

from ._types import Bits
from .bit_ops import bit_shift, bit_unshift
from .utils import parse_precision, split_by_ratio


def pack_bits(
    q: Tensor,
    /,
    num_bits: Bits,
    group_last: bool = True,
    *,
    dim: int = 2,
) -> Tensor:
    assert q.dtype in (torch.int8, torch.uint8)

    symmetric = q.dtype == torch.int8
    if symmetric:
        q = bit_shift(q, num_bits)

    mask = (1 << num_bits) - 1
    q = q & mask  # mask out the extra bits

    # AAAA_AAAA
    if num_bits == 8:
        pass

    # 4bits: AAAA_BBBB | 2bits: AABB_CCDD | 1bits: ABCD_EFGH
    elif num_bits in (4, 2, 1):
        groups = 8 // num_bits
        sizes = (-1, groups) if group_last else (groups, -1)
        unbind_dim = dim + 1 if group_last else dim

        qs = q.unflatten(dim, sizes).unbind(unbind_dim)

        if num_bits == 4:
            q = (qs[0] << 0) | (qs[1] << 4)
        elif num_bits == 2:
            q = ((qs[0] << 0) | (qs[1] << 2)) | (qs[2] << 4) | (qs[3] << 6)
        elif num_bits == 1:
            qa = ((qs[0] << 0) | (qs[1] << 1)) | ((qs[2] << 2) | (qs[3] << 3))
            qb = ((qs[4] << 4) | (qs[5] << 5)) | ((qs[6] << 6) | (qs[7] << 7))
            q = qa | qb

    # AAAA_BBBB CCCC_DDDD EEEE_FFFF GGGG_HHHH + abcd_efgh
    elif num_bits == 5:
        q1 = pack_bits(q >> 4, num_bits=1, dim=dim, group_last=group_last)
        q4 = pack_bits(q >> 0, num_bits=4, dim=dim, group_last=group_last)

        q = torch.cat([q1, q4], dim)

    # AAAA_BBBB CCCC_DDDD + aabb_ccdd
    elif num_bits == 6:
        q2 = pack_bits(q >> 4, num_bits=2, dim=dim, group_last=group_last)
        q4 = pack_bits(q >> 0, num_bits=4, dim=dim, group_last=group_last)

        q = torch.cat([q2, q4], dim)

    # AABB_CCDD EEFF_GGHH + abcd_efgh
    elif num_bits == 3:
        q1 = pack_bits(q >> 2, num_bits=1, dim=dim, group_last=group_last)
        q2 = pack_bits(q >> 0, num_bits=2, dim=dim, group_last=group_last)

        q = torch.cat([q1, q2], dim)

    # AAAA_BBBB + CCCC_DDDD EEEE_FFFF GGGG_HHHH + aabb_ccdd + eeff_gghh + abcd_efgh
    elif num_bits == 7:
        q1 = pack_bits(q >> 6, num_bits=1, dim=dim, group_last=group_last)
        q2 = pack_bits(q >> 4, num_bits=2, dim=dim, group_last=group_last)
        q4 = pack_bits(q >> 0, num_bits=4, dim=dim, group_last=group_last)

        q = torch.cat([q1, q2, q4], dim)

    return q


def unpack_bits(
    q: Tensor,
    /,
    num_bits: Bits,
    symmetric: bool,
    group_last: bool = True,
    *,
    dim: int = 2,
) -> Tensor:
    assert q.dtype == torch.uint8

    extra_dims = [1] * (q.ndim - dim - 1)
    mask = (1 << num_bits) - 1

    # AAAA_AAAA
    if num_bits == 8:
        pass

    # 4bits: AAAA_BBBB | 2bits: AABB_CCDD | 1bits: ABCD_EFGH
    elif num_bits in (4, 2, 1):
        idx = torch.arange(0, 8, num_bits, device=q.device, dtype=torch.uint8)

        q = q.unflatten(dim, (-1, 1) if group_last else (1, -1))
        idx = idx.view(1, -1, *extra_dims) if group_last else idx.view(-1, 1, *extra_dims)

        q = ((q >> idx) & mask).flatten(dim, dim + 1)

    # AAAA_BBBB CCCC_DDDD EEEE_FFFF GGGG_HHHH + abcd_efgh
    elif num_bits == 5:
        q1, q4 = split_by_ratio(q, ratios=(4, 1), dim=dim)

        q1 = unpack_bits(q1, num_bits=1, symmetric=False, dim=dim, group_last=group_last)
        q4 = unpack_bits(q4, num_bits=4, symmetric=False, dim=dim, group_last=group_last)

        q = q4 | (q1 << 4)

    # AAAA_BBBB CCCC_DDDD + aabb_ccdd
    elif num_bits == 6:
        q4, q2 = split_by_ratio(q, ratios=(4, 2), dim=dim)

        q4 = unpack_bits(q4, num_bits=4, symmetric=False, dim=dim, group_last=group_last)
        q2 = unpack_bits(q2, num_bits=2, symmetric=False, dim=dim, group_last=group_last)

        q = q4 | (q2 << 4)

    # AABB_CCDD EEFF_GGHH + abcd_efgh
    elif num_bits == 3:
        q1, q2 = split_by_ratio(q, ratios=(1, 2), dim=dim)

        q2 = unpack_bits(q2, num_bits=2, symmetric=False, dim=dim, group_last=group_last)
        q1 = unpack_bits(q1, num_bits=1, symmetric=False, dim=dim, group_last=group_last)

        q = q2 | (q1 << 2)

    # AAAA_BBBB + CCCC_DDDD EEEE_FFFF GGGG_HHHH + aabb_ccdd + eeff_gghh + abcd_efgh
    elif num_bits == 7:
        q4, q2, q1 = split_by_ratio(q, ratios=(4, 2, 1), dim=dim)

        q4 = unpack_bits(q4, num_bits=4, symmetric=False, dim=dim, group_last=group_last)
        q2 = unpack_bits(q2, num_bits=2, symmetric=False, dim=dim, group_last=group_last)
        q1 = unpack_bits(q1, num_bits=1, symmetric=False, dim=dim, group_last=group_last)

        q = q4 | (q2 << 4) | (q1 << 6)

    # to int8 for easier packing and unpacking
    if symmetric:
        q = bit_unshift(q, num_bits)

    return q


def pack_scalar(
    x: Tensor,
    /,
    precision: Literal["fp16", "bf16", "fp8"],
    dim: int = 2,
) -> Tensor:
    dtype = parse_precision(precision)

    x = x.swapaxes(dim, -1)
    x = x.to(dtype).contiguous().view(torch.uint8)
    x = x.swapaxes(dim, -1)

    return x


def unpack_scalar(
    x: Tensor,
    precision: Literal["fp16", "bf16", "fp8"],
    dim: int = 2,
) -> Tensor:
    dtype = parse_precision(precision)

    x = x.swapaxes(dim, -1)
    x = x.view(dtype)
    x = x.swapaxes(dim, -1)

    return x
