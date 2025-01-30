from __future__ import annotations
from typing import Literal, Optional

import torch
from torch import Tensor

from .bit_packing import pack_bits, pack_scalar, unpack_bits, unpack_scalar
from ._types import Bits, QuantizedParameters
from .utils import dynamic_split, get_bit_range, get_qsize, parse_precision, signed_max, min_max, safe_reciprocal


def get_scale_and_xmin(
    x: Tensor,
    /,
    num_bits: Bits,
    symmetric: bool,
    dim: int,
) -> tuple[Tensor, Optional[Tensor]]:
    bit_min, bit_max = get_bit_range(num_bits, symmetric)

    if symmetric:
        xsmax = signed_max(x, dim)
        scale = xsmax / bit_min
        xmin = None
    else:
        xmin, xmax = min_max(x, dim)
        scale = (xmax - xmin) / bit_max

    return scale, xmin


def quantize_q(
    x: Tensor,
    scale: Tensor,
    xmin: Optional[Tensor],
    /,
    num_bits: Bits,
) -> Tensor:
    symmetric = xmin is None
    bit_min, bit_max = get_bit_range(num_bits, symmetric)

    q = (x - xmin if xmin is not None else x) * safe_reciprocal(scale)
    q = q.round().clamp(bit_min, bit_max)
    q = q.to(torch.int8 if symmetric else torch.uint8)

    return q


def quantize_tensor(
    x: Tensor,
    /,
    num_bits: Bits,
    block_size: int = 32,
    symmetric: bool = True,
    *,
    dim: int = 1,
) -> QuantizedParameters:
    # split into blocks, use higher precision for numerical stability
    x = x.unflatten(dim, sizes=(-1, block_size)).float().contiguous()
    scale, xmin = get_scale_and_xmin(x, num_bits, symmetric, dim=dim + 1)

    q = quantize_q(x, scale, xmin, num_bits)

    # TODO add optimization to minimize loss, optional

    return QuantizedParameters(q, scale, xmin)


def dequantize_tensor(
    q: Tensor,
    scale: Tensor,
    xmin: Optional[Tensor] = None,
    /,
) -> Tensor:
    return q * scale if xmin is None else q * scale + xmin


def quantize(
    x: Tensor,
    /,
    num_bits: Bits,
    block_size: int = 32,
    symmetric: bool = True,
    group_last: bool = True,
    store_precision: Literal["fp16", "bf16", "fp8"] = "fp16",
    *,
    dim: int = 1,
) -> Tensor:
    q, scale, xmin = quantize_tensor(x, num_bits, block_size, symmetric, dim=dim)

    q = pack_bits(q, num_bits, group_last, dim=dim + 1)
    scale = pack_scalar(scale, store_precision, dim=dim + 1)
    xmin = pack_scalar(xmin, store_precision, dim=dim + 1) if xmin is not None else None

    Q = torch.cat([q, scale] if xmin is None else [q, scale, xmin], dim=dim + 1)

    return Q


def dequantize(
    Q: Tensor,
    /,
    num_bits: Bits,
    symmetric: bool = True,
    group_last: bool = True,
    *,
    store_precision: Literal["fp16", "bf16", "fp8"] = "fp16",
    compute_precision: Optional[Literal["fp32", "fp16", "bf16"]] = None,
    dim: int = 1,
) -> Tensor:
    # TODO DRY
    if store_precision == "fp8":
        assert compute_precision is not None
    elif compute_precision is None:
        compute_precision = store_precision
    elif compute_precision != "fp32":
        compute_precision = store_precision
    compute_dtype = parse_precision(compute_precision)

    scaler_num_bytes = 1 if store_precision == "fp8" else 2
    if symmetric:
        q, scale = dynamic_split(Q, sizes=(-1, scaler_num_bytes), dim=dim + 1)
        xmin = None
    else:
        q, scale, xmin = dynamic_split(Q, sizes=(-1, scaler_num_bytes, scaler_num_bytes), dim=dim + 1)

    q = unpack_bits(q, num_bits, symmetric, group_last, dim=dim + 1)
    scale = unpack_scalar(scale, store_precision, dim=dim + 1)
    xmin = unpack_scalar(xmin, store_precision, dim=dim + 1) if xmin is not None else None

    scale = scale.to(compute_dtype)
    xmin = xmin.to(compute_dtype) if xmin is not None else None

    y = dequantize_tensor(q, scale, xmin)
    y = y.flatten(dim, dim + 1)

    return y


def double_quantize(
    x: Tensor,
    /,
    num_bits: tuple[Bits, Bits],
    block_size: tuple[int, int] = (16, 16),
    symmetric: tuple[bool, bool] = (True, True),
    group_last: bool = True,
    store_precision: Literal["fp16", "bf16", "fp8"] = "fp16",
    *,
    dim: int = 1,
) -> Tensor:
    x = x.unflatten(dim=1, sizes=(-1, block_size[1] * block_size[0]))

    q, scale, xmin = quantize_tensor(x, num_bits[0], block_size[0], symmetric[0], dim=dim + 1)

    q = pack_bits(q, num_bits[0], group_last, dim=dim + 2)
    q = q.flatten(dim + 1)

    scale = quantize(scale.flatten(dim), num_bits[1], block_size[1], symmetric[1], group_last, store_precision, dim=dim)
    if xmin is not None:
        xmin = quantize(xmin.flatten(dim), num_bits[1], block_size[1], symmetric[1], group_last, store_precision, dim=dim)

    Q = torch.cat([q, scale] if xmin is None else [q, scale, xmin], dim=dim + 1)

    return Q


def double_dequantize(
    Q: Tensor,
    /,
    num_bits: tuple[Bits, Bits],
    block_size: tuple[int, int] = (16, 16),
    symmetric: tuple[bool, bool] = (True, True),
    group_last: bool = True,
    store_precision: Literal["fp16", "bf16", "fp8"] = "fp16",
    compute_precision: Optional[Literal["fp32", "fp16", "bf16"]] = None,
    *,
    dim: int = 1,
) -> Tensor:
    qsize = get_qsize(num_bits[0], block_size[0]) * block_size[1]
    if symmetric[0]:
        q, scale = dynamic_split(Q, sizes=(qsize, -1), dim=dim + 1)
        xmin = None
    else:
        sx_size = (Q.size(dim + 1) - qsize) // 2
        q, scale, xmin = dynamic_split(Q, sizes=(qsize, sx_size, sx_size), dim=dim + 1)
    q = q.unflatten(dim + 1, sizes=(block_size[1], -1))
    q = unpack_bits(q, num_bits[0], symmetric[0], group_last, dim=dim + 2)

    scale = dequantize(scale, num_bits[1], symmetric[1], group_last, store_precision=store_precision, compute_precision=compute_precision, dim=dim)
    scale = scale.unflatten(dim, sizes=(-1, block_size[1], 1))
    if xmin is not None:
        xmin = dequantize(xmin, num_bits[1], symmetric[1], group_last, store_precision=store_precision, compute_precision=compute_precision, dim=dim)
        xmin = xmin.unflatten(dim, sizes=(-1, block_size[1], 1))

    y = dequantize_tensor(q, scale, xmin)
    y = y.flatten(dim, dim + 2)

    return y
