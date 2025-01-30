from __future__ import annotations
from typing import Literal, Optional

import torch
from torch import Tensor

from .bit_ops import bit_shift
from .bit_packing import pack_bits, pack_scalar, unpack_bits, unpack_scalar
from ._types import Bits, QuantizedParameters, DoubleQuantizedParameters
from .utils import dynamic_split, get_bit_range, parse_precision, signed_max, min_max, safe_reciprocal


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


def double_quantize_tensor(
    x: Tensor,
    /,
    num_bits: tuple[Bits, Bits],
    block_size: tuple[int, int] = (16, 16),
    symmetric: tuple[bool, bool] = (True, True),
    # *,
    # dim: int = 1, # TODO add
) -> DoubleQuantizedParameters:
    # split into blocks, use higher precision for numerical stability
    x = x.unflatten(dim=1, sizes=(-1, block_size[1] * block_size[0])).float().contiguous()

    # first quantization
    q, scale, xmin = quantize_tensor(x, num_bits[0], block_size[0], symmetric[0], dim=2)

    # second quantize scale and xmin
    qsx_scale = quantize_tensor(scale.flatten(1, 2), num_bits[1], block_size[1], symmetric[1])
    if xmin is not None:
        qsx_xmin = quantize_tensor(xmin.flatten(1, 2), num_bits[1], block_size[1], symmetric[1])
    else:
        qsx_xmin = None

    qsx2 = DoubleQuantizedParameters(q, qsx_scale, qsx_xmin)

    return qsx2


def double_dequantize_tensor(qsx2: DoubleQuantizedParameters, /) -> Tensor:
    q, qsx_scale, qsx_xmin = qsx2
    scale = dequantize_tensor(*qsx_scale)
    xmin = dequantize_tensor(*qsx_xmin) if qsx_xmin is not None else None
    y = dequantize_tensor(q, scale, xmin)

    return y


def quantize(
    x: Tensor,
    /,
    num_bits: Bits,
    block_size: int = 32,
    symmetric: bool = True,
    group_last: bool = True,
    *,
    store_precision: Literal["fp16", "bf16", "fp8"] = "fp16",
    dim: int = 1,
) -> Tensor:
    q, scale, xmin = quantize_tensor(x, num_bits, block_size, symmetric, dim=dim)

    if symmetric:
        q = bit_shift(q, num_bits)
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
    if store_precision == "fp8":
        assert compute_precision is not None
    elif compute_precision is None:
        compute_precision = store_precision
    compute_dtype = parse_precision(compute_precision)

    if symmetric:
        q, scale = dynamic_split(Q, sizes=(-1, 2), dim=dim + 1)
        xmin = None
    else:
        q, scale, xmin = dynamic_split(Q, sizes=(-1, 2, 2), dim=dim + 1)

    q = unpack_bits(q, num_bits, symmetric, group_last, dim=dim + 1)
    scale = unpack_scalar(scale, store_precision, dim=dim + 1)
    xmin = unpack_scalar(xmin, store_precision, dim=dim + 1) if xmin is not None else None

    scale = scale.to(compute_dtype)
    xmin = xmin.to(compute_dtype) if xmin is not None else None

    y = dequantize_tensor(q, scale, xmin)
    y = y.flatten(dim, dim + 1)

    return y
