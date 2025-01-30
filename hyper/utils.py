from __future__ import annotations
from typing import Iterable, Literal

import torch
from torch import Tensor

from ._types import Bits


def get_num_states(num_bits: Bits, /):
    return 1 << num_bits


def get_bit_mask(num_bits: Bits, /):
    return (1 << num_bits) - 1


def get_bit_range(num_bits: Bits, symmetric: bool, /):
    num_states = get_num_states(num_bits)

    bit_min = -num_states // 2 if symmetric else 0
    bit_max = num_states // 2 - 1 if symmetric else num_states - 1

    return bit_min, bit_max


def signed_max(x: Tensor, /, dim: int) -> Tensor:
    imax = x.abs().argmax(dim, keepdim=True)

    return x.take_along_dim(imax, dim)


def min_max(x: Tensor, /, dim: int) -> tuple[Tensor, Tensor]:
    xmin = x.amin(dim, keepdim=True)
    xmax = x.amax(dim, keepdim=True)

    return xmin, xmax


def safe_reciprocal(x: Tensor, /) -> Tensor:
    return torch.where(x != 0, 1 / x, torch.zeros_like(x))


def split_by_ratio(
    x: Tensor,
    ratios: tuple[int, ...],
    dim: int,
) -> Iterable[Tensor]:
    n = sum(ratios)
    N = x.size(dim)

    assert N % n == 0

    qsizes = tuple(s * N // n for s in ratios)

    return x.split(qsizes, dim)


def dynamic_split(
    x: Tensor,
    sizes: tuple[int, ...],
    dim: int,
) -> tuple[Tensor, ...]:
    total_size = x.size(dim)

    assert sizes.count(-1) in (0, 1)

    if sizes.count(-1) == 1:
        idx = sizes.index(-1)

        infer_sizes = list(sizes)
        infer_sizes[idx] = 0
        infer_sizes[idx] = total_size - sum(infer_sizes)
    else:
        infer_sizes = sizes
        assert sum(infer_sizes) == total_size

    return x.split(infer_sizes, dim)


def parse_precision(precision: Literal["fp32", "fp16", "bf16", "fp8"], /) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    elif precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    elif precision == "fp8":
        return torch.float8_e4m3fn

    raise ValueError(f"Invalid precision: {precision}")


def effective_num_bits(x: Tensor, Q: Tensor, /) -> float:
    assert Q.dtype == torch.uint8

    q_total_bits = Q.numel() * 8  # each byte in q is 8 bits
    effective_bits = q_total_bits / x.numel()

    return effective_bits


def block_normalized_loss(x: Tensor, y: Tensor, /, block_size: int, dim=1) -> float:
    assert x.shape == y.shape

    x = x.unflatten(dim, sizes=(-1, block_size))
    y = y.unflatten(dim, sizes=(-1, block_size))

    block_error = x.sub(y).abs().float().mean(dim) * safe_reciprocal(x.std(dim, unbiased=True))
    return block_error.mean().item()


def quant_name(
    num_bits: Bits,
    block_size: int,
    symmetric: bool,
    group_last: bool,
    store_precision: Literal["fp16", "bf16", "fp8"],
) -> str:
    s = "s" if symmetric else "a"
    g = "g" if group_last else ""
    p = "" if store_precision == "fp16" else f"_{store_precision}"

    return f"Q{num_bits}{s}{block_size}{g}{p}"


def double_quant_name(
    num_bits: tuple[Bits, Bits],
    block_size: tuple[int, int],
    symmetric: tuple[bool, bool],
    group_last: bool,
    store_precision: Literal["fp16", "bf16", "fp8"],
) -> str:
    s0 = "s" if symmetric[0] else "a"
    s1 = "s" if symmetric[1] else "a"
    g = "g" if group_last else ""
    p = "" if store_precision == "fp16" else f"_{store_precision}"

    return f"Q({num_bits[0]}{s0}{block_size[0]})xQ({num_bits[1]}{s1}{block_size[1]}){g}{p}"


def get_qsize(num_bits: Bits, block_size: int) -> int:
    return num_bits * block_size // 8
