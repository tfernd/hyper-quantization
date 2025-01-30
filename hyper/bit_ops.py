from __future__ import annotations

import torch
from torch import Tensor

from ._types import Bits


# TODO better name
def bit_shift(q: Tensor, /, num_bits: Bits) -> Tensor:
    assert q.dtype == torch.int8

    offset = 1 << (num_bits - 1)
    q = q.view(torch.uint8) if num_bits == 8 else q.add(offset).to(torch.uint8)

    return q


# TODO better name
def bit_unshift(q: Tensor, /, num_bits: Bits) -> Tensor:
    offset = 1 << (num_bits - 1)
    q = q.view(torch.int8) if num_bits == 8 else q.to(torch.int8).sub(offset)

    return q
