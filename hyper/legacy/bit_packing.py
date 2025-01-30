from __future__ import annotations

import torch
from torch import Tensor


def pack_half(x: Tensor, /) -> Tensor:
    return x.half().view(torch.uint8)


def unpack_half(x: Tensor, /) -> Tensor:
    return x.view(torch.half)


def pack4bits(q: Tensor, /) -> Tensor:
    ql, qh = (q & 0b1111).unflatten(dim=2, sizes=(2, -1)).unbind(dim=2)
    q = ql | (qh << 4)

    return q


def unpack4bits(q: Tensor, /) -> Tensor:
    idx = torch.arange(0, 8, 4, device=q.device, dtype=torch.uint8)
    q = q.unsqueeze(dim=2) >> idx.view(-1, 1)
    q = (q & 0b1111).flatten(2, 3)

    return q


def pack2bits(q: Tensor, /) -> Tensor:
    q0, q1, q2, q3 = (q & 0b11).unflatten(dim=2, sizes=(4, -1)).unbind(dim=2)
    q = (q0 | (q1 << 2)) | ((q2 << 4) | (q3 << 6))

    return q


def unpack2bits(q: Tensor, /) -> Tensor:
    idx = torch.arange(0, 8, 2, device=q.device, dtype=torch.uint8)
    q = q.unsqueeze(dim=2) >> idx.view(-1, 1)
    q = (q & 0b11).flatten(2, 3)

    return q


def pack1bit(q: Tensor, /) -> Tensor:
    values = 2 ** torch.arange(0, 8, device=q.device, dtype=torch.uint8)
    q = (q & 0b1).unflatten(dim=2, sizes=(-1, 8))
    q = q.mul(values).sum(dim=3, dtype=torch.uint8)  # qh @ values

    return q


def unpack1bit(q: Tensor, /) -> Tensor:
    idx = torch.arange(0, 8, 1, device=q.device, dtype=torch.uint8)
    q = q.unsqueeze(dim=3) >> idx
    q = (q & 0b1).flatten(2, 3)

    return q
