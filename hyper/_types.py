from __future__ import annotations
from typing import Literal, NamedTuple, Optional

from torch import Tensor

Bits = Literal[1, 2, 3, 4, 5, 6, 7, 8]


class QuantizedParameters(NamedTuple):
    q: Tensor
    scale: Tensor
    xmin: Optional[Tensor]


class DoubleQuantizedParameters(NamedTuple):
    q: Tensor
    qsx_scale: QuantizedParameters
    qsx_xmin: Optional[QuantizedParameters]
