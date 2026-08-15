"""
build123d topology

name: shape_batch.py
date: August 15th 2026

desc:
    Explicit accumulator for algebra-mode booleans. Sketch/Part ``+=`` stays
    eager; ShapeBatch queues operands and evaluates them as one kernel call.

license:

    Copyright 2026 Gumyr

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .shape_core import BooleanBatch, BooleanMode, Shape


class ShapeBatch:
    """Collect algebra operands and evaluate them as one boolean.

    ``Sketch.__iadd__`` / ``Part.__iadd__`` remain eager: each ``+=`` fuses
    immediately and returns a new object, so aliases do not see later
    additions. Use ``ShapeBatch`` when a loop should not pay that cost:

    .. code-block:: python

        holes = ShapeBatch()
        for loc in locations:
            holes += loc * rectangle
        plate = Circle(40) - holes  # tools passed directly to cut

    A ``ShapeBatch`` is not a ``Shape``. Geometry queries such as ``area``
    require ``fuse()`` or using the batch as an operand of ``+`` / ``-``.
    """

    def __init__(self, operands: Shape | Iterable[Shape] | None = None):
        self._operands: list[Shape] = []
        if operands is not None:
            self._extend(operands)

    @property
    def operands(self) -> tuple[Shape, ...]:
        """Queued top-level shapes, in insertion order."""
        return tuple(self._operands)

    @property
    def _dim(self) -> int | None:
        return self._operands[0]._dim if self._operands else None

    def __bool__(self) -> bool:
        return bool(self._operands)

    def __len__(self) -> int:
        return len(self._operands)

    def __iter__(self) -> Iterator[Shape]:
        return iter(self._operands)

    def __enter__(self) -> ShapeBatch:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def __iadd__(self, other: Shape | Iterable[Shape] | None) -> ShapeBatch:
        self._extend(other)
        return self

    def __add__(self, other: Shape | Iterable[Shape] | None) -> ShapeBatch:
        result = ShapeBatch(self._operands)
        result += other
        return result

    def __radd__(self, other: Shape | Iterable[Shape] | None):
        if isinstance(other, Shape):
            return NotImplemented
        result = ShapeBatch(other)
        result += self
        return result

    def __sub__(self, other: Shape | Iterable[Shape] | None):
        return self.fuse() - other

    def fuse(self) -> Shape:
        """Fuse queued operands with one kernel boolean and one clean."""
        if not self._operands:
            raise ValueError("Cannot fuse an empty ShapeBatch")
        if len(self._operands) == 1:
            return Shape.make_composite(self._operands, self._dim)
        fused = BooleanBatch(
            self._operands[0], BooleanMode.FUSE, self._operands[1:]
        ).execute()
        if isinstance(fused, Shape) and fused._dim == self._dim:
            return fused
        return Shape.make_composite([fused], self._dim)

    def _extend(self, other: Shape | Iterable[Shape] | None) -> None:
        items = _flatten_operands(other)
        if not items:
            return
        dim = self._dim
        for item in items:
            item_dim = item._dim
            if item_dim is None:
                raise ValueError("Dimensions of objects to add to are inconsistent")
            if dim is None:
                dim = item_dim
            elif item_dim != dim:
                raise ValueError("Only shapes with the same dimension can be added")
        self._operands.extend(items)


def _flatten_operands(other: Shape | Iterable[Shape] | None) -> list[Shape]:
    if other is None:
        return []
    if isinstance(other, ShapeBatch):
        return list(other._operands)
    if isinstance(other, Shape):
        return list(other.get_top_level_shapes())
    items: list[Shape] = []
    for obj in other:
        if obj is None:
            continue
        if isinstance(obj, ShapeBatch):
            items.extend(obj._operands)
        elif isinstance(obj, Shape):
            items.extend(obj.get_top_level_shapes())
        else:
            raise TypeError(f"ShapeBatch does not accept {type(obj).__name__}")
    return items
