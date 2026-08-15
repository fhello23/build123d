"""Shared CAD workloads for performance benchmarks and semantic tests."""

from __future__ import annotations

from collections.abc import Sequence

from build123d import (
    Axis,
    Box,
    BuildPart,
    BuildSketch,
    Circle,
    Cylinder,
    GridLocations,
    Locations,
    Mode,
    Rectangle,
    Select,
    Sketch,
    Sphere,
    fillet,
)

HOLE_DIAMETER = 80.0
HOLE_RECTANGLE = (2.0, 2.0)
HOLE_GRID = (4.0, 4.0, 20, 20)
EXPECTED_HOLE_COUNT = 284
FACE_GRID = (2.0, 2.0, 40, 50)
EXPECTED_FACE_GRID_COUNT = 2000


def hole_locations():
    """Locations of the 284-rectangle hole pattern from algebra_performance.rst."""
    radius_sq = (HOLE_DIAMETER / 2 - 1.8) ** 2
    return [
        loc
        for loc in GridLocations(*HOLE_GRID)
        if loc.position.X**2 + loc.position.Y**2 < radius_sq
    ]


def hole_rectangles():
    """The 284 rectangles used by sequential and batched fuse/cut workloads."""
    width, height = HOLE_RECTANGLE
    rectangle = Rectangle(width, height)
    return [loc * rectangle for loc in hole_locations()]


def sequential_fuse(shapes: Sequence):
    result = shapes[0]
    for shape in shapes[1:]:
        result = result.fuse(shape)
    return result


def batched_fuse(shapes: Sequence):
    return shapes[0].fuse(*shapes[1:])


def sequential_algebra_fuse(shapes: Sequence):
    holes = Sketch()
    for shape in shapes:
        holes += shape
    return holes


def batched_algebra_fuse(shapes: Sequence):
    return Sketch() + list(shapes)


def plate_with_holes_base():
    return Circle(HOLE_DIAMETER / 2)


def sequential_cut(base, tools: Sequence):
    result = base
    for tool in tools:
        result = result.cut(tool)
    return result


def batched_cut(base, tools: Sequence):
    return base.cut(*tools)


def face_grid_rectangles():
    """A 40x50 grid of disjoint unit rectangles (2000 faces)."""
    rectangle = Rectangle(1, 1)
    return GridLocations(*FACE_GRID) * rectangle


def face_grid_sketch():
    return Sketch() + face_grid_rectangles()


def selector_heavy_builder():
    """Builder workload that reads LAST/NEW and filters after every publish."""
    with BuildPart() as part:
        Box(40, 40, 10)
        for loc in GridLocations(8, 8, 5, 5):
            with Locations(loc):
                Cylinder(1.5, 20)
            _ = part.faces(Select.LAST)
            _ = part.edges(Select.NEW)
            _ = part.faces().filter_by(Axis.Z)
            _ = part.edges().group_by(Axis.Z)
        fillet(part.edges().group_by(Axis.Z)[-1], 0.4)
        _ = part.edges(Select.LAST)
        _ = part.edges(Select.NEW)
    return part.part


def mixed_mode_builder():
    """Interleaved ADD, SUBTRACT, INTERSECT, and REPLACE operations."""
    with BuildPart() as part:
        Box(20, 20, 20)
        Cylinder(6, 30, mode=Mode.SUBTRACT)
        Box(8, 8, 30, mode=Mode.ADD)
        Sphere(7, mode=Mode.INTERSECT)
        Box(12, 12, 12, mode=Mode.REPLACE)
        with Locations((20, 0, 0)):
            Sphere(4, mode=Mode.ADD)
    return part.part


def mixed_mode_sketch():
    with BuildSketch() as sketch:
        Rectangle(30, 30)
        Circle(8, mode=Mode.SUBTRACT)
        Rectangle(10, 10, mode=Mode.ADD)
        Circle(6, mode=Mode.INTERSECT)
        Rectangle(20, 8, mode=Mode.REPLACE)
        with Locations((12, 0)):
            Circle(3, mode=Mode.ADD)
    return sketch.sketch


def ordered_add_subtract_add_part():
    """ADD A, SUBTRACT B, ADD C where C overlaps the subtracted region."""
    with BuildPart() as part:
        Box(20, 20, 4)
        Cylinder(3, 10, mode=Mode.SUBTRACT)
        Cylinder(3, 4, mode=Mode.ADD)
    return part.part


def regrouped_add_then_subtract_part():
    """Incorrect global regrouping of the ordered ADD/SUBTRACT/ADD sequence."""
    with BuildPart() as part:
        Box(20, 20, 4)
        Cylinder(3, 4)
        Cylinder(3, 10, mode=Mode.SUBTRACT)
    return part.part


def synthetic_mesh(vertex_count: int):
    """Unique vertices along X with a simple triangle strip."""
    vertices = [(float(i), 0.0, 0.0) for i in range(vertex_count)]
    triangles = [[i, i + 1, i + 2] for i in range(0, vertex_count - 3, 3)]
    return vertices, triangles
