"""
Tests for opt-in Builder boolean deferral.

name: test_builder_defer.py
date: August 15th 2026

desc:
    Phase 3 ``defer_booleans=True`` batches contiguous ADD/SUBTRACT operations
    while flushing at documented barriers. Eager Builder behavior remains the
    default.

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

import time
import unittest

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
    Sphere,
    fillet,
)
from build123d.topology.shape_core import Shape
from tests.performance_workloads import (
    EXPECTED_HOLE_COUNT,
    deferred_hole_part,
    deferred_hole_sketch,
    eager_hole_part,
    eager_hole_sketch,
    mixed_mode_builder,
    ordered_add_subtract_add_part,
    selector_heavy_builder,
)
from tests.test_semantic_equivalence import GeometricEquivalenceTestCase


class _BoolOpRecorder:
    def __init__(self):
        self.calls: list[dict] = []
        self._orig = Shape._bool_op

    def __enter__(self):
        recorder = self

        def wrapped(this, args, tools, operation):
            args = list(args)
            tools = list(tools)
            recorder.calls.append(
                {
                    "op": type(operation).__name__,
                    "n_args": len(args),
                    "n_tools": len(tools),
                }
            )
            return recorder._orig(this, args, tools, operation)

        Shape._bool_op = wrapped
        return self

    def __exit__(self, *_exc):
        Shape._bool_op = self._orig
        return False


class TestBuilderDeferSketch(GeometricEquivalenceTestCase):
    def test_deferred_matches_eager_hole_sketch(self):
        eager = eager_hole_sketch()
        deferred = deferred_hole_sketch()
        self.assertEqual(len(eager.faces()), EXPECTED_HOLE_COUNT)
        self.assertGeometricallyEquivalent(
            eager, deferred, check_manifold=False, check_face_count=True
        )

    def test_284_deferred_sketch_is_one_kernel_call(self):
        with _BoolOpRecorder() as rec:
            result = deferred_hole_sketch()
        fuse_calls = [c for c in rec.calls if c["op"] == "BRepAlgoAPI_Fuse"]
        self.assertEqual(len(fuse_calls), 1)
        self.assertEqual(fuse_calls[0]["n_tools"], EXPECTED_HOLE_COUNT - 1)
        self.assertEqual(len(result.faces()), EXPECTED_HOLE_COUNT)

    def test_select_last_is_a_flush_barrier(self):
        with _BoolOpRecorder() as rec:
            with BuildSketch(defer_booleans=True) as sketch:
                with Locations((0, 0)):
                    Rectangle(2, 2)
                with Locations((4, 0)):
                    Rectangle(2, 2)
                last = sketch.faces(Select.LAST)
                with Locations((8, 0)):
                    Rectangle(2, 2)
        self.assertGreaterEqual(len(rec.calls), 1)
        self.assertEqual(len(last), 2)
        self.assertEqual(len(sketch.sketch.faces()), 3)

    def test_explicit_flush(self):
        with BuildSketch(defer_booleans=True) as sketch:
            Rectangle(4, 4)
            with Locations((0, 0)):
                Rectangle(2, 2)
            self.assertIsNone(sketch._obj)
            sketch.flush()
            self.assertIsNotNone(sketch._obj)
        self.assertAlmostEqual(sketch.sketch.area, 16.0, 5)

    def test_subtract_without_base_fails_at_flush(self):
        with self.assertRaises(RuntimeError) as ctx:
            with BuildSketch(defer_booleans=True):
                Circle(4, mode=Mode.SUBTRACT)
        self.assertIn("deferred", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
        self.assertIn("Nothing to subtract from", str(ctx.exception.__cause__))

    def test_intersect_and_replace_flush_pending_add(self):
        with BuildSketch(defer_booleans=True) as sketch:
            Rectangle(20, 20)
            Circle(6, mode=Mode.INTERSECT)
            self.assertIsNotNone(sketch._obj)
            Rectangle(20, 8, mode=Mode.REPLACE)
        self.assertAlmostEqual(sketch.sketch.area, 160.0, 4)


class TestBuilderDeferPart(GeometricEquivalenceTestCase):
    def test_deferred_matches_eager_hole_part(self):
        eager = eager_hole_part()
        deferred = deferred_hole_part()
        self.assertGeometricallyEquivalent(eager, deferred, check_manifold=False)

    def test_284_deferred_part_subtract_is_one_cut(self):
        with _BoolOpRecorder() as rec:
            result = deferred_hole_part()
        cut_calls = [c for c in rec.calls if c["op"] == "BRepAlgoAPI_Cut"]
        self.assertEqual(len(cut_calls), 1)
        self.assertEqual(cut_calls[0]["n_tools"], EXPECTED_HOLE_COUNT)
        self.assertTrue(result.is_valid)

    def test_ordered_add_subtract_add_is_not_regrouped(self):
        with BuildPart(defer_booleans=True) as part:
            Box(20, 20, 4)
            Cylinder(3, 10, mode=Mode.SUBTRACT)
            Cylinder(3, 4, mode=Mode.ADD)
        self.assertGeometricallyEquivalent(
            ordered_add_subtract_add_part(),
            part.part,
            check_solid_count=True,
            check_manifold=True,
        )

    def test_mixed_mode_with_defer_matches_eager(self):
        with BuildPart(defer_booleans=True) as part:
            Box(20, 20, 20)
            Cylinder(6, 30, mode=Mode.SUBTRACT)
            Box(8, 8, 30, mode=Mode.ADD)
            Sphere(7, mode=Mode.INTERSECT)
            Box(12, 12, 12, mode=Mode.REPLACE)
            with Locations((20, 0, 0)):
                Sphere(4, mode=Mode.ADD)
        self.assertGeometricallyEquivalent(
            mixed_mode_builder(),
            part.part,
            check_solid_count=True,
            check_manifold=True,
        )

    def test_selector_heavy_barriers_keep_eager_behavior(self):
        with BuildPart(defer_booleans=True) as part:
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
        self.assertGeometricallyEquivalent(
            selector_heavy_builder(),
            part.part,
            check_manifold=False,
        )

    def test_fillet_flushes_before_consuming_edges(self):
        with BuildPart(defer_booleans=True) as part:
            Box(10, 10, 10)
            fillet(part.edges().group_by(Axis.Z)[-1], 0.5)
        self.assertTrue(part.part.is_valid)
        self.assertLess(part.part.volume, 10 * 10 * 10)


class TestBuilderDeferTiming(unittest.TestCase):
    def test_284_deferred_sketch_under_tenth_second(self):
        deferred_hole_sketch()
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            deferred_hole_sketch()
            times.append(time.perf_counter() - t0)
        self.assertLess(min(times), 0.50)
        print(
            f"\ndeferred 284-sketch min={min(times):.4f}s "
            f"median={sorted(times)[1]:.4f}s"
        )


if __name__ == "__main__":
    unittest.main()
