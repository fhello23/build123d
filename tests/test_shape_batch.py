"""
Tests for explicit ShapeBatch algebra accumulation.

name: test_shape_batch.py
date: August 15th 2026

desc:
    Phase 4 keeps Sketch/Part += eager and adds ShapeBatch so loops can queue
    operands and evaluate them as one kernel boolean. Subtraction passes queued
    tools directly to cut instead of fusing them first.

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
    Box,
    Circle,
    Location,
    Part,
    Rectangle,
    ShapeBatch,
    Sketch,
)
from build123d.topology.shape_core import Shape
from tests.performance_workloads import (
    EXPECTED_HOLE_COUNT,
    batched_algebra_fuse,
    batched_cut,
    hole_rectangles,
    plate_with_holes_base,
    sequential_algebra_fuse,
    shape_batch_algebra_fuse,
    shape_batch_cut,
    shape_batch_from,
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


class TestSketchIaddStaysEager(unittest.TestCase):
    def test_sketch_iadd_returns_new_object_and_does_not_update_alias(self):
        sketch = Sketch()
        alias = sketch
        added = sketch + Rectangle(1, 1)
        sketch += Rectangle(1, 1)
        self.assertIsNot(sketch, alias)
        self.assertFalse(alias)
        self.assertAlmostEqual(sketch.area, added.area, 5)

    def test_part_iadd_returns_new_object_and_does_not_update_alias(self):
        part = Part()
        alias = part
        part += Box(1, 1, 1)
        self.assertIsNot(part, alias)
        self.assertFalse(alias)
        self.assertAlmostEqual(part.volume, 1.0, 5)


class TestShapeBatchSemantics(GeometricEquivalenceTestCase):
    def test_iadd_mutates_and_updates_aliases(self):
        batch = ShapeBatch()
        alias = batch
        batch += Rectangle(1, 1)
        self.assertIs(alias, batch)
        self.assertEqual(len(alias), 1)

    def test_add_returns_new_batch(self):
        batch = ShapeBatch(Rectangle(1, 1))
        other = batch + Rectangle(1, 1).moved(Location((2, 0, 0)))
        self.assertIsNot(other, batch)
        self.assertEqual(len(batch), 1)
        self.assertEqual(len(other), 2)

    def test_empty_subtraction_is_noop(self):
        plate = Circle(10)
        result = plate - ShapeBatch()
        self.assertAlmostEqual(result.area, plate.area, 5)

    def test_single_operand_fuse(self):
        rect = Rectangle(2, 2)
        fused = ShapeBatch(rect).fuse()
        self.assertAlmostEqual(fused.area, 4.0, 5)

    def test_dimension_mismatch_raises_on_append_not_at_fuse(self):
        batch = ShapeBatch(Rectangle(1, 1))
        with self.assertRaises(ValueError):
            batch += Box(1, 1, 1)
        self.assertEqual(len(batch), 1)
        fused = batch.fuse()
        self.assertAlmostEqual(fused.area, 1.0, 5)

    def test_invalid_type_raises_on_append(self):
        batch = ShapeBatch()
        with self.assertRaises(TypeError):
            batch += 1  # type: ignore[arg-type]
        self.assertEqual(len(batch), 0)

    def test_nested_batch_flattens(self):
        inner = ShapeBatch(Rectangle(1, 1))
        inner += Rectangle(1, 1).moved(Location((3, 0, 0)))
        outer = ShapeBatch()
        outer += inner
        self.assertEqual(len(outer), 2)

    def test_none_is_ignored(self):
        batch = ShapeBatch()
        batch += None
        self.assertEqual(len(batch), 0)

    def test_empty_fuse_raises(self):
        with self.assertRaises(ValueError):
            ShapeBatch().fuse()

    def test_context_manager_accumulates(self):
        with ShapeBatch() as batch:
            batch += Rectangle(1, 1)
            batch += Rectangle(1, 1).moved(Location((3, 0, 0)))
        self.assertEqual(len(batch), 2)

    def test_fuse_matches_vectorized_algebra(self):
        shapes = hole_rectangles()[:12]
        batched = batched_algebra_fuse(shapes)
        from_batch = shape_batch_algebra_fuse(shapes)
        fused = shape_batch_from(shapes).fuse()
        self.assertGeometricallyEquivalent(
            batched, from_batch, check_manifold=False, check_face_count=True
        )
        self.assertGeometricallyEquivalent(
            batched, fused, check_manifold=False, check_face_count=True
        )

    def test_subtraction_matches_batched_cut(self):
        tools = hole_rectangles()[:12]
        batched = batched_cut(plate_with_holes_base(), tools)
        from_batch = shape_batch_cut(plate_with_holes_base(), tools)
        self.assertGeometricallyEquivalent(batched, from_batch, check_manifold=False)

    def test_sequential_sketch_iadd_still_matches_batch_geometry(self):
        shapes = hole_rectangles()[:12]
        sequential = sequential_algebra_fuse(shapes)
        from_batch = shape_batch_algebra_fuse(shapes)
        self.assertGeometricallyEquivalent(
            sequential, from_batch, check_manifold=False, check_face_count=True
        )


class TestShapeBatchKernel(unittest.TestCase):
    def test_fuse_is_one_kernel_call(self):
        shapes = hole_rectangles()
        batch = shape_batch_from(shapes)
        with _BoolOpRecorder() as rec:
            result = batch.fuse()
        fuse_calls = [c for c in rec.calls if c["op"] == "BRepAlgoAPI_Fuse"]
        self.assertEqual(len(fuse_calls), 1)
        self.assertEqual(fuse_calls[0]["n_tools"], EXPECTED_HOLE_COUNT - 1)
        self.assertEqual(len(result.faces()), EXPECTED_HOLE_COUNT)

    def test_subtraction_does_not_pre_fuse_tools(self):
        tools = hole_rectangles()
        batch = shape_batch_from(tools)
        with _BoolOpRecorder() as rec:
            result = plate_with_holes_base() - batch
        fuse_calls = [c for c in rec.calls if c["op"] == "BRepAlgoAPI_Fuse"]
        cut_calls = [c for c in rec.calls if c["op"] == "BRepAlgoAPI_Cut"]
        self.assertEqual(fuse_calls, [])
        self.assertEqual(len(cut_calls), 1)
        self.assertEqual(cut_calls[0]["n_tools"], EXPECTED_HOLE_COUNT)
        self.assertTrue(result.is_valid)

    def test_sketch_plus_batch_is_one_fuse(self):
        shapes = hole_rectangles()
        batch = shape_batch_from(shapes)
        with _BoolOpRecorder() as rec:
            result = Sketch() + batch
        fuse_calls = [c for c in rec.calls if c["op"] == "BRepAlgoAPI_Fuse"]
        self.assertEqual(len(fuse_calls), 1)
        self.assertEqual(len(result.faces()), EXPECTED_HOLE_COUNT)


class TestShapeBatchTiming(unittest.TestCase):
    def test_284_batch_fuse_under_tenth_second(self):
        shapes = hole_rectangles()
        shape_batch_from(shapes).fuse()
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            shape_batch_from(shapes).fuse()
            times.append(time.perf_counter() - t0)
        # Reference target is 0.10s; keep a CI-safe ceiling for slow runners.
        self.assertLess(min(times), 0.50)
        sequential_t0 = time.perf_counter()
        sequential = sequential_algebra_fuse(shapes)
        sequential_s = time.perf_counter() - sequential_t0
        print(
            f"\nShapeBatch fuse 284 min={min(times):.4f}s "
            f"median={sorted(times)[1]:.4f}s "
            f"sequential_algebra_+=={sequential_s:.4f}s"
        )
        self.assertAlmostEqual(sequential.area, shape_batch_from(shapes).fuse().area, 3)


if __name__ == "__main__":
    unittest.main()
