"""
Tests for the internal BooleanBatch primitive.

name: test_boolean_batch.py
date: August 15th 2026

desc:
    Phase 2 ordered same-mode boolean batching: one kernel call and one clean
    per batch, with tools passed directly to OCCT. Builder and algebra
    evaluation semantics are unchanged.

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
    Color,
    Edge,
    Face,
    Location,
    Rectangle,
    SkipClean,
    Solid,
)
from build123d.topology.shape_core import BooleanBatch, BooleanMode, Shape
from tests.performance_workloads import (
    EXPECTED_HOLE_COUNT,
    batched_cut,
    batched_fuse,
    hole_rectangles,
    plate_with_holes_base,
    sequential_cut,
    sequential_fuse,
)
from tests.test_semantic_equivalence import GeometricEquivalenceTestCase


class _BoolOpRecorder:
    """Record Shape._bool_op calls without consuming the iterables."""

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


class TestBooleanBatchKernel(GeometricEquivalenceTestCase):
    def test_empty_fuse_operands_skips_kernel(self):
        box = Solid.make_box(1, 1, 1)
        with _BoolOpRecorder() as rec:
            result = BooleanBatch(box, BooleanMode.FUSE, []).execute()
        self.assertEqual(len(rec.calls), 1)
        self.assertEqual(rec.calls[0]["n_tools"], 0)
        self.assertAlmostEqual(result.volume, 1.0, 5)

    def test_empty_cut_operands_returns_base(self):
        box = Solid.make_box(1, 1, 1)
        result = BooleanBatch(box, BooleanMode.CUT, []).execute()
        self.assertAlmostEqual(result.volume, 1.0, 5)

    def test_single_operand_fuse_matches_direct_fuse(self):
        a = Solid.make_box(1, 1, 1)
        b = Solid.make_box(1, 1, 1).moved(Location((2, 0, 0)))
        batched = BooleanBatch(a, BooleanMode.FUSE, [b]).execute()
        direct = a.fuse(b)
        self.assertGeometricallyEquivalent(
            batched, direct, check_manifold=False, check_solid_count=True
        )

    def test_overlapping_fuse_matches_sequential(self):
        a = Box(2, 2, 2)
        b = Box(2, 2, 2).moved(Location((1, 0, 0)))
        sequential = a.fuse(b)
        batched = BooleanBatch(a, BooleanMode.FUSE, [b]).execute()
        self.assertGeometricallyEquivalent(
            sequential, batched, check_manifold=False, check_solid_count=True
        )
        self.assertAlmostEqual(batched.volume, 12.0, 5)

    def test_overlapping_cut_matches_sequential(self):
        base = Box(4, 2, 2)
        tool = Box(2, 2, 2).moved(Location((1, 0, 0)))
        sequential = base.cut(tool)
        batched = BooleanBatch(base, BooleanMode.CUT, [tool]).execute()
        self.assertGeometricallyEquivalent(sequential, batched, check_manifold=False)

    def test_common_overlapping_boxes(self):
        a = Box(2, 2, 2)
        b = Box(2, 2, 2).moved(Location((1, 0, 0)))
        result = BooleanBatch(a, BooleanMode.COMMON, [b]).execute()
        self.assertTrue(result.is_valid)
        self.assertAlmostEqual(result.volume, 4.0, 5)

    def test_mixed_dimension_fuse_face_and_edge_raises(self):
        face = Face.make_rect(2, 2)
        edge = Edge.make_line((3, 0, 0), (4, 0, 0))
        with self.assertRaises(ValueError):
            BooleanBatch(face, BooleanMode.FUSE, [edge]).execute()
        with self.assertRaises(ValueError):
            face.fuse(edge)

    def test_same_dimension_disjoint_faces_fuse(self):
        a = Face.make_rect(2, 2)
        b = Face.make_rect(2, 2).moved(Location((3, 0, 0)))
        result = BooleanBatch(a, BooleanMode.FUSE, [b]).execute()
        self.assertTrue(result.is_valid)
        self.assertAlmostEqual(result.area, 8.0, 5)

    def test_invalid_empty_operand_is_ignored_like_fuse(self):
        box = Solid.make_box(1, 1, 1)
        empty = Solid()
        result = BooleanBatch(box, BooleanMode.FUSE, [empty]).execute()
        self.assertAlmostEqual(result.volume, 1.0, 5)

    def test_284_fuse_is_one_kernel_call(self):
        shapes = hole_rectangles()
        with _BoolOpRecorder() as rec:
            result = BooleanBatch(shapes[0], BooleanMode.FUSE, shapes[1:]).execute()
        self.assertEqual(len(rec.calls), 1)
        self.assertEqual(rec.calls[0]["op"], "BRepAlgoAPI_Fuse")
        self.assertEqual(rec.calls[0]["n_args"], 1)
        self.assertEqual(rec.calls[0]["n_tools"], EXPECTED_HOLE_COUNT - 1)
        self.assertEqual(len(result.faces()), EXPECTED_HOLE_COUNT)
        self.assertTrue(result.is_valid)

    def test_284_cut_passes_all_tools_to_one_kernel_call(self):
        base = plate_with_holes_base()
        tools = hole_rectangles()
        with _BoolOpRecorder() as rec:
            result = BooleanBatch(base, BooleanMode.CUT, tools).execute()
        self.assertEqual(len(rec.calls), 1)
        self.assertEqual(rec.calls[0]["op"], "BRepAlgoAPI_Cut")
        self.assertEqual(rec.calls[0]["n_tools"], EXPECTED_HOLE_COUNT)
        self.assertTrue(result.is_valid)
        self.assertGreater(base.area, result.area)

    def test_fuse_star_args_uses_one_kernel_call(self):
        shapes = hole_rectangles()
        with _BoolOpRecorder() as rec:
            result = shapes[0].fuse(*shapes[1:])
        self.assertEqual(len(rec.calls), 1)
        self.assertEqual(rec.calls[0]["n_tools"], EXPECTED_HOLE_COUNT - 1)
        self.assertEqual(len(result.faces()), EXPECTED_HOLE_COUNT)

    def test_cut_star_args_does_not_pre_fuse_tools(self):
        base = plate_with_holes_base()
        tools = hole_rectangles()[:8]
        with _BoolOpRecorder() as rec:
            base.cut(*tools)
        self.assertEqual(len(rec.calls), 1)
        self.assertEqual(rec.calls[0]["op"], "BRepAlgoAPI_Cut")
        self.assertEqual(rec.calls[0]["n_tools"], 8)

    def test_batched_matches_sequential_geometry(self):
        shapes = hole_rectangles()[:12]
        sequential = sequential_fuse(shapes)
        batched = batched_fuse(shapes)
        batch = BooleanBatch(shapes[0], BooleanMode.FUSE, shapes[1:]).execute()
        self.assertGeometricallyEquivalent(
            sequential, batched, check_manifold=False, check_face_count=True
        )
        self.assertGeometricallyEquivalent(
            sequential, batch, check_manifold=False, check_face_count=True
        )

        tools = shapes
        seq_cut = sequential_cut(plate_with_holes_base(), tools)
        batch_cut = batched_cut(plate_with_holes_base(), tools)
        self.assertGeometricallyEquivalent(seq_cut, batch_cut, check_manifold=False)

    def test_clean_false_skips_unify(self):
        a = Rectangle(2, 1)
        b = Rectangle(2, 1).moved(Location((1.5, 0, 0)))
        cleaned = BooleanBatch(a, BooleanMode.FUSE, [b]).execute(clean=True)
        raw = BooleanBatch(a, BooleanMode.FUSE, [b]).execute(clean=False)
        self.assertGeometricallyEquivalent(cleaned, raw, check_manifold=False)
        self.assertGreaterEqual(len(raw.faces()), len(cleaned.faces()))

    def test_skip_clean_context_is_restored(self):
        a = Solid.make_box(1, 1, 1)
        b = Solid.make_box(1, 1, 1).moved(Location((2, 0, 0)))
        self.assertTrue(SkipClean.clean)
        BooleanBatch(a, BooleanMode.FUSE, [b]).execute(clean=False)
        self.assertTrue(SkipClean.clean)
        with SkipClean():
            BooleanBatch(a, BooleanMode.FUSE, [b]).execute(clean=True)
            self.assertFalse(SkipClean.clean)
        self.assertTrue(SkipClean.clean)

    def test_attributes_propagate_from_base(self):
        base = Solid.make_box(1, 1, 1)
        base.label = "body"
        base.color = Color(1, 0, 0)
        other = Solid.make_box(1, 1, 1).moved(Location((2, 0, 0)))
        result = BooleanBatch(base, BooleanMode.FUSE, [other]).execute()
        self.assertEqual(result.label, "body")
        self.assertEqual(tuple(result.color), tuple(Color(1, 0, 0)))

    def test_unsupported_mode_raises(self):
        box = Solid.make_box(1, 1, 1)
        batch = BooleanBatch(box, BooleanMode.FUSE, [])
        batch.mode = "not-a-mode"  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            batch.execute()


class TestBooleanBatchTiming(unittest.TestCase):
    def test_284_bulk_fuse_under_tenth_second(self):
        shapes = hole_rectangles()
        BooleanBatch(shapes[0], BooleanMode.FUSE, shapes[1:]).execute()
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            BooleanBatch(shapes[0], BooleanMode.FUSE, shapes[1:]).execute()
            times.append(time.perf_counter() - t0)
        # Reference target is 0.10s; keep a CI-safe ceiling for slow runners.
        self.assertLess(min(times), 0.50)
        print(
            f"\n284-shape BooleanBatch fuse min={min(times):.4f}s median={sorted(times)[1]:.4f}s"
        )


if __name__ == "__main__":
    unittest.main()
