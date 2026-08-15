"""
Semantic-equivalence corpus for performance work.

name: test_semantic_equivalence.py
date: August 15th 2026

desc:
    Observable-behavior tests that later boolean-batching and wrapper changes
    must preserve. Geometric comparisons use volume, area, bounding box,
    validity, and manifold state. Face/edge identity and traversal order are
    not part of the promised contract: sequential and batched OCCT operations
    may split faces differently while remaining geometrically equivalent.

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

import unittest

from bd_materials import metals

from build123d import (
    Align,
    Axis,
    Box,
    BuildPart,
    BuildSketch,
    Circle,
    Color,
    Compound,
    Cylinder,
    Location,
    Locations,
    Mode,
    Rectangle,
    RigidJoint,
    Select,
    Sketch,
    Solid,
    Sphere,
    extrude,
    fillet,
)
from tests.performance_workloads import (
    EXPECTED_FACE_GRID_COUNT,
    EXPECTED_HOLE_COUNT,
    batched_algebra_fuse,
    batched_cut,
    batched_fuse,
    face_grid_rectangles,
    face_grid_sketch,
    hole_rectangles,
    mixed_mode_builder,
    mixed_mode_sketch,
    ordered_add_subtract_add_part,
    plate_with_holes_base,
    regrouped_add_then_subtract_part,
    sequential_algebra_fuse,
    sequential_cut,
    sequential_fuse,
)


def _bbox_minmax(shape):
    bbox = shape.bounding_box()
    return (
        (bbox.min.X, bbox.min.Y, bbox.min.Z),
        (bbox.max.X, bbox.max.Y, bbox.max.Z),
    )


class GeometricEquivalenceTestCase(unittest.TestCase):
    """Compare promised geometric properties, not OCCT face identity."""

    def assertAlmostTuple(self, first, second, places=5, msg=None):
        self.assertEqual(len(first), len(second), msg)
        for left, right in zip(first, second):
            self.assertAlmostEqual(left, right, places, msg)

    def assertGeometricallyEquivalent(
        self,
        first,
        second,
        *,
        places=5,
        check_manifold=True,
        check_solid_count=False,
        check_face_count=False,
        check_edge_count=False,
    ):
        self.assertAlmostEqual(first.volume, second.volume, places)
        self.assertAlmostEqual(first.area, second.area, places - 1)
        first_min, first_max = _bbox_minmax(first)
        second_min, second_max = _bbox_minmax(second)
        self.assertAlmostTuple(first_min, second_min, places)
        self.assertAlmostTuple(first_max, second_max, places)
        self.assertEqual(first.is_valid, second.is_valid)
        self.assertTrue(first.is_valid)
        if check_manifold:
            self.assertEqual(first.is_manifold, second.is_manifold)
        if check_solid_count:
            self.assertEqual(len(first.solids()), len(second.solids()))
        if check_face_count:
            self.assertEqual(len(first.faces()), len(second.faces()))
        if check_edge_count:
            self.assertEqual(len(first.edges()), len(second.edges()))


class TestHolePatternWorkloads(GeometricEquivalenceTestCase):
    def test_hole_count(self):
        self.assertEqual(len(hole_rectangles()), EXPECTED_HOLE_COUNT)

    def test_sequential_and_batched_fuse_match(self):
        shapes = hole_rectangles()[:12]
        sequential = sequential_fuse(shapes)
        batched = batched_fuse(shapes)
        algebra = sequential_algebra_fuse(shapes)
        algebra_batched = batched_algebra_fuse(shapes)
        self.assertGeometricallyEquivalent(
            sequential,
            batched,
            check_manifold=False,
            check_face_count=True,
        )
        self.assertGeometricallyEquivalent(
            sequential,
            algebra,
            check_manifold=False,
            check_face_count=True,
        )
        self.assertGeometricallyEquivalent(
            sequential,
            algebra_batched,
            check_manifold=False,
            check_face_count=True,
        )

    def test_batched_284_fuse_is_valid(self):
        shapes = hole_rectangles()
        fused = batched_fuse(shapes)
        algebra = batched_algebra_fuse(shapes)
        self.assertGeometricallyEquivalent(
            fused, algebra, check_manifold=False, check_face_count=True
        )
        self.assertEqual(len(fused.faces()), EXPECTED_HOLE_COUNT)

    def test_sequential_and_batched_cut_match(self):
        tools = hole_rectangles()[:12]
        sequential = sequential_cut(plate_with_holes_base(), tools)
        batched = batched_cut(plate_with_holes_base(), tools)
        algebra = plate_with_holes_base() - tools
        self.assertGeometricallyEquivalent(sequential, batched, check_manifold=False)
        self.assertGeometricallyEquivalent(sequential, algebra, check_manifold=False)
        self.assertGreater(plate_with_holes_base().area, sequential.area)

    def test_batched_284_cut_is_valid(self):
        tools = hole_rectangles()
        result = batched_cut(plate_with_holes_base(), tools)
        algebra = plate_with_holes_base() - tools
        self.assertGeometricallyEquivalent(result, algebra, check_manifold=False)
        self.assertTrue(result.is_valid)

    def test_fuse_compares_geometry_not_face_identity(self):
        shapes = hole_rectangles()[:8]
        sequential = sequential_fuse(shapes)
        batched = batched_fuse(shapes)
        self.assertGeometricallyEquivalent(
            sequential, batched, check_manifold=False, check_face_count=True
        )
        self.assertIsNot(sequential, batched)


class TestFaceGrid(GeometricEquivalenceTestCase):
    def test_face_grid_count(self):
        rectangles = face_grid_rectangles()
        self.assertEqual(len(rectangles), EXPECTED_FACE_GRID_COUNT)
        sketch = face_grid_sketch()
        self.assertEqual(len(sketch.faces()), EXPECTED_FACE_GRID_COUNT)
        self.assertTrue(sketch.is_valid)
        self.assertAlmostEqual(sketch.area, EXPECTED_FACE_GRID_COUNT, 5)


class TestSelectAtBarriers(unittest.TestCase):
    def test_select_last_and_new_after_each_part_operation(self):
        with BuildPart() as part:
            Box(10, 10, 10)
            self.assertEqual(len(part.faces(Select.LAST)), 6)
            self.assertEqual(len(part.edges(Select.LAST)), 12)
            self.assertEqual(len(part.vertices(Select.LAST)), 8)
            self.assertEqual(len(part.solids(Select.LAST)), 1)
            # First operation has no prior topology, so Select.NEW is empty.
            self.assertEqual(len(part.edges(Select.NEW)), 0)

            Box(5, 5, 20, align=(Align.CENTER, Align.CENTER, Align.MIN))
            self.assertEqual(len(part.faces(Select.LAST)), 6)
            self.assertGreater(len(part.edges(Select.NEW)), 0)
            self.assertEqual(len(part.solids(Select.LAST)), 1)

            fillet(part.edges().group_by(Axis.Z)[-1], 1)
            self.assertGreater(len(part.edges(Select.LAST)), 0)
            _ = part.edges(Select.NEW)

            with BuildSketch(part.faces().filter_by(Axis.Z)[-1]):
                Circle(1)
            extruded = extrude(amount=2)
            self.assertEqual(len(part.solids(Select.LAST)), 1)
            self.assertGreater(extruded.volume, 0)
            _ = part.edges(Select.NEW)

        self.assertEqual(len(part.solids()), 1)
        self.assertTrue(part.part.is_valid)
        self.assertTrue(part.part.is_manifold)

    def test_select_new_after_cut_seams(self):
        with BuildPart() as part:
            Sphere(5)
            Cylinder(2, 20, mode=Mode.SUBTRACT)
            self.assertGreater(len(part.edges(Select.NEW)), 0)
            self.assertGreater(len(part.edges(Select.LAST)), 0)
        self.assertTrue(part.part.is_valid)

    def test_select_last_and_new_after_each_sketch_operation(self):
        with BuildSketch() as sketch:
            Rectangle(20, 20)
            self.assertEqual(len(sketch.faces(Select.LAST)), 1)
            self.assertEqual(len(sketch.edges(Select.LAST)), 4)
            self.assertEqual(len(sketch.vertices(Select.LAST)), 4)

            Circle(5, mode=Mode.SUBTRACT)
            self.assertEqual(len(sketch.faces(Select.LAST)), 1)
            self.assertEqual(len(sketch.edges(Select.LAST)), 1)

            Rectangle(4, 4)
            self.assertEqual(len(sketch.faces(Select.LAST)), 1)
            _ = sketch.edges(Select.NEW)

        self.assertTrue(sketch.sketch.is_valid)
        self.assertAlmostEqual(sketch.sketch.area, 20 * 20 - 25 * 3.1415926535 + 16, 3)

    def test_select_new_rejected_for_non_edges(self):
        with BuildPart() as part:
            Box(4, 4, 4)
            with self.assertRaises(ValueError):
                part.faces(Select.NEW)
            with self.assertRaises(ValueError):
                part.vertices(Select.NEW)
            with self.assertRaises(ValueError):
                part.solids(Select.NEW)
            with self.assertRaises(ValueError):
                part.wires(Select.NEW)


class TestOrderedBooleanModes(GeometricEquivalenceTestCase):
    def test_builder_modes_add_subtract_intersect_replace(self):
        part = mixed_mode_builder()
        self.assertTrue(part.is_valid)
        self.assertEqual(len(part.solids()), 2)
        self.assertAlmostEqual(
            part.volume, Box(12, 12, 12).volume + Sphere(4).volume, 4
        )
        self.assertTrue(all(solid.is_valid for solid in part.solids()))

    def test_mixed_mode_replace_is_the_current_object(self):
        with BuildPart() as part:
            Box(20, 20, 20)
            Sphere(12, mode=Mode.REPLACE)
            replaced_volume = part.part.volume
            self.assertAlmostEqual(replaced_volume, Sphere(12).volume, 5)
        self.assertAlmostEqual(part.part.volume, Sphere(12).volume, 5)

    def test_ordered_add_subtract_add_is_not_regrouped(self):
        ordered = ordered_add_subtract_add_part()
        regrouped = regrouped_add_then_subtract_part()
        filled = Box(20, 20, 4)
        self.assertGeometricallyEquivalent(
            ordered, filled, check_solid_count=True, check_manifold=True
        )
        self.assertLess(regrouped.volume, ordered.volume)

    def test_sketch_mode_sequence(self):
        sketch = mixed_mode_sketch()
        self.assertTrue(sketch.is_valid)
        self.assertGreater(sketch.area, 20 * 8)
        with BuildSketch() as replaced:
            Rectangle(20, 8)
            with Locations((12, 0)):
                Circle(3)
        self.assertAlmostEqual(sketch.area, replaced.sketch.area, 5)

    def test_algebra_ordered_combinations(self):
        box = Box(10, 10, 10)
        cylinder = Cylinder(3, 20)
        sphere = Sphere(4)
        added = box + cylinder
        subtracted = box - cylinder
        intersected = box & sphere
        self.assertGreater(added.volume, box.volume)
        self.assertLess(subtracted.volume, box.volume)
        self.assertAlmostEqual(intersected.volume, (box & sphere).volume, 5)
        self.assertTrue(added.is_valid)
        self.assertTrue(subtracted.is_valid)
        self.assertTrue(intersected.is_valid)

        # Ordered add-then-subtract vs subtract-then-add.
        add_then_subtract = (box + cylinder) - sphere
        subtract_then_add = (box - sphere) + cylinder
        self.assertGreater(
            abs(add_then_subtract.volume - subtract_then_add.volume), 1.0
        )


class TestAttributesAndAssemblies(GeometricEquivalenceTestCase):
    def test_fuse_copies_label_color_material_and_joints(self):
        base = Box(10, 10, 10)
        base.label = "base"
        base.color = Color(1, 0, 0)
        base.material = metals.brass()
        joint = RigidJoint("mount", base, Location((0, 0, 5)))
        other = Box(4, 4, 4).locate(Location((7, 0, 0)))
        result = base.fuse(other)
        self.assertEqual(result.label, "base")
        self.assertAlmostTuple(tuple(result.color), tuple(base.color))
        self.assertEqual(result.material.material.name, base.material.material.name)
        self.assertIn("mount", result.joints)
        self.assertIs(result.joints["mount"].parent, result)
        self.assertEqual(result.joints["mount"].label, joint.label)

    def test_cut_preserves_base_attributes(self):
        base = Box(10, 10, 10)
        base.label = "stock"
        base.color = Color(0, 0, 1)
        base.material = metals.brass()
        result = base.cut(Cylinder(2, 20))
        self.assertEqual(result.label, "stock")
        self.assertAlmostTuple(tuple(result.color), tuple(base.color))
        self.assertEqual(result.material.material.name, base.material.material.name)

    def test_assembly_relationships(self):
        box = Solid.make_box(1, 1, 1)
        box.label = "box"
        box.color = Color("green")
        sphere = Solid.make_sphere(0.5)
        sphere.label = "sphere"
        assembly = Compound(label="assembly", children=[box])
        sphere.parent = assembly
        self.assertEqual(box.parent, assembly)
        self.assertEqual(sphere.parent, assembly)
        self.assertEqual(len(assembly.children), 2)
        self.assertEqual(
            [child.label for child in assembly.children], ["box", "sphere"]
        )
        self.assertEqual(assembly.label, "assembly")
        moved = assembly.locate(Location((10, 0, 0)))
        self.assertEqual(moved.label, "assembly")
        self.assertEqual(len(moved.children), 2)
        self.assertEqual(box.parent, assembly)

    def test_algebra_fuse_list_keeps_sketch_type(self):
        result = Sketch() + hole_rectangles()[:4]
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.faces()), 4)
        self.assertAlmostEqual(result.area, 16.0, 5)


class TestErrorTypeAndTiming(unittest.TestCase):
    def test_subtract_without_base_raises_during_operation(self):
        with self.assertRaises(RuntimeError):
            with BuildPart():
                Sphere(10, mode=Mode.SUBTRACT)
                self.fail("SUBTRACT must fail before the next statement")

    def test_intersect_without_base_raises_during_operation(self):
        with self.assertRaises(RuntimeError):
            with BuildPart():
                Sphere(10, mode=Mode.INTERSECT)
                self.fail("INTERSECT must fail before the next statement")

    def test_sketch_subtract_without_base_raises_during_operation(self):
        with self.assertRaises(RuntimeError):
            with BuildSketch():
                Circle(10, mode=Mode.SUBTRACT)
                self.fail("SUBTRACT must fail before the next statement")

    def test_empty_cut_raises(self):
        empty = Sketch()
        with self.assertRaises(ValueError):
            empty - Rectangle(1, 1)


if __name__ == "__main__":
    unittest.main()
