"""
Tests for NumPy mesh vertex welding and triangle remapping.

name: test_mesh_weld.py
date: August 15th 2026

desc:
    Phase 5 vectorizes Mesher._weld_mesh_primitives. These tests pin first-seen
    vertex order, rounding-boundary welding, winding, and degenerate removal.
    Full 3MF/STL byte equality is not required.

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

import math
import os
import tempfile
import time
import unittest

import numpy as np

from build123d.geometry import TOLERANCE
from build123d.mesher import Mesher
from build123d.topology import Solid
from tests.performance_workloads import synthetic_mesh


def _rows(arr) -> list[tuple]:
    """Convert a weld result array to a list of tuples for equality checks."""
    return [tuple(row) for row in np.asarray(arr).tolist()]


def _weld_digits() -> int:
    return -int(round(math.log(TOLERANCE, 10), 1))


def _python_weld(vertices, triangles):
    """Reference implementation of the pre-NumPy welder."""
    digits = _weld_digits()
    vertex_to_idx = {}
    unique_vertices = []
    vert_table = {}
    for i, (x, y, z) in enumerate(vertices):
        key = (round(x, digits), round(y, digits), round(z, digits))
        mapped = vertex_to_idx.get(key)
        if mapped is None:
            mapped = len(unique_vertices)
            vertex_to_idx[key] = mapped
            unique_vertices.append(key)
        vert_table[i] = mapped
    remapped = []
    for tri in triangles:
        a, b, c = vert_table[tri[0]], vert_table[tri[1]], vert_table[tri[2]]
        if a != b and b != c and c != a:
            remapped.append((a, b, c))
    return unique_vertices, remapped


class TestWeldEmptyAndDegenerate(unittest.TestCase):
    def test_empty_vertices(self):
        unique, remapped = Mesher._weld_mesh_primitives([], [])
        self.assertEqual(_rows(unique), [])
        self.assertEqual(_rows(remapped), [])

    def test_vertices_without_triangles(self):
        unique, remapped = Mesher._weld_mesh_primitives(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], []
        )
        self.assertEqual(_rows(unique), [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        self.assertEqual(_rows(remapped), [])

    def test_degenerate_triangles_dropped(self):
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        triangles = [[0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 2]]
        unique, remapped = Mesher._weld_mesh_primitives(vertices, triangles)
        self.assertEqual(len(unique), 3)
        self.assertEqual(_rows(remapped), [(0, 1, 2)])

    def test_degenerate_after_weld_dropped(self):
        vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1e-10, 0.0, 0.0),
        ]
        triangles = [[0, 1, 2]]
        unique, remapped = Mesher._weld_mesh_primitives(vertices, triangles)
        self.assertEqual(len(unique), 2)
        self.assertEqual(_rows(remapped), [])


class TestWeldOrderAndWinding(unittest.TestCase):
    def test_first_seen_order_with_interleaved_duplicates(self):
        vertices = [
            (2.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
        triangles = [[1, 2, 5], [3, 2, 5], [0, 1, 2]]
        unique, remapped = Mesher._weld_mesh_primitives(vertices, triangles)
        self.assertEqual(
            _rows(unique),
            [
                (2.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
        )
        self.assertEqual(_rows(remapped), [(1, 2, 3), (1, 2, 3), (0, 1, 2)])

    def test_triangle_winding_is_preserved(self):
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        triangles = [[2, 0, 1], [1, 2, 0]]
        _unique, remapped = Mesher._weld_mesh_primitives(vertices, triangles)
        self.assertEqual(_rows(remapped), [(2, 0, 1), (1, 2, 0)])

    def test_negative_zero_welds_with_positive_zero(self):
        vertices = [
            (0.0, 0.0, 0.0),
            (-0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
        triangles = [[0, 2, 3], [1, 2, 3]]
        unique, remapped = Mesher._weld_mesh_primitives(vertices, triangles)
        self.assertEqual(len(unique), 3)
        self.assertEqual(_rows(remapped), [(0, 1, 2), (0, 1, 2)])


class TestWeldRoundingBoundaries(unittest.TestCase):
    def test_values_inside_half_quantum_weld(self):
        digits = _weld_digits()
        quantum = 10 ** (-digits)
        delta = 0.4 * quantum
        vertices = [
            (1.0, 0.0, 0.0),
            (1.0 + delta, 0.0, 0.0),
            (1.0 - delta, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        triangles = [[0, 3, 4], [1, 3, 4], [2, 3, 4]]
        unique, remapped = Mesher._weld_mesh_primitives(vertices, triangles)
        self.assertEqual(len(unique), 3)
        self.assertEqual(_rows(remapped), [(0, 1, 2), (0, 1, 2), (0, 1, 2)])

    def test_values_on_opposite_sides_of_boundary_stay_distinct(self):
        digits = _weld_digits()
        quantum = 10 ** (-digits)
        vertices = [
            (1.0 + 0.6 * quantum, 0.0, 0.0),
            (1.0 - 0.6 * quantum, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        triangles = [[0, 2, 3], [1, 2, 3]]
        unique, remapped = Mesher._weld_mesh_primitives(vertices, triangles)
        self.assertEqual(len(unique), 4)
        self.assertEqual(_rows(remapped), [(0, 2, 3), (1, 2, 3)])

    def test_first_seen_matches_python_on_pre_rounded_vertices(self):
        rng = np.random.default_rng(20260815)
        digits = _weld_digits()
        raw = rng.normal(scale=10.0, size=(500, 3))
        vertices = [tuple(np.round(row, decimals=digits)) for row in raw]
        vertices.extend(vertices[:80])
        triangles = [[i, i + 1, i + 2] for i in range(0, len(vertices) - 2, 3)]
        expected_unique, expected_tris = _python_weld(vertices, triangles)
        unique, remapped = Mesher._weld_mesh_primitives(vertices, triangles)
        self.assertEqual(_rows(unique), expected_unique)
        self.assertEqual(_rows(remapped), expected_tris)


class TestWeldRoundTrip(unittest.TestCase):
    def _round_trip(self, suffix: str):
        box = Solid.make_box(2, 3, 4)
        exporter = Mesher()
        exporter.add_shape(box)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"box{suffix}")
            exporter.write(path)
            imported = Mesher().read(path)
        self.assertEqual(len(imported), 1)
        self.assertTrue(imported[0].is_valid)
        self.assertAlmostEqual(imported[0].volume, box.volume, places=1)

    def test_box_3mf_round_trip_volume(self):
        self._round_trip(".3mf")

    def test_box_stl_round_trip_volume(self):
        self._round_trip(".stl")


class TestWeldTiming(unittest.TestCase):
    def test_100k_primitive_weld_under_50ms(self):
        vertices, triangles = synthetic_mesh(100_000)
        Mesher._weld_mesh_primitives(vertices, triangles)
        weld_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            unique, remapped = Mesher._weld_mesh_primitives(vertices, triangles)
            weld_times.append(time.perf_counter() - t0)
        self.assertEqual(len(unique), 100_000)
        self.assertEqual(len(remapped), 100_000 // 3)
        # Reference target is 0.05s; keep a CI-safe ceiling for slow runners.
        self.assertLess(min(weld_times), 0.10)

        t0 = time.perf_counter()
        Mesher._lib3mf_mesh_objects(unique, remapped)
        lib3mf_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        Mesher._create_3mf_mesh(vertices, triangles)
        full_s = time.perf_counter() - t0
        print(
            f"\n100k primitive weld min={min(weld_times):.4f}s "
            f"median={sorted(weld_times)[1]:.4f}s "
            f"lib3mf_objects={lib3mf_s:.4f}s full=_create_3mf_mesh={full_s:.4f}s"
        )


if __name__ == "__main__":
    unittest.main()
