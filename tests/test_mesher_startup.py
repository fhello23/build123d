"""
Tests for Lib3MF startup, wrapper lifetime, and triangulation without Mesher.

name: test_mesher_startup.py
date: August 15th 2026

desc:
    Phase 6 lazy-loads lib3mf and caches one process-wide Wrapper. OCCT
    triangulation (mesh_shape) must not pay Lib3MF import or dlopen. First
    Mesher() still pays native library load; subsequent instances share the
    wrapper and get independent models.

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

import os
import subprocess
import sys
import tempfile
import unittest

from build123d.mesher import Mesher, mesh_shape
from build123d.topology import Solid


def _subprocess_output(script: str) -> str:
    return subprocess.check_output([sys.executable, "-c", script], text=True)


def _subprocess_duration(script: str) -> float:
    return float(_subprocess_output(script).strip().splitlines()[-1])


def _min_duration(script: str, runs: int = 3) -> tuple[float, float]:
    times = [_subprocess_duration(script) for _ in range(runs)]
    return min(times), sorted(times)[len(times) // 2]


class TestLib3MFNotImportedUntilNeeded(unittest.TestCase):
    def test_import_build123d_does_not_import_lib3mf(self):
        script = (
            "import sys\n"
            "import build123d\n"
            "loaded = any(name == 'lib3mf' or name.startswith('lib3mf.') "
            "for name in sys.modules)\n"
            "print(int(loaded))\n"
        )
        self.assertEqual(_subprocess_output(script).strip(), "0")

    def test_mesh_shape_does_not_import_lib3mf(self):
        script = (
            "import sys\n"
            "from build123d.mesher import mesh_shape\n"
            "from build123d.topology import Solid\n"
            "vertices, triangles = mesh_shape(Solid.make_box(1, 1, 1))\n"
            "loaded = any(name == 'lib3mf' or name.startswith('lib3mf.') "
            "for name in sys.modules)\n"
            "print(len(vertices), len(triangles), int(loaded))\n"
        )
        parts = _subprocess_output(script).strip().split()
        self.assertGreaterEqual(int(parts[0]), 3)
        self.assertGreaterEqual(int(parts[1]), 1)
        self.assertEqual(parts[2], "0")

    def test_weld_does_not_import_lib3mf(self):
        script = (
            "import sys\n"
            "from build123d.mesher import Mesher\n"
            "Mesher._weld_mesh_primitives(\n"
            "    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],\n"
            "    [[0, 1, 2]],\n"
            ")\n"
            "loaded = any(name == 'lib3mf' or name.startswith('lib3mf.') "
            "for name in sys.modules)\n"
            "print(int(loaded))\n"
        )
        self.assertEqual(_subprocess_output(script).strip(), "0")

    def test_first_mesher_imports_lib3mf(self):
        script = (
            "import sys\n"
            "from build123d.mesher import Mesher\n"
            "Mesher()\n"
            "loaded = any(name == 'lib3mf' or name.startswith('lib3mf.') "
            "for name in sys.modules)\n"
            "print(int(loaded))\n"
        )
        self.assertEqual(_subprocess_output(script).strip(), "1")


class TestWrapperLifetime(unittest.TestCase):
    def test_meshers_share_wrapper_and_have_independent_models(self):
        first = Mesher()
        second = Mesher(unit=first.unit)
        self.assertIs(first.wrapper, second.wrapper)
        self.assertIsNot(first.model, second.model)

    def test_independent_exports(self):
        box_a = Solid.make_box(1, 1, 1)
        box_b = Solid.make_box(2, 2, 2)
        mesher_a = Mesher()
        mesher_b = Mesher()
        mesher_a.add_shape(box_a)
        mesher_b.add_shape(box_b)
        with tempfile.TemporaryDirectory() as tmp:
            path_a = os.path.join(tmp, "a.3mf")
            path_b = os.path.join(tmp, "b.3mf")
            mesher_a.write(path_a)
            mesher_b.write(path_b)
            imported_a = Mesher().read(path_a)
            imported_b = Mesher().read(path_b)
        self.assertAlmostEqual(imported_a[0].volume, 1.0, places=1)
        self.assertAlmostEqual(imported_b[0].volume, 8.0, places=1)

    def test_mesh_shape_matches_mesher_static(self):
        public = mesh_shape(Solid.make_box(1, 1, 1), 0.001, 0.1)
        wrapped = Mesher._mesh_shape(Solid.make_box(1, 1, 1), 0.001, 0.1)
        self.assertEqual(len(public[0]), len(wrapped[0]))
        self.assertEqual(len(public[1]), len(wrapped[1]))


class TestStartupTiming(unittest.TestCase):
    def test_report_cold_start_and_first_export(self):
        import_s, _ = _min_duration(
            "import time\n"
            "start = time.perf_counter()\n"
            "import build123d\n"
            "print(time.perf_counter() - start)\n"
        )
        lib3mf_import_s, _ = _min_duration(
            "import time\n"
            "start = time.perf_counter()\n"
            "from lib3mf import Lib3MF\n"
            "print(time.perf_counter() - start)\n"
        )
        first_mesher_s, _ = _min_duration(
            "import time\n"
            "from build123d.mesher import Mesher\n"
            "start = time.perf_counter()\n"
            "Mesher()\n"
            "print(time.perf_counter() - start)\n"
        )
        subsequent_s, _ = _min_duration(
            "import time\n"
            "from build123d.mesher import Mesher\n"
            "Mesher()\n"
            "start = time.perf_counter()\n"
            "Mesher()\n"
            "print(time.perf_counter() - start)\n"
        )
        mesh_shape_s, _ = _min_duration(
            "import time\n"
            "from build123d.mesher import mesh_shape\n"
            "from build123d.topology import Solid\n"
            "box = Solid.make_box(1, 1, 1)\n"
            "start = time.perf_counter()\n"
            "mesh_shape(box)\n"
            "print(time.perf_counter() - start)\n"
        )
        first_export_s, _ = _min_duration(
            "import time, os, tempfile\n"
            "from build123d.mesher import Mesher\n"
            "from build123d.topology import Solid\n"
            "start = time.perf_counter()\n"
            "mesher = Mesher()\n"
            "mesher.add_shape(Solid.make_box(1, 1, 1))\n"
            "with tempfile.TemporaryDirectory() as tmp:\n"
            "    mesher.write(os.path.join(tmp, 'a.3mf'))\n"
            "print(time.perf_counter() - start)\n"
        )
        end_to_end_s, _ = _min_duration(
            "import time, os, tempfile\n"
            "start = time.perf_counter()\n"
            "from build123d.mesher import Mesher\n"
            "from build123d.topology import Solid\n"
            "mesher = Mesher()\n"
            "mesher.add_shape(Solid.make_box(1, 1, 1))\n"
            "with tempfile.TemporaryDirectory() as tmp:\n"
            "    mesher.write(os.path.join(tmp, 'a.3mf'))\n"
            "print(time.perf_counter() - start)\n"
        )
        # Subsequent construction must stay cheap once the wrapper exists.
        self.assertLess(subsequent_s, 0.05)
        print(
            "\nPhase 6 startup "
            f"import_build123d={import_s:.4f}s "
            f"import_lib3mf={lib3mf_import_s:.4f}s "
            f"first_Mesher={first_mesher_s:.4f}s "
            f"subsequent_Mesher={subsequent_s:.4f}s "
            f"mesh_shape={mesh_shape_s:.4f}s "
            f"first_export={first_export_s:.4f}s "
            f"end_to_end_import_and_export={end_to_end_s:.4f}s"
        )


if __name__ == "__main__":
    unittest.main()
