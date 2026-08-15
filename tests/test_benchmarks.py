import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from build123d.mesher import Mesher

from tests.performance_workloads import (
    EXPECTED_FACE_GRID_COUNT,
    EXPECTED_HOLE_COUNT,
    batched_algebra_fuse,
    batched_cut,
    batched_fuse,
    deferred_hole_sketch,
    eager_hole_sketch,
    face_grid_sketch,
    hole_rectangles,
    mixed_mode_builder,
    plate_with_holes_base,
    selector_heavy_builder,
    sequential_algebra_fuse,
    sequential_cut,
    sequential_fuse,
    shape_batch_algebra_fuse,
    shape_batch_cut,
    synthetic_mesh,
)

mock_module = Mock()
mock_module.show = Mock()
mock_module.show_object = Mock()
mock_module.show_all = Mock()
sys.modules["ocp_vscode"] = mock_module

_ = pytest.importorskip("pytest_benchmark")


def _read_docs_ttt_code(name):
    checkout_dir = Path(__file__).parent.parent
    ttt_dir = checkout_dir / "docs/assets/ttt"
    name = "ttt-" + name + ".py"
    with open(ttt_dir / name, "r", encoding="utf-8") as f:
        return f.read()


def test_ppp_0101(benchmark):
    def model():
        exec(_read_docs_ttt_code("ppp0101"))

    benchmark(model)


def test_ppp_0102(benchmark):
    def model():
        exec(_read_docs_ttt_code("ppp0102"))

    benchmark(model)


def test_ppp_0103(benchmark):
    def model():
        exec(_read_docs_ttt_code("ppp0103"))

    benchmark(model)


def test_ppp_0104(benchmark):
    def model():
        exec(_read_docs_ttt_code("ppp0104"))

    benchmark(model)


def test_ppp_0105(benchmark):
    def model():
        exec(_read_docs_ttt_code("ppp0105"))

    benchmark(model)


def test_ppp_0106(benchmark):
    def model():
        exec(_read_docs_ttt_code("ppp0106"))

    benchmark(model)


def test_ppp_0107(benchmark):
    def model():
        exec(_read_docs_ttt_code("ppp0107"))

    benchmark(model)


def test_ppp_0108(benchmark):
    def model():
        exec(_read_docs_ttt_code("ppp0108"))

    benchmark(model)


def test_ppp_0109(benchmark):
    def model():
        exec(_read_docs_ttt_code("ppp0109"))

    benchmark(model)


def test_ppp_0110(benchmark):
    def model():
        exec(_read_docs_ttt_code("ppp0110"))

    benchmark(model)


def test_ttt_23_02_02(benchmark):
    def model():
        exec(_read_docs_ttt_code("23-02-02-sm_hanger"))

    benchmark(model)


def test_ttt_23_T_24(benchmark):
    def model():
        exec(_read_docs_ttt_code("23-t-24-curved_support"))

    benchmark(model)


def test_ttt_24_SPO_06(benchmark):
    def model():
        exec(_read_docs_ttt_code("24-SPO-06-Buffer_Stand"))

    benchmark(model)


@pytest.mark.benchmark(group="fuse_284")
def test_sequential_fuse_284(benchmark):
    shapes = hole_rectangles()
    assert len(shapes) == EXPECTED_HOLE_COUNT
    result = benchmark(sequential_fuse, shapes)
    assert result.is_valid


@pytest.mark.benchmark(group="fuse_284")
def test_batched_fuse_284(benchmark):
    shapes = hole_rectangles()
    result = benchmark(batched_fuse, shapes)
    assert result.is_valid


@pytest.mark.benchmark(group="fuse_284")
def test_sequential_algebra_fuse_284(benchmark):
    shapes = hole_rectangles()
    result = benchmark(sequential_algebra_fuse, shapes)
    assert result.is_valid


@pytest.mark.benchmark(group="fuse_284")
def test_batched_algebra_fuse_284(benchmark):
    shapes = hole_rectangles()
    result = benchmark(batched_algebra_fuse, shapes)
    assert result.is_valid


@pytest.mark.benchmark(group="fuse_284")
def test_shape_batch_algebra_fuse_284(benchmark):
    shapes = hole_rectangles()
    result = benchmark(shape_batch_algebra_fuse, shapes)
    assert result.is_valid


@pytest.mark.benchmark(group="cut_284")
def test_sequential_cut_284(benchmark):
    tools = hole_rectangles()
    result = benchmark(lambda: sequential_cut(plate_with_holes_base(), tools))
    assert result.is_valid


@pytest.mark.benchmark(group="cut_284")
def test_batched_cut_284(benchmark):
    tools = hole_rectangles()
    result = benchmark(lambda: batched_cut(plate_with_holes_base(), tools))
    assert result.is_valid


@pytest.mark.benchmark(group="cut_284")
def test_shape_batch_cut_284(benchmark):
    tools = hole_rectangles()
    result = benchmark(lambda: shape_batch_cut(plate_with_holes_base(), tools))
    assert result.is_valid


@pytest.mark.benchmark(group="face_grid")
def test_face_grid_2000(benchmark):
    sketch = benchmark(face_grid_sketch)
    assert len(sketch.faces()) == EXPECTED_FACE_GRID_COUNT
    assert sketch.is_valid


@pytest.mark.benchmark(group="builder")
def test_selector_heavy_builder(benchmark):
    result = benchmark(selector_heavy_builder)
    assert result.is_valid


@pytest.mark.benchmark(group="builder")
def test_mixed_mode_builder(benchmark):
    result = benchmark(mixed_mode_builder)
    assert result.is_valid


@pytest.mark.benchmark(group="builder_defer")
def test_eager_hole_sketch_284(benchmark):
    result = benchmark(eager_hole_sketch)
    assert len(result.faces()) == EXPECTED_HOLE_COUNT
    assert result.is_valid


@pytest.mark.benchmark(group="builder_defer")
def test_deferred_hole_sketch_284(benchmark):
    result = benchmark(deferred_hole_sketch)
    assert len(result.faces()) == EXPECTED_HOLE_COUNT
    assert result.is_valid


def _ensure_lib3mf_loaded():
    Mesher()


@pytest.mark.parametrize("test_input", [100, 1000, 10000, 100000])
def test_mesher_benchmark(benchmark, test_input):
    # in the 100_000 case test should take on the order of 0.2 seconds
    # but usually less than 1 second
    vertices, triangles = synthetic_mesh(test_input)

    def test_create_3mf_mesh():
        mesh = Mesher._create_3mf_mesh(vertices, triangles)
        assert len(mesh[0]) == test_input
        assert len(mesh[1]) == int(test_input / 3)

    benchmark(test_create_3mf_mesh)


@pytest.mark.parametrize("test_input", [1000, 100000])
@pytest.mark.benchmark(group="mesh_assembly")
def test_mesher_primitive_weld(benchmark, test_input):
    vertices, triangles = synthetic_mesh(test_input)

    def weld():
        unique, remapped = Mesher._weld_mesh_primitives(vertices, triangles)
        assert len(unique) == test_input
        assert len(remapped) == int(test_input / 3)
        return unique, remapped

    benchmark(weld)


@pytest.mark.parametrize("test_input", [1000, 100000])
@pytest.mark.benchmark(group="mesh_assembly")
def test_mesher_lib3mf_objects(benchmark, test_input):
    _ensure_lib3mf_loaded()
    vertices, triangles = synthetic_mesh(test_input)
    unique, remapped = Mesher._weld_mesh_primitives(vertices, triangles)

    def construct():
        mesh = Mesher._lib3mf_mesh_objects(unique, remapped)
        assert len(mesh[0]) == test_input
        assert len(mesh[1]) == int(test_input / 3)

    benchmark(construct)


def _subprocess_duration(script: str) -> float:
    output = subprocess.check_output([sys.executable, "-c", script], text=True)
    return float(output.strip().splitlines()[-1])


@pytest.mark.benchmark(group="startup")
def test_startup_import_build123d(benchmark):
    script = (
        "import time\n"
        "start = time.perf_counter()\n"
        "import build123d\n"
        "print(time.perf_counter() - start)\n"
    )
    benchmark.pedantic(
        lambda: _subprocess_duration(script), rounds=3, iterations=1, warmup_rounds=0
    )


@pytest.mark.benchmark(group="startup")
def test_startup_import_lib3mf(benchmark):
    script = (
        "import time\n"
        "start = time.perf_counter()\n"
        "from lib3mf import Lib3MF\n"
        "print(time.perf_counter() - start)\n"
    )
    benchmark.pedantic(
        lambda: _subprocess_duration(script), rounds=3, iterations=1, warmup_rounds=0
    )


@pytest.mark.benchmark(group="startup")
def test_startup_first_mesher(benchmark):
    script = (
        "import time\n"
        "from build123d.mesher import Mesher\n"
        "start = time.perf_counter()\n"
        "Mesher()\n"
        "print(time.perf_counter() - start)\n"
    )
    benchmark.pedantic(
        lambda: _subprocess_duration(script), rounds=3, iterations=1, warmup_rounds=0
    )


@pytest.mark.benchmark(group="startup")
def test_startup_subsequent_mesher(benchmark):
    Mesher()
    benchmark(Mesher)
