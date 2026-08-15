# build123d Performance Plan

> Based on cProfile/wall-clock profiling of the benchmark suite on 2026-08-15
> (Python 3.12, macOS arm64, build123d 0.11.2.dev223+gb8f855108).
> Raw findings are summarized in the appendix.

## Guiding principles

Profiling shows that build123d's Python layer is not the primary bottleneck in
boolean-heavy workloads:

- Most elapsed time is inside OCCT's C++ boolean algorithms
  (`BRepAlgoAPI_*.Build()`), which are already invoked with
  `SetRunParallel(True)`.
- OCCT's boolean algorithm is superlinear in accumulated shape complexity:
  284 sequential fuses take 1.88s, while the same 284 shapes in one batched
  fuse take 0.014s (134x faster).
- The largest opportunity is therefore changing how often and with which
  operands OCCT is called, rather than rewriting Python geometry code.

The implementation order is:

1. Establish performance and semantic baselines.
2. Remove low-risk redundant wrapper work.
3. Add an internal ordered boolean-batching primitive.
4. Introduce opt-in Builder batching with well-defined flush barriers.
5. Decide whether automatic algebra-mode laziness is worth its semantic cost.
6. Vectorize mesh assembly with NumPy.
7. Improve lib3mf startup behavior based on separately measured costs.
8. Consider a small Rust accelerator only if a meaningful hotspot remains.

Each phase should be delivered separately so its performance effect and
correctness can be evaluated independently.

---

## Phase 0 — Measurement and correctness baseline

Performance changes must be evaluated against both elapsed time and build123d's
observable behavior.

### Tasks

1. Extend `tests/test_benchmarks.py` with:
   - The 284-shape sequential-fuse workload.
   - The equivalent batched fuse/cut.
   - A 2000-face grid workload.
   - Selector-heavy Builder workloads.
   - Mixed-mode Builder workloads.
   - Mesh assembly benchmarks that separate primitive-data processing from
     Lib3MF Python-object construction.
2. Change `benchmark.yml` so it compares against an actual stored baseline and
   can flag meaningful regressions. The current workflow records a new result
   and prints it, but does not enforce a comparison with an earlier revision.
3. Add semantic-equivalence tests covering:
   - Volume, area, bounding box, validity, and manifold state.
   - Shape/face/edge/solid counts where these are part of expected behavior.
   - `Select.LAST` and `Select.NEW` at every explicit evaluation barrier.
   - Ordered combinations of `ADD`, `SUBTRACT`, `INTERSECT`, and `REPLACE`.
   - Labels, colors, materials, joints, and assembly relationships.
   - Error type and timing where practical.
4. Record topology identity/order only where build123d promises it. Batched and
   sequential OCCT operations may produce geometrically equivalent results with
   different face splitting or traversal order.
5. Document the profiling method precisely. cProfile may fail to attribute
   synchronous native time to the specific pybind11 call, or may charge it to
   the nearest Python caller; it should not be used by itself to separate
   Python and OCCT cost. Use wall-clock deltas around kernel calls for that
   separation.

### Verification

- Benchmarks run reproducibly on the existing CI platform matrix.
- The semantic corpus passes on the unmodified implementation.
- The baseline results are stored in a form later PRs can compare against.

### Profiling method

cProfile is useful for counting Python-level calls (`shapetype()`, `downcast()`,
`hash()`, wrapper `__init__`) and for finding Python hot spots. It is not a
reliable way to separate Python cost from OCCT cost:

- Synchronous work inside pybind11/OCCT often appears under the nearest Python
  caller, a C-extension dummy, or not at all.
- `BRepAlgoAPI_*.Build()` can dominate elapsed time while barely showing up in
  the cProfile `tottime`/`cumtime` tables.
- Enabling or disabling the profiler changes GC and wrapper traffic, so profile
  totals should not be compared directly with wall-clock batching claims.

To attribute kernel time, use `time.perf_counter()` (or an equivalent monotonic
clock) immediately around the native call, for example:

```python
start = time.perf_counter()
operation.Build()
kernel_s = time.perf_counter() - start
```

Record the Python setup separately (shape construction, wrapper casts, selector
queries). A complete measurement of a boolean workload therefore has three
numbers: wall time for the full operation, wall time inside `Build()`, and
cProfile call counts for the Python wrappers.

Import and Lib3MF startup costs must be measured in a fresh interpreter. A
subprocess that prints a `perf_counter` delta around `import build123d` or the
first `Mesher()` construction avoids mixing those one-time `dlopen` costs with
later geometry work.

CI regression gates use stored same-run ratios (sequential vs batched on the
same runner) and a generous comparison against the previous successful
benchmark artifact for that OS. Absolute thresholds from a reference laptop
are recorded for local work; they are not CI fail criteria.

The committed ratio file is `tests/benchmarks/ratio_baseline.json`. Each
benchmark job writes `benchmark-current.json` and compares it with:

- the stored ratios, and
- the previous successful `benchmark-json-<os>` artifact from the default
  branch, when one exists.

## Phase 1 — Remove redundant wrapper casting

### Problem

The 284-iteration workload spends about 0.47s in Python wrapper machinery:
162k `TopoDS.Face` casts, 82k `shapetype()` calls, 41k `downcast()` calls,
plus hash/dict/GC churn.

The current cast path can inspect and downcast the same object multiple times:

1. A `cast()` method calls `shapetype()`.
2. `cast()` calls `downcast()`, which calls `shapetype()` again.
3. The resulting build123d wrapper calls `Shape.__init__()`, which may downcast
   the already-specialized TopoDS object again.

### Tasks

1. Determine the shape type once per cast operation.
2. Downcast each TopoDS object at most once before constructing its build123d
   wrapper.
3. Allow constructors to accept already-specialized TopoDS objects without
   repeating the downcast.
4. Move constructor/type lookup tables out of hot per-call paths.
5. Re-check `copy_attributes_to()` and `__deepcopy__()` frequency after cast
   cleanup, then optimize only the remaining measured work.
6. Audit anytree `NodeMixin` interactions, but preserve parent/child and
   assembly behavior.

Broad `lru_cache` use is not the first approach:

- Downcast dispatch already uses a dictionary lookup.
- TopoDS handles are not generally immutable; locations/orientations can
  change, and compound contents can be modified.
- Caching OCP wrapper objects could retain large native object graphs.
- Caching `topods_dim()` for compounds requires explicit invalidation.

Likewise, `__slots__` is deferred unless measurement shows a worthwhile gain;
it may provide little benefit while `NodeMixin` or subclasses retain instance
dictionaries.

### Verification

- Measure calls per extracted topology object as well as elapsed time.
- Target at least a 50% reduction in redundant `shapetype()`/`downcast()` calls.
- The full test, lint, and type-check suites pass unchanged.

## Phase 2 — Internal ordered boolean-batching primitive

### Goal

Create a small internal abstraction that can execute bulk operations without
yet changing Builder or algebra evaluation semantics:

```text
base + [A, B, C]  -> one fuse and one clean
base - [A, B, C]  -> one cut and one clean
```

### Tasks

1. Define an internal representation for a base shape, an ordered mode, and a
   list of operands.
2. Reuse the existing variadic OCCT boolean APIs rather than fusing tools in
   advance. In particular, `base.cut(*tools)` should receive all subtraction
   tools directly.
3. Ensure cleaning happens once per executed batch, while preserving
   `SkipClean` and explicit `clean=False` behavior.
4. Preserve attribute propagation and result wrapping.
5. Test empty, single-operand, disjoint, overlapping, invalid, and mixed
   dimensional inputs.

### Verification

- The 284-shape bulk operation performs one kernel boolean call.
- Target no more than 0.10s for the profiled 284-shape case on the reference
  machine.
- Output passes the Phase 0 semantic-equivalence corpus.

## Phase 3 — Opt-in Builder batching

### Problem

`BuildPart` and `BuildSketch` currently integrate each published object
immediately. Repeated additions therefore produce an increasingly expensive
sequence of booleans and cleans.

### Initial API

```python
with BuildPart(defer_booleans=True) as part:
    ...
```

The eager path remains the default during this phase.

### Ordering rule

Only contiguous compatible operations may be batched:

```text
ADD A, ADD B, ADD C             -> one fuse(A, B, C)
SUBTRACT A, SUBTRACT B          -> one cut(A, B)
ADD A, SUBTRACT B, ADD C        -> three ordered segments
```

Operations must never be regrouped globally by mode. For example,
`ADD -> SUBTRACT -> ADD` is not equivalent to `ADD all -> SUBTRACT all` when
the final added shape overlaps a subtraction tool.

### Flush barriers

The pending batch must be evaluated before:

- A change to an incompatible mode.
- `INTERSECT` or `REPLACE`.
- Any read of current topology, geometry, mass properties, or bounds.
- `Select.LAST`, `Select.NEW`, or operations that depend on their results.
- Fillet, chamfer, extrude, project, offset, split, or any operation consuming
  the current Builder result.
- Publication to a parent Builder.
- Builder context exit.
- Any explicit `.flush()` call.

### Additional semantic requirements

- `lasts`, `obj_before`, `to_combine`, and `new_edges()` must describe the
  same logical operation visible at each barrier.
- Buffered operands must have defined capture semantics. Either capture their
  current topology safely when queued or flush before a mutation could change
  them.
- Exceptions should occur at a predictable barrier and identify the logical
  operation that queued the invalid input.
- Cleaning and attribute propagation must match the logical ordered segments.

### Rollout

1. Implement and validate `BuildSketch`, using repeated planar shapes as the
   first stress case.
2. Extend the same ordered queue and barriers to `BuildPart`.
3. Run the documentation examples in addition to the unit suite because many
   examples immediately use `Select.LAST`/`Select.NEW` for a later operation.
4. Consider making deferral the default only after the opt-in implementation
   has broad compatibility evidence.

### Verification

- The 284-shape Builder workload meets the 0.10s reference target where the
  operation sequence permits one batch.
- Mixed-mode and selector-heavy workloads preserve eager behavior at barriers.
- The full test, documentation-example, lint, and type-check suites pass.

## Phase 4 — Algebra-mode evaluation decision

### Problem

A loop such as `holes += loc * rectangle` performs eager boolean work today.
Automatic lazy algebra evaluation could accelerate it, but it changes more
semantics than Builder-local batching.

In particular, `Part.__iadd__()` and `Sketch.__iadd__()` currently return the
result of addition rather than mutating a persistent lazy container. Changing
this can affect aliases, error timing, mutation behavior, and object identity.

### Tasks

1. Evaluate the need after Builder batching and documentation improvements.
   Algebra mode already supports fast vectorized forms such as:

   ```python
   result = Sketch() + shapes
   result = base - subtraction_tools
   ```

2. Compare three options:
   - Keep eager algebra semantics and improve guidance/vectorized APIs.
   - Add an explicit `ShapeBatch`/batch context with clear evaluation rules.
   - Add lazy `Part`/`Sketch` evaluation with comprehensive flush barriers.
3. If an operand batch is consumed by subtraction, pass its individual tools
   directly to the cut when possible rather than first materializing a fused
   shape and then cutting.
4. Do not promote automatic laziness solely because the existing test suite
   passes; add aliasing, mutation, exception-timing, and attribute tests first.

### Decision criterion

Proceed with automatic lazy algebra only if it delivers a substantial benefit
over an explicit batch API without creating surprising object semantics.

## Phase 5 — NumPy mesh assembly

### Problem

`Mesher._create_3mf_mesh()` spends about 0.26s per 100k input vertices in a
pure-Python vertex-welding and triangle-remapping loop dominated by `round()`.

### Tasks

1. Use NumPy, which is already a required dependency, for vectorized rounding,
   duplicate detection, inverse-index construction, triangle remapping, and
   degenerate-triangle filtering.
2. Preserve first-seen vertex order. A direct `numpy.unique(axis=0)` sorts its
   output, so it must be reordered or replaced with a stable method.
3. Test values around rounding boundaries to preserve the intended tolerance
   behavior.
4. Use dense arrays/lists instead of an integer-keyed dictionary for the
   original-index to welded-index mapping.
5. Measure separately:
   - Primitive vertex welding and triangle remapping.
   - Construction of `Lib3MF.Position` and `Lib3MF.Triangle` Python objects.
   - `SetGeometry()` and complete export time.

The Lib3MF object-construction loop may become the dominant cost after NumPy;
the complete pipeline, rather than only the vectorized portion, determines the
real benefit.

### Verification

- Target less than 0.05s for the 100k-point primitive assembly step on the
  reference machine.
- Compare exact welded vertex order and triangle indices with fixtures.
- Verify triangle winding and degenerate-triangle removal.
- Verify semantic 3MF/STL round trips. Full-file byte equality is not required
  because container metadata and library versions can change otherwise
  equivalent output bytes.

## Phase 6 — Lib3MF startup and lifetime

The current 0.73s measurement is a one-time `dlopen` cost at `Mesher()`
construction. Startup work must distinguish unavoidable first-use cost from
cost that can actually be removed.

### Tasks

1. Measure independently:
   - `import build123d`.
   - Importing `build123d.mesher` and the `lib3mf` Python package.
   - The first `Mesher()` construction.
   - Subsequent `Mesher()` constructions.
   - The first real read/write operation.
2. Consider lazy importing lib3mf to reduce general build123d import time.
   This requires moving the module-level enum maps and annotations that
   currently depend on `Lib3MF` at import time.
3. Consider caching a process-wide Lib3MF wrapper if it safely reduces repeated
   construction cost while keeping each `Mesher` model independent.
4. Consider separating generic OCCT triangulation from the 3MF-specific class;
   `Mesher._mesh_shape()` is useful without constructing a Lib3MF model.

### Verification

- Report both cold-start improvement and end-to-end first-operation time.
- Do not claim a gain if lazy initialization only moves the same cost from
  construction to the first method call.

## Phase 7 — Rust accelerator go/no-go

Rust is optional and should be evaluated only after the NumPy implementation is
profiled end to end.

### Suitable scope

A small extension such as `build123d._accel` may be appropriate for a coarse,
data-oriented operation:

```python
unique_vertices, remapped_triangles = _accel.weld_vertices(
    vertices, triangles, digits
)
```

The extension should accept and return primitive buffers or NumPy arrays in a
single call. It should not cross the Python/native boundary once per vertex,
triangle, or OCP object.

### Go criteria

Proceed only if all of the following are true:

1. A meaningful hotspot remains after NumPy and Lib3MF object construction are
   measured separately.
2. The native portion is expected to improve the optimized path by at least
   2-3x and save at least 50-100ms on a realistic export.
3. The Python fallback remains fully functional and tested.
4. Packaging and CI cover the supported Python 3.10-3.14 and
   macOS/Linux/Windows architecture matrix.
5. The wheel/source-install strategy does not make Rust mandatory for users
   who do not need the accelerator. A separately distributed optional
   accelerator may be the safest initial experiment.

### Out of scope for Rust

- `geometry.py` Vector/Matrix/Location math without new profiling evidence.
- Shape wrappers, selectors, and anytree bookkeeping.
- OCCT boolean dispatch or replacing OCCT.
- Code that would need frequent interaction with OCP or Lib3MF Python objects.

---

## Non-goals

- Replacing OCCT: no Rust alternative currently provides equivalent booleans,
  fillets, topology, and STEP I/O for build123d's requirements.
- Porting the library wholesale to Rust: the measured hot geometry operations
  are already in OCCT's C++ implementation.
- Merely enabling OCCT parallel flags: booleans/sections already use
  `SetRunParallel(True)` and meshing uses `isInParallel=True`. Individual
  workloads may still benchmark the parallel setting, but it is not a missing
  global switch.

## Success metrics

| Workload | Baseline | Target |
|---|---:|---:|
| 284 compatible operations with batching enabled | 2.50s | <= 0.10s |
| Redundant `shapetype()`/`downcast()` calls per extracted shape | current path repeats work | >= 50% reduction |
| Mesher primitive assembly, 100k vertices | 0.26s | <= 0.05s |
| General import/startup | establish in Phase 0/6 | no shifted-cost claims |
| Existing behavior | current suite and examples pass | no unintended semantic changes |

Performance thresholds are reference-machine goals. CI regression gates should
use stored comparisons, reasonable tolerances, or same-run ratios rather than
assuming identical absolute timing on heterogeneous runners.

## Appendix — profiling evidence (2026-08-15)

Environment: Python 3.12 venv, `pip install -e .` at commit `b8f85510`, macOS
arm64. Method: cProfile for Python call counts, plus `time.perf_counter()`
wall-clock deltas around kernel calls; cProfile was not used to split Python
from OCCT time. Doc benchmark models executed from `docs/assets/ttt/` through
`tests/test_benchmarks.py`.

Key numbers:

- Doc models are fast: ttt-ppp0101 0.19s wall / 0.04s attributed Python;
  sm_hanger 0.60s / 0.09s; Buffer_Stand 0.20s / 0.05s. Top Python entries are
  wrapper machinery (`shapetype`, `downcast`, `hash`, `isinstance`, and
  `Vector.__init__`) at millisecond scale.
- Sequential fuse of 284 rectangles: 1.88s wall. Manual timing around the
  kernel call attributes nearly all of this case to OCCT `Build()`. One batched
  fuse of the same shapes takes 0.014s.
- Full 284-iteration `holes += loc * r` pattern: 2.50s wall; about 0.47s in
  Python wrapper work (1.5M calls) and 558 generation-0 GC collections.
  `gc.disable()` makes it slower (9.4s, memory pressure); keeping all
  intermediates alive takes 16.9s.
- Mesher: `_create_3mf_mesh()` takes 0.26s per 100k vertices, dominated by
  per-vertex rounding, plus a one-time 0.73s Lib3MF `dlopen` during the first
  `Mesher()` construction.
- Parallel flags are already enabled: booleans/sections use
  `SetRunParallel(True)` and meshing uses `isInParallel=True`.
