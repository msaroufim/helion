# Modal-based Distributed Autotuner for Helion

This fork adds `ModalSearch`, a distributed autotuner that dispatches kernel benchmarking to parallel Modal GPU workers. It enables autotuning from machines **without a local GPU** (e.g., a Mac laptop) and parallelizes what is otherwise serial benchmarking.

**Upstream:** [pytorch/helion](https://github.com/pytorch/helion)

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/msaroufim/helion.git
cd helion
uv venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pip install modal
```

### 2. Set up Modal

```bash
modal token set  # authenticate with your Modal account
```

### 3. Deploy the autotuner app (one time)

This pre-builds the container image (CUDA 12.6 + torch + triton + helion) so subsequent calls don't have cold starts:

```bash
modal deploy helion/autotuner/_modal_app.py
```

### 4. Run autotuning

```bash
python benchmarks/bench_modal_matmul.py
```

This runs helion's full pattern search autotuner on a 4096x4096 fp16 matmul, dispatching ~600 configs to Modal H100 workers. Output:

```
[0s] Autotune random seed: 1942923406
[0s] Dispatching 20 configs to Modal (H100)
[8s] Initial population: ok=20 min=1.4953 ...
[9s] Dispatching 65 configs to Modal (H100)
[22s] Generation 1: improved 1.4953ms -> 0.7259ms (51.45%)
...
[64s] Autotuning complete in 64.3s after searching 574 configs.
One can hardcode the best config and skip autotuning with:
    @helion.kernel(config=helion.Config(block_sizes=[128, 256, 64], ...), static_shapes=True)
```

Copy-paste the `@helion.kernel(config=...)` decorator onto your kernel to skip autotuning in production.

### 5. Use with any helion kernel

Set the environment variable to use Modal for any kernel:

```bash
HELION_AUTOTUNER=ModalSearch python my_kernel.py
```

Or programmatically:

```python
from helion.autotuner.modal_search import modal_autotune

best_config = modal_autotune(my_kernel_fn, *args, gpu_type="H100", n_configs=20)
```

## How It Works

### Architecture

```
Mac / CPU machine                          Modal Cloud
┌─────────────────────┐                    ┌──────────────────┐
│  helion autotuner   │                    │  H100 Worker 1   │
│  (ModalSearch)      │  starmap (N calls) │  H100 Worker 2   │
│                     │ ─────────────────> │  ...              │
│  1. Generate configs│  triton code only  │  H100 Worker 10  │
│  2. Generate triton │  (~few KB each)    │                  │
│     code per config │                    │  Each worker:    │
│  3. Dispatch to     │ <───────────────── │  - Reads args    │
│     Modal workers   │  timing results    │    from Dict     │
│  4. Collect results │                    │  - JIT compiles  │
│  5. Next generation │                    │  - Benchmarks    │
└─────────────────────┘                    └──────────────────┘
         │
         │ Upload once (67MB)
         ▼
   ┌─────────────┐
   │  Modal Dict  │  Shared args store
   └─────────────┘
```

### Key design decisions

1. **Args uploaded once to Modal Dict** — Serialized kernel arguments (tensors) are uploaded once to a shared `modal.Dict`. Each starmap call only sends the triton code string (~few KB) + a dict key. Workers fetch args from the Dict on first use and cache them per container. This avoids sending 67MB per call.

2. **Overrides `parallel_benchmark`** — `ModalSearch` subclasses `PopulationBasedSearch` and overrides `parallel_benchmark`, `rebenchmark`, and `_compute_baseline`. This means any search algorithm (PatternSearch, LFBO, DE) that calls `parallel_benchmark` automatically dispatches to Modal.

3. **Deployed vs ephemeral modes** — If you run `modal deploy`, the dispatcher uses `Function.from_name()` to call the pre-registered function (warm containers). If not deployed, it falls back to an ephemeral `app.run()` context with cold start.

4. **Worker uses tempfile + importlib** — Triton's `@jit` requires kernel code in a real `.py` file (not `exec()`'d). The worker writes triton code to a tempfile and imports via `importlib.util.spec_from_file_location`.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `HELION_AUTOTUNER=ModalSearch` | `LFBOPatternSearch` | Use Modal for autotuning |
| `HELION_AUTOTUNE_MODAL_GPU` | `H100` | GPU type for Modal workers |
| `HELION_AUTOTUNE_MODAL_MAX_CONCURRENT` | `50` | Max parallel workers |

## Files

| File | Description |
|------|-------------|
| `helion/autotuner/modal_search.py` | `ModalSearch` algorithm + `ModalBenchmarkDispatcher` |
| `helion/autotuner/_modal_worker.py` | GPU worker function that runs on Modal |
| `helion/autotuner/_modal_app.py` | Deployable Modal app for warm containers |
| `helion/autotuner/__init__.py` | Registry entry for `ModalSearch` |
| `helion/runtime/settings.py` | Settings for GPU type and concurrency |
| `benchmarks/bench_modal_matmul.py` | Example: autotune matmul from Mac |
| `test_modal_search.py` | Offline unit tests |
| `test_modal_e2e.py` | End-to-end Modal dispatch test |

## Performance

4096x4096 fp16 matmul autotuning on Modal H100s from a Mac:

| Metric | Value |
|--------|-------|
| Configs searched | 574 |
| Total wall time | 64s |
| Per-generation time | 3-5s |
| Best perf | 0.187ms (735 TFLOPS) |
| Estimated cost | ~$0.80 |

## Limitations

- The traceback from `compile_config` prints on Mac since triton isn't available locally. The autotuner still prints the decorator correctly. The process exits cleanly.
- Cold container startup takes ~30s on first call. Subsequent calls within ~5 minutes reuse warm containers.
- Modal Dict entries expire after 7 days of inactivity.
