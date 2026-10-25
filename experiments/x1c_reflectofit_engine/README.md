# Generator on ReflectoFit (X1c): measurements

Scripts behind the numbers in the X1c pull request and ReflectoFit#6. Run from the repository root
with `uv run python experiments/x1c_reflectofit_engine/<script>.py`.

| script | what | cost |
|---|---|---|
| `pilot.py [cuda\|cpu]` | ReflectoFit forward against the legacy generator on one batch of 64: time per node count, and the error of each against a 4001-node reference | about 2 minutes and up to 17 GB of RAM (the default resolution rule on real-SLD edges, ReflectoFit#6). Treat it as a heavy run. |
| `timing.py` | `generate_synthetic_batch` on the GPU, legacy against the reflectofit engine | seconds; at 61 nodes the GPU allocation exceeded a 4 GB job memory cap on Windows |
| `resolution_error.py` | `engine_metadata` at 27, 61 and 101 nodes | about 1 minute, 1.9 GB |

Results (RTX 5070, not in a quiet window):

- legacy batch 3.4 ms; reflectofit engine at 27 nodes 43 ms.
- Error of the fixed node count against ReflectoFit's per-point rule, two curves, σ_q = 1e-4 and
  6e-3: max relative 8.7e-7 / 0.26 at 27 nodes, 1.8e-7 / 0.037 at 61, 4.9e-8 / 6.3e-3 at 101; median
  below 1e-11.
- Shipped complex64 legacy kernel against float64 on the pilot batch: max relative 6.8e-2, median
  5.0e-5.
