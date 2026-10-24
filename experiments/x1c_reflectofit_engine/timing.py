"""X1c (2): generator time per batch of 64 on the GPU, legacy vs reflectofit engine; engine metadata."""

import json
import time

import torch

from reflectolearn.config import Config
from reflectolearn.data import XRRAugmentor, generate_synthetic_batch
from reflectolearn.data.generator import engine_metadata


def per_batch(cfg, reps=10):
    aug = XRRAugmentor(cfg, cfg.q_grid)
    generate_synthetic_batch(cfg, 64, aug)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        generate_synthetic_batch(cfg, 64, aug)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


torch.manual_seed(5)
for engine, nodes in (("legacy", None), ("reflectofit", 27), ("reflectofit", 61)):
    cfg = Config(device="cuda")
    cfg.simulation.engine = engine
    if nodes:
        cfg.simulation.resolution_nodes = nodes
    print(f"{engine} nodes={nodes}: {per_batch(cfg):.1f} ms per batch of 64")
    print(f"  peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    torch.cuda.reset_peak_memory_stats()

cfg = Config(device="cuda")
cfg.simulation.engine = "reflectofit"
print(json.dumps(engine_metadata(cfg, n=2, seed=7)))
