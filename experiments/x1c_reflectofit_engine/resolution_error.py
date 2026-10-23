import json

from reflectolearn.config import Config
from reflectolearn.data.generator import engine_metadata

for nodes in (27, 61, 101):
    cfg = Config(device="cpu")
    cfg.simulation.engine = "reflectofit"
    cfg.simulation.resolution_nodes = nodes
    print(json.dumps(engine_metadata(cfg, n=2, seed=7)))
