"""X1c pilot: legacy RL generator vs ReflectoFit forward on the RL default distribution.

Speed per batch of 64 (q 2000) and accuracy of the resolution integral against a 4001-node
reference, in units of the Poisson noise at the RL counts range. Pilot only: no quiet window.
Heavy: the default resolution rule on real-SLD edges took about 17 GB of RAM (ReflectoFit#6).
"""

import math
import sys
import time

import numpy as np
import torch
from reflectofit import Instrument, forward

from reflectolearn.config import Config
from reflectolearn.data import XRRAugmentor, generate_synthetic_batch
from reflectolearn.data.augmentations import XRRAugmentor as Aug
from reflectolearn.physics import AbelesMatrix

dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
cfg = Config(device=dev)
q = cfg.q_grid.to(dev)
B = cfg.training.batch_size
wl = cfg.simulation.wavelength
aug = cfg.augmentation


def timed(fn, reps=3):
    fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


torch.manual_seed(0)
augmentor = XRRAugmentor(cfg, q)
print(f"device {dev}, B {B}, Q {q.numel()}")
print(
    f"legacy generate_synthetic_batch: {timed(lambda: generate_synthetic_batch(cfg, B, augmentor)) * 1e3:.1f} ms"
)

# one RL-distributed batch of stacks, as the generator draws them (oxide on top)
g = torch.Generator().manual_seed(1)


def u(lo, hi, n=B):
    return torch.rand(n, generator=g, dtype=torch.float64) * (hi - lo) + lo


pr, sub = cfg.param_ranges, cfg.substrate
thk, rgh, sld = u(*pr.thickness), u(*pr.roughness), u(*pr.sld)
s_sld, s_rgh = u(*sub.sld), u(*sub.roughness)
has = (torch.rand(B, generator=g, dtype=torch.float64) < aug.oxide_prob).double()
o_thk, o_sld, o_rgh = (
    u(*aug.oxide_thickness) * has,
    torch.where(has > 0, u(*aug.oxide_sld), sld),
    u(*aug.oxide_roughness),
)
z = torch.zeros(B, dtype=torch.float64)
top = torch.where(has > 0, o_rgh, rgh)
mid = torch.where(has > 0, rgh, z)
packed = (
    torch.stack([z, o_sld, sld, s_sld], 1).to(torch.complex128),
    torch.stack([z, o_thk, thk, z], 1),
    torch.stack([z, top, mid, s_rgh], 1),
)
packed_np = tuple(a.numpy() for a in packed)
sigma = u(*aug.res_sigma_range).numpy()
offset = u(*aug.tth_offset).numpy()
ka2 = u(0.0, aug.kalpha2_ratio).numpy()
inst = Instrument(
    wavelength=wl,
    kalpha2_wavelength=aug.kalpha2_wavelength,
    tth_offset=offset,
    kalpha2_ratio=ka2,
    sigma_q=sigma,
)
q64 = q.double()

for nodes in (None, 27, 41, 61, 101):
    i = inst.replace(nodes=nodes)
    t_t = timed(
        lambda i=i: forward(q64, tuple(a.to(dev) for a in packed), i, backend="torch"), reps=1
    )
    t_n = timed(lambda i=i: forward(q64.cpu().numpy(), packed_np, i), reps=1)
    print(
        f"reflectofit nodes={nodes}: torch {dev} {t_t * 1e3:.0f} ms, numba {t_n * 1e3:.0f} ms per batch"
    )

# accuracy on 16 curves vs a 4001-node reference
k = 16
sub_p = tuple(a[:k] for a in packed_np)
sub_i = Instrument(
    wavelength=wl,
    kalpha2_wavelength=aug.kalpha2_wavelength,
    tth_offset=offset[:k],
    kalpha2_ratio=ka2[:k],
    sigma_q=sigma[:k],
)
qn = q64.cpu().numpy()
ref = forward(qn, sub_p, sub_i.replace(nodes=4001))
bare = forward(qn, sub_p, sub_i.replace(sigma_q=0.0))


def legacy_smear(r):
    out = np.empty_like(r)
    for j in range(k):
        out[j] = Aug._smear(torch.as_tensor(qn), torch.as_tensor(r[j : j + 1]), float(sigma[j]))[
            0
        ].numpy()
    return out


def report(name, got):
    rel = np.abs(got / ref - 1)
    for s in (1e5, 1e8):
        lam = s * ref
        z = np.abs(got - ref) * s / np.sqrt(np.maximum(lam, 1.0))
        print(
            f"  {name}: S={s:.0e} max |dR|/noise {z.max():.2e}, frac > 0.1 noise {np.mean(z > 0.1):.3f}",
            end="",
        )
        print(f"; rel max {rel.max():.1e} median {np.median(rel):.1e}" if s == 1e5 else "")


report("legacy pixel smear (f64 kernel)", legacy_smear(bare))
for nodes in (None, 27, 41, 61, 101):
    report(f"reflectofit nodes={nodes}", forward(qn, sub_p, sub_i.replace(nodes=nodes)))

# the shipped c64 kernel without resolution, against f64, same stacks and offset/Ka2
sin_t = np.clip(qn * wl / (4 * math.pi), 0, 1)
r32 = (
    AbelesMatrix()(
        torch.as_tensor(qn).float().expand(k, -1),
        packed[1][:k, :3].float(),
        packed[2][:k, 1:].float(),
        packed[0][:k].real.float(),
    )
    .double()
    .numpy()
)
r64 = forward(qn, sub_p)
rel = np.abs(r32 / r64 - 1)
print(
    f"shipped c64 kernel vs f64 (no instrument): rel max {rel.max():.1e} median {np.median(rel):.1e}"
)
