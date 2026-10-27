"""
For each volume in VOLUME_PCT and each rattled start:
  1. build a 64-atom Si supercell, scale to the target volume, apply a small random
     volume-preserving strain, and rattle the atoms
  2. relax the atoms (fixed cell) with the foundation MLIP (cheap!)
  3. keep snapshots along the relaxation path at chosen energy fractions
Then write one pw.x input per snapshot + manifest.json
"""
import json
import os
import numpy as np
from train_common import *

_CALC = None


def build_start(vol_pct, seed):
    """Rattled, randomly strained 64-atom cell at (1 + vol_pct/100) of the reference volume."""
    rng = np.random.default_rng(seed)
    cell, pos = build_si()
    eps = rng.normal(0.0, STRAIN_STD, (3, 3))
    F = np.eye(3) + (eps + eps.T) / 2
    F /= np.linalg.det(F) ** (1.0 / 3.0)                       # strain without volume change
    s = (1.0 + vol_pct / 100.0) ** (1.0 / 3.0)                  # isotropic volume scaling
    cell = (cell * s) @ F.T
    pos = (pos * s) @ F.T + rng.normal(0.0, RATTLE_STD, pos.shape)
    return cell, pos


def relax_with_mace(cell, pos):
    """Relax atomic positions (fixed cell) with the foundation model.
    Returns a list of (energy_eV, positions) for every step, starting structure first.
    """
    global _CALC
    from ase import Atoms
    from ase.optimize import FIRE
    from mace.calculators import mace_mp

    if _CALC is None:
        _CALC = mace_mp(model=MACE_MODEL, default_dtype="float64", device=DEVICE)
    atoms = Atoms(f"Si{len(pos)}", positions=pos, cell=cell, pbc=True)
    atoms.calc = _CALC
    traj = [(float(atoms.get_potential_energy()), atoms.get_positions().copy())]
    opt = FIRE(atoms, logfile=None)
    opt.attach(lambda: traj.append((float(atoms.get_potential_energy()),
                                    atoms.get_positions().copy())), interval=1)
    opt.run(fmax=FMAX, steps=MAX_STEPS)
    return traj


def pick_snapshots(traj, fracs):
    """Indices of frames whose excess energy is closest to each fraction of the initial excess."""
    E = np.array([t[0] for t in traj])
    e_final, e_start = E.min(), E[0]
    chosen = []
    for f in fracs:
        order = np.argsort(np.abs(E - (e_final + f * (e_start - e_final))))
        idx = next((int(i) for i in order if int(i) not in chosen), int(order[0]))
        chosen.append(idx)
    return chosen


def main():
    os.makedirs("qe", exist_ok=True)
    manifest, k = [], 0
    for iv, vol in enumerate(VOLUME_PCT):
        for ir in range(RATTLES_PER_VOLUME):
            seed = SEED + 100 * iv + ir
            cell, pos0 = build_start(vol, seed)
            traj = relax_with_mace(cell, pos0)
            idxs = pick_snapshots(traj, SNAP_FRACS)
            print(f"vol {vol:+.1f}%  rattle {ir}: {len(traj)} steps, "
                  f"E {traj[0][0]:.3f} -> {traj[-1][0]:.3f} eV, snapshots at steps {idxs}")
            for step in idxs:
                k += 1
                name = f"si_{k:03d}"
                e_mace, pos = traj[step]
                write_pw_input(f"qe/{name}.in", name, cell, pos)
                manifest.append(dict(
                    name=name, vol_pct=vol, seed=seed, step=step, n_atoms=len(pos),
                    split="valid" if abs(vol - VALID_VOLUME_PCT) < 1e-9 else "train",
                    mace_energy=e_mace, cell=cell.tolist(), pos=np.asarray(pos).tolist()))
    with open("manifest.json", "w") as f:
        json.dump(manifest, f)
    ntr = sum(m["split"] == "train" for m in manifest)
    print(f"\nWrote {len(manifest)} inputs to qe/  ({ntr} train, {len(manifest) - ntr} valid) "
          f"and manifest.json")

if __name__ == "__main__":
    main()
