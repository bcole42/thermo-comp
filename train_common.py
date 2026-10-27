import re
import numpy as np

A_LATTICE = 5.469            # Angstrom, PBE lattice constant of diamond Si
N_SUPER = 2                  # 2x2x2 of the 8-atom cubic cell = 64 atoms (~10.9 A)
VOLUME_PCT = [-5.0, -2.5, 0.0, 2.5, 5.0]   # volume change in percent
RATTLES_PER_VOLUME = 2       # independent rattled starting structures per volume
SNAP_FRACS = [1.0, 0.3]      # snapshots per relaxation, as a fraction of the initial
                             #   excess energy (1.0 = rattled start, 0.3 = partly relaxed)
                             # => 5 volumes x 2 rattles x 2 snapshots = 20 configurations
VALID_VOLUME_PCT = 0.0       # frames at this volume become the validation set (4 of 20)
RATTLE_STD = 0.05            # Angstrom per Cartesian component (~0.1 A displacements)
STRAIN_STD = 0.01            # std of random symmetric strain (volume-preserving)
SEED = 2026                  # fixed seed -> everyone generates identical structures

MACE_MODEL = "medium-0b3"    # model name

DEVICE = "cpu"
FMAX = 0.02                  # eV/A relaxation threshold
MAX_STEPS = 300

# DFT settings (must match convergence tests + reference phonons)
PSEUDO_FILE = "Si.us.pbesol.z_4.ld1.psl.v1.0.0-high.upf"
PSEUDO_DIR = "../pseudo"
MASS_SI = 28.0855
ECUT = 40                    # Ry, from your convergence test
DUAL = 8                     # ecutrho = DUAL*ecutwfc
K_TRAIN = 3                  # NxNxN k-mesh on the 64-atom cell, from your k-point test
CONV_THR = "1.0d-10"

RY_TO_EV = 13.605693123
BOHR = 0.529177210903
RY_BOHR_TO_EV_ANG = RY_TO_EV / BOHR
KBAR_TO_EV_A3 = 6.241509074e-4      # 1 kbar = 0.1 GPa


def build_si(n=N_SUPER):
    """Perfect diamond-cubic Si, n x n x n copies of the 8-atom cell. Returns (cell, pos)."""
    fcc = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    basis = np.vstack([fcc, fcc + 0.25])
    frac = np.vstack([(basis + [i, j, k]) / n
                      for i in range(n) for j in range(n) for k in range(n)])
    cell = np.eye(3) * A_LATTICE * n
    return cell, frac @ cell


def write_pw_input(path, prefix, cell, pos):
    """pw.x scf single point that prints energy, forces and stress."""
    lines = [
        "&CONTROL", "  calculation = 'scf'", f"  prefix = '{prefix}'", "  outdir = './tmp'",
        f"  pseudo_dir = '{PSEUDO_DIR}'", "  tprnfor = .true.", "  tstress = .true.",
        "  disk_io = 'none'", "/",
        "&SYSTEM", "  ibrav = 0", f"  nat = {len(pos)}", "  ntyp = 1",
        f"  ecutwfc = {ECUT}", f"  ecutrho = {ECUT * DUAL}",
        "  occupations = 'fixed'", "  nspin = 1", "/",
        "&ELECTRONS", f"  conv_thr = {CONV_THR}", "  mixing_beta = 0.5", "  electron_maxstep = 200", "/",
        "ATOMIC_SPECIES", f"  Si {MASS_SI} {PSEUDO_FILE}",
        "K_POINTS automatic", f"  {K_TRAIN} {K_TRAIN} {K_TRAIN} 0 0 0",
        "CELL_PARAMETERS angstrom",
    ]
    lines += ["  %.10f %.10f %.10f" % tuple(v) for v in cell]
    lines.append("ATOMIC_POSITIONS angstrom")
    lines += ["  Si %.10f %.10f %.10f" % tuple(p) for p in pos]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def parse_pw_output(path):
    """dict(energy [eV], forces [eV/A, (N,3)], stress_qe [kbar, 3x3, QE sign convention])."""
    text = open(path).read()
    if "JOB DONE" not in text:
        raise RuntimeError("calculation did not finish")
    e = re.findall(r"^!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", text, re.M)
    energy = float(e[-1]) * RY_TO_EV
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if "Forces acting on atoms" in l)
    forces = []
    for l in lines[start + 1:]:
        m = re.match(r"\s+atom\s+\d+\s+type\s+\d+\s+force =\s+(\S+)\s+(\S+)\s+(\S+)", l)
        if m:
            forces.append([float(v) for v in m.groups()])
        elif forces:
            break
    s0 = next(i for i, l in enumerate(lines) if "total   stress" in l)
    stress = np.array([[float(x) for x in lines[s0 + 1 + r].split()[3:6]] for r in range(3)])
    return dict(energy=energy, forces=np.array(forces) * RY_BOHR_TO_EV_ANG, stress_qe=stress)


def write_extxyz(path, frames):
    """Write frames as extended XYZ with MACE-style REF_* keys.

    Each frame: dict(cell, pos, forces, energy, stress (3x3, ASE sign, eV/A^3), info {key: value}).
    """
    with open(path, "w") as f:
        for fr in frames:
            n = len(fr["pos"])
            lat = " ".join("%.10f" % x for x in np.asarray(fr["cell"]).flatten())
            st = " ".join("%.10e" % x for x in np.asarray(fr["stress"]).flatten())
            extra = " ".join(f"{k}={v}" for k, v in fr["info"].items())
            f.write(f"{n}\n")
            f.write(f'Lattice="{lat}" Properties=species:S:1:pos:R:3:REF_forces:R:3 '
                    f'REF_energy={fr["energy"]:.10f} REF_stress="{st}" pbc="T T T" {extra}\n')
            for p, g in zip(fr["pos"], fr["forces"]):
                f.write("Si %.10f %.10f %.10f %.10e %.10e %.10e\n" % (*p, *g))
