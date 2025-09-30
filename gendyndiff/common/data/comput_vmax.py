# compute_vmax.py
# Minimal script: prints the largest absolute velocity component in the dump.

import numpy as np
from ase.io.lammpsrun import read_lammps_dump_text

dump_file_path = "/home/advaitgore/PycharmProjects/GenDynDiff/datasets/SrTiO3/dump.NPT"

with open(dump_file_path, "r") as f:
    traj = read_lammps_dump_text(fileobj=f, index=slice(None))

vmax = np.abs(np.concatenate([atoms.get_velocities() for atoms in traj])).max()
print(f"max |v| ≈ {vmax:.3f}")
