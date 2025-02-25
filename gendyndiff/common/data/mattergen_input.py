#How the input of the NPT file would work


import torch
from gendyndiff.diffusion.data.batched_data import SimpleBatchedData
from ase.io.lammpsrun import read_lammps_dump_text
from ase.io import read

atoms=read_lammps_dump_text(fileobj=open('../../../datasets/SrTiO3/dump.NPT'), index=-1) # reading the  last structure
cfg = read('../../../datasets/SrTiO3/SrTiO3_supercell_555.cfg')

positions = atoms.get_positions()
velocities = atoms.get_velocities()
atomic_numbers = atoms.numbers
symbols = atoms.get_chemical_symbols()
lattice = atoms.cell

print("Positions:", positions)
print("Velocities:", velocities)
print("Atomic Numbers:", atomic_numbers)
print("Chemical Symbols:", symbols)
print("Lattice Parameters:", lattice)

positions_tensor = torch.tensor(positions, dtype=torch.float32)  # Shape: (N_atoms, 3)
velocities_tensor = torch.tensor(velocities, dtype=torch.float32)  # Shape: (N_atoms, 3)
atomic_types_tensor = torch.tensor(atomic_numbers, dtype=torch.long)  # Shape: (N_atoms,)
batch_indices_tensor = torch.zeros(len(atomic_numbers), dtype=torch.long)  # Single batch index for all atoms
lattice_tensor = torch.tensor(lattice, dtype=torch.float32).unsqueeze(0)

data = {
    "positions": positions_tensor,
    "lattice": lattice_tensor,
    "atomic_types": atomic_types_tensor,
}

batch_idx = {
    "positions": batch_indices_tensor,
    "lattice": torch.tensor([0], dtype=torch.long),
    "atomic_types": batch_indices_tensor,
}

batched_dataset = SimpleBatchedData(data=data, batch_idx=batch_idx)
print("Batched Dataset (mattergen model input):")
print(batched_dataset)