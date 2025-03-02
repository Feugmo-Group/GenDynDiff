#How the input of the NPT file would work


import torch
from gendyndiff.diffusion.data.batched_data import SimpleBatchedData
from ase.io.lammpsrun import read_lammps_dump_text
from ase.io import read
from gendyndiff.common.data.collate import collate
from torch_geometric.data import Data

atoms=read_lammps_dump_text(fileobj=open('../../../datasets/SrTiO3/dump.NPT'), index=-1) # reading the  last structure
cfg = read('../../../datasets/SrTiO3/SrTiO3_supercell_555.cfg')

positions = atoms.get_positions()
velocities = atoms.get_velocities()
atomic_numbers = atoms.numbers
symbols = atoms.get_chemical_symbols()
lattice = atoms.cell.array
forces = atoms.get_forces()

structure_id = atoms
num_atoms = len(symbols)


print("Positions:", positions)
print("Velocities:", velocities)
print("Atomic Numbers:", atomic_numbers)
print("Chemical Symbols:", symbols)
print("Lattice Parameters:", lattice)
print("Forces:", forces)

positions_tensor = torch.tensor(positions, dtype=torch.float32)
velocities_tensor = torch.tensor(velocities, dtype=torch.float32)
atomic_types_tensor = torch.tensor(atomic_numbers, dtype=torch.long)
forces_tensor = torch.tensor(forces, dtype=torch.float32)
batch_indices_tensor = torch.zeros(len(atomic_numbers), dtype=torch.long)
lattice_tensor = torch.tensor(lattice, dtype=torch.float32).unsqueeze(0)

data = {
    "positions": positions_tensor,
    "velocities": velocities_tensor,
    "lattice": lattice_tensor,
    "atomic_types": atomic_types_tensor,
    "forces": forces_tensor,
}
batch_idx = {
    "positions": batch_indices_tensor,
    "velocities": batch_indices_tensor,
    "lattice": torch.tensor([0], dtype=torch.long),
    "atomic_types": batch_indices_tensor,
    "forces": batch_indices_tensor,
}

batched_dataset = SimpleBatchedData(data=data, batch_idx=batch_idx)

#using collate function, turn all data input


data_obj = Data(
    positions=positions_tensor,
    velocities=velocities_tensor,
    lattice=lattice_tensor.squeeze(0),  # if you want to remove the extra batch dim
    atomic_types=atomic_types_tensor,
    forces=forces_tensor,
)
batched_data_collate = collate([data_obj])
print("Batched data:")
print(batched_data_collate)

for i in range(625):
    pos = batched_data_collate.positions[i].tolist()
    atype = batched_data_collate.atomic_types[i].item()
    vel = batched_data_collate.velocities[i].tolist()
    frc = batched_data_collate.forces[i].tolist()
    lattice = batched_data_collate.lattice.tolist()
    print(f"Atom {i + 1}:")
    print(f" Position: {pos}")
    print(f" Atomic Type: {atype}")
    print(f" Lattice: {lattice}")
    print(f" Velocity: {vel}")
    print(f" Force: {frc}")