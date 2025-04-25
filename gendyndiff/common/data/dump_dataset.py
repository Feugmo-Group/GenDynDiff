# custom_dump_dataset.py

"""
Custom Dump Dataset for Molecular Dynamics Trajectories

This module defines DumpCrystalDataset, a custom dataset class for loading
a LAMMPS dump file (e.g. an NPT simulation file of SrTiO3), extracting the
velocities (instead of positions), and converting each timestep into a ChemGraph
object. The ChemGraph objects store velocity data in the 'pos' field, along with
dummy placeholders for required fields (cell, atomic_numbers, and num_atoms).
This ensures compatibility with the GenDynDiff (MatterGen) pipeline and its
CrystDataModule.
"""

import torch
from torch.utils.data import Dataset
from ase.io.lammpsrun import read_lammps_dump_text
from gendyndiff.common.data.chemgraph import ChemGraph

class DumpDataset(Dataset):
    """
    DumpCrystalDataset loads a LAMMPS dump file and converts the MD trajectory into a
    dataset of ChemGraph objects. Each ChemGraph represents a single timestep with:
      - pos: Velocity data as a (N, 3) tensor (where N is the number of atoms)
      - cell: A dummy (3, 3) cell matrix (using an identity matrix)
      - atomic_numbers: Dummy atomic numbers (zeros) of shape (N,)
      - num_atoms: The number of atoms in that timestep
      - timestep: The timestep index
    This dataset is intended to be used with the GenDynDiff data module.
    """

    def __init__(self, data_objs):
        self.data_objs = data_objs

    def __len__(self):
        return len(self.data_objs)

    def __getitem__(self, idx):
        return self.data_objs[idx]

    @classmethod
    def from_dump_file(cls, dump_file_path: str) -> "DumpDataset":
        """
        Load a dump file using ASE's read_lammps_dump_text and convert all timesteps
        into ChemGraph objects with velocity data stored in the 'pos' field.

        Args:
            dump_file_path (str): Path to the LAMMPS dump file (e.g., an NPT file).

        Returns:
            DumpCrystalDataset: A dataset that can be used by the GenDynDiff data module.
        """
        with open(dump_file_path, 'r') as file:
            all_timesteps = read_lammps_dump_text(fileobj=file, index=slice(None))

        data_objects = []
        for idx, atoms in enumerate(all_timesteps):
            # Extract velocities from the current timestep (shape: [N, 3])
            velocities_tensor = torch.tensor(atoms.get_velocities(), dtype=torch.float32)
            lattice_shape = torch.tensor(atoms.get_cell(), dtype=torch.float32)            # Retrieve the timestep information (if available)
            timestep = atoms.info.get('ITEM: TIMESTEP', idx)
            atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
            # Create a ChemGraph; placeholders are used for required fields:
            # - cell: an identity matrix as a dummy cell (3x3)
            # - atomic_numbers: a tensor of zeros with length equal to the number of atoms
            # - num_atoms: the actual number of atoms in this timestep
            chemgraph_obj = ChemGraph(
                pos=velocities_tensor,  # Velocity data in pos field
                cell=lattice_shape.unsqueeze(0),  # Real cells, work on fixing the pbs problem wednesday
                atomic_numbers=atomic_numbers,  # Real atomic numbers
                num_atoms=torch.tensor(len(atoms), dtype=torch.long),
                timestep=torch.tensor(idx, dtype=torch.long)
            )
            data_objects.append(chemgraph_obj)
        return cls(data_objects)


if __name__ == "__main__":
    # Example usage:
    dump_file_path = "/home/agore/GenDynDiff/datasets/SrTiO3/dump.NPT"
    dataset = DumpDataset.from_dump_file(dump_file_path)
    print(f"Loaded dataset with {len(dataset)} timesteps.")
    print(dataset[0])
