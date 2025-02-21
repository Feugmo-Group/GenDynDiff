from ase.io.lammpsrun import read_lammps_dump_text
from ase.io import read

from pymatgen.io.ase import AseAtomsAdaptor
atoms=read_lammps_dump_text(fileobj=open('../../../datasets/SrTiO3/dump.NPT'), index=-1) # reading the  last structure

cfg = read('../../../datasets/SrTiO3/SrTiO3_supercell_555.cfg')
atoms.set_atomic_numbers(cfg.get_atomic_numbers())
atoms.set_chemical_symbols(cfg.get_chemical_symbols())
def print_atoms_info(atoms):
    """Prints detailed information about an atomic system."""

    print("\n=== Atomic System Information ===\n")

    print(f"Total number of atoms: {atoms.get_number_of_atoms}\n")

    #print("Momenta of atoms (Å):")
    #print(atoms.get_momenta(), "\n")

    print("Positions of atoms (Å):")
    print(atoms.get_positions(), "\n")

    print("Velocities of atoms (Å/fs):")
    print(atoms.get_velocities(), "\n")

    print("Atomic positions (alternative access):")
    print(atoms.positions, "\n")

    print("Atomic numbers:")
    print(atoms.numbers, "\n")

    print("Unit cell parameters (Å):")
    print(atoms.cell, "\n")

    #print(f"Kinetic energy (eV): {atoms.get_kinetic_energy()}\n")

    print("Unit cell matrix (alternative method):")
    print(atoms.get_cell(), "\n")

    #print("Forces on atoms (eV/Å):")
    #print(atoms.get_forces(), "\n")

    #print(f"Potential energy (eV): {atoms.get_potential_energy()}\n")

    print(f"Chemical formula: {atoms.get_chemical_formula()}\n")

    print("Chemical symbols of atoms:")
    print(atoms.get_chemical_symbols(), "\n")

    print("You can set atomic numbers and chemical symbols using:")
    print("atoms.set_atomic_numbers([...])")
    print("atoms.set_chemical_symbols([...])\n")

    print("=================================\n")

# Example usage (assuming 'atoms' is an ASE Atoms object)
print_atoms_info(atoms)


#transform from ASE to pymatgen

structure = AseAtomsAdaptor.get_structure(atoms)







