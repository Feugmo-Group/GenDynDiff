import torch
from ase.io.lammpsrun import read_lammps_dump_text

from gendyndiff.common.data.collate import CustomCollate
from gendyndiff.diffusion.data.batched_data import CustomCrystalDataset
from gendyndiff.diffusion.diffusion_module import DiffusionModule
from gendyndiff.diffusion.losses import DenoisingScoreMatchingLoss
from gendyndiff.diffusion.timestep_samplers import UniformTimestepSampler
from gendyndiff.diffusion.corruption.sde_lib import VPSDE
from gendyndiff.diffusion.corruption.multi_corruption import MultiCorruption


dump_file_path = "/home/agore/GenDynDiff/datasets/SrTiO3/dump.NPT"
cfg_file_path = "/home/agore/GenDynDiff/datasets/SrTiO3/SrTiO3_555_5pO_5pSr.cfg"

# Load dataset using CustomCrystalDataset class
custom_collate = CustomCollate()
datasets = CustomCrystalDataset.from_dump_file(dump_file_path, cfg_file_path)

# Iterate through each timestep's dataset
for timestep_idx, dataset in enumerate(datasets):
    print(f"Timestep {timestep_idx + 1}:")
    batched_data = custom_collate([dataset])
    custom_collate.print_atoms(batched_data)
    print("-" * 50)

