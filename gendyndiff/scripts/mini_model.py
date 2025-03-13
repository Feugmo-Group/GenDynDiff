import torch
from gendyndiff.common.data.collate import CustomCollate
from gendyndiff.diffusion.data.batched_data import CustomCrystalDataset
from gendyndiff.diffusion.diffusion_module import DiffusionModule
from gendyndiff.diffusion.losses import DenoisingScoreMatchingLoss
from gendyndiff.diffusion.timestep_samplers import UniformTimestepSampler
from gendyndiff.diffusion.corruption.sde_lib import VPSDE
from gendyndiff.diffusion.corruption.multi_corruption import MultiCorruption
from gendyndiff.diffusion.corruption.sde_lib import ZeroSDE

dump_file_path = "/home/agore/GenDynDiff/datasets/SrTiO3/dump.NPT"
cfg_file_path = "/home/agore/GenDynDiff/datasets/SrTiO3/SrTiO3_supercell_555.cfg"

# Load dataset using CustomCrystalDataset class
custom_collate = CustomCollate()
datasets = CustomCrystalDataset.from_dump_file(dump_file_path, cfg_file_path)

# Iterate through each timestep's dataset
for timestep_idx, dataset in enumerate(datasets):
    print(f"Timestep {timestep_idx + 1}:")
    batched_data = custom_collate([dataset])
    custom_collate.print_atoms(batched_data)
    print("-" * 50)

    velocity_corruption = VPSDE(beta_min=0.1, beta_max=20)

    # Define model targets and weights for position-only loss
    model_targets = {"velocities": "score_times_std"}
    weights = {"positions": 1.0, "velocities": 1.0}  # Only track position errors

    # Configure corruption process
    corruption = MultiCorruption(
        sdes={"velocities": velocity_corruption},
        # Add dummy corruption for required fields if needed
        # sdes={"velocities": velocity_corruption, "positions": ZeroSDE()}
    )

score_model = GemNetT(
    num_targets=3,
    latent_dim=128,
    atom_embedding=atom_embedding,
    num_spherical=7,
    num_radial=128,
    num_blocks=3,
    emb_size_atom=512,
    emb_size_edge=512,
    emb_size_trip=64,
    emb_size_rbf=16,
    emb_size_cbf=16,
    cutoff=6.0,
    activation="swish",
    max_neighbors=50,
)

model_targets = {
    "positions" : "score_times_std",
    "lattice": "score_times_std",
    "atomic_types": "logits",
}
#weights(lambda) for each loss in the loss function
weights = {
    "positions": 1.0,
    "velocities": 1.0,
}
#loss function with a concrete implementation
loss_fn = DenoisingScoreMatchingLoss(
    model_targets=model_targets,
    weights=weights,
)