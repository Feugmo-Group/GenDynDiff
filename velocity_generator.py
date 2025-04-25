

import io
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from gendyndiff.common.data.chemgraph import ChemGraph
from gendyndiff.common.data.collate import collate
from gendyndiff.common.data.condition_factory import ConditionLoader
from gendyndiff.diffusion.sampling.pc_sampler import PredictorCorrector
from gendyndiff.common.utils.globals import DEFAULT_SAMPLING_CONFIG_PATH, get_device
from gendyndiff.diffusion.lightning_module import DiffusionLightningModule


@dataclass
class VelocityGenerator:
    checkpoint_info: dict  # Contains model checkpoint info
    batch_size: int = 1
    num_batches: int = 1
    output_dir: str = "velocity_outputs"
    record_trajectories: bool = True

    _model: Optional[DiffusionLightningModule] = None
    _cfg: Optional[DictConfig] = None

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare(self):
        """Load the trained velocity diffusion model"""
        if self._model is not None:
            return

        # Load model configuration and weights
        self._model = instantiate(self.checkpoint_info.config.lightning_module)
        self._model.load_state_dict(torch.load(self.checkpoint_info.path))
        self._model = self._model.to(get_device())
        self._cfg = self.checkpoint_info.config

    def generate_velocities(self):
        """Main generation entry point"""
        self.prepare()

        # Initialize sampler with velocity diffusion configuration
        sampler = instantiate(self._cfg.sampling.sampler_partial)
        sampler.setup(self._model)

        # Generate velocities
        all_velocities = []
        for _ in tqdm(range(self.num_batches), desc="Generating velocity batches"):
            batch_velocities = self._generate_batch(sampler)
            all_velocities.extend(batch_velocities)

        # Save results
        self._save_velocities(all_velocities)
        return all_velocities

    def _generate_batch(self, sampler: PredictorCorrector) -> List[np.ndarray]:
        """Generate a single batch of velocities"""
        # Create dummy conditioning (modify based on your actual conditioning needs)
        condition_loader = ConditionLoader(
            batch_size=self.batch_size,
            num_samples=self.batch_size,
            properties_to_condition_on={}
        )

        # Run sampling process
        if self.record_trajectories:
            samples, _, trajectories = sampler.sample_with_record(condition_loader)
        else:
            samples, _ = sampler.sample(condition_loader)

        # Extract velocities from ChemGraph
        batch_velocities = [graph.pos.cpu().numpy() for graph in samples.to_data_list()]

        if self.record_trajectories:
            self._save_trajectories(trajectories)

        return batch_velocities

    def _save_velocities(self, velocities: List[np.ndarray]):
        """Save generated velocities to NPZ file"""
        output_path = Path(self.output_dir) / "generated_velocities.npz"
        np.savez(output_path, velocities=velocities)
        print(f"Saved {len(velocities)} velocity sets to {output_path}")

    def _save_trajectories(self, trajectories: List[List[ChemGraph]]):
        """Save denoising trajectories (velocity evolution)"""
        trajectory_data = []
        for traj in trajectories:
            trajectory_data.append([graph.pos.cpu().numpy() for graph in traj])

        output_path = Path(self.output_dir) / "velocity_trajectories.npz"
        np.savez(output_path, trajectories=trajectory_data)
        print(f"Saved denoising trajectories to {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize with your checkpoint info
    checkpoint_info = {
        "path": "path/to/checkpoint.ckpt",
        "config": OmegaConf.load("your/config.yaml")
    }

    generator = VelocityGenerator(
        checkpoint_info=checkpoint_info,
        batch_size=4,
        num_batches=5,
        output_dir="velocity_results"
    )

    generated_velocities = generator.generate_velocities()
