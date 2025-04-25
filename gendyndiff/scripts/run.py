# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging

import hydra
import omegaconf
import torch
from omegaconf import OmegaConf

from gendyndiff.common.utils.globals import MODELS_PROJECT_ROOT
from gendyndiff.diffusion.config import Config
from gendyndiff.diffusion.run import main
from gendyndiff.diffusion.data.batched_data import CustomCrystalDataset

from torch_geometric.data import Data
from gendyndiff.common.data.collate import CustomCollate

logger = logging.getLogger(__name__)
@hydra.main(
    config_path=str(MODELS_PROJECT_ROOT / "conf" ), config_name="default", version_base="1.1"
)
def gendyndiff_main(cfg: omegaconf.DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"data_module.root_dir: {cfg.data.data_module.root_dir}")
    # Tensor Core acceleration (leads to ~2x speed-up during training)
    torch.set_float32_matmul_precision("high")
    dump_file_path = cfg.data.train_dataset.dump_file_path    #cfg is not related here, need the correct path
    cfg_file_path = cfg.data.train_dataset.cfg_file_path
    print(dump_file_path)
    print(cfg_file_path)
    # Load custom dataset
    dataset = CustomCrystalDataset.from_dump_file(
        dump_file_path=dump_file_path,
    )
    print("Custom Dataset Contents:")
    print(dataset)
    data_obj = dataset
    custom_collate = CustomCollate()

    # Collate the single Data object (wrapped in a list)
    batched_data = custom_collate([data_obj])
    # Print every atom's details (positions, atomic types, lattice, velocity, force)
    custom_collate.print_atoms(batched_data)


if __name__ == "__main__":
    gendyndiff_main()

    #why isnt commit working
#commit
#commitss