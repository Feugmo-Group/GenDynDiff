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
    config_path=str(MODELS_PROJECT_ROOT / "conf" / "data" ), config_name="defaults", version_base="1.1"
)
def orgomol_main(cfg: omegaconf.DictConfig):
    # Tensor Core acceleration (leads to ~2x speed-up during training)
    torch.set_float32_matmul_precision("high")
    dump_file_path = cfg.data_module.train_dataset.dump_file_path
    cfg_file_path = cfg.data_module.train_dataset.cfg_file_path

    # Load custom dataset
    dataset = CustomCrystalDataset.from_dump_file(
        dump_file_path=dump_file_path,
        cfg_file_path=cfg_file_path,
    )
    print("Custom Dataset Contents:")
    print(dataset)

    #main(config)
    data_obj = dataset[0]
    custom_collate = CustomCollate()

    # Collate the single Data object (wrapped in a list)
    batched_data = custom_collate([data_obj])
    # Print every atom's details (positions, atomic types, lattice, velocity, force)
    custom_collate.print_atoms(batched_data)


if __name__ == "__main__":
    orgomol_main()

    #why isnt commit working
#commit
#commits