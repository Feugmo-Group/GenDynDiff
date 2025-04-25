import hydra
from omegaconf import DictConfig, OmegaConf
from gendyndiff.common.utils.globals import MODELS_PROJECT_ROOT
def train(cfg: DictConfig) -> None:
    print(f"Training {cfg.data.lightning_module.diffusion_module.model} with {cfg.data.data_module}...")
    from hydra.utils import instantiate
    # Instantiate your data module from the config
    datamodule = instantiate(cfg.data.data_module)
    # Instantiate the Lightning module wrapping the diffusion model
    lightning_module = instantiate(cfg.data.lightning_module)
    # Instantiate the PyTorch Lightning Trainer from the top-level trainer config
    trainer = instantiate(cfg.data.trainer)

    # Start training
    trainer.fit(lightning_module, datamodule=datamodule)
def validate_batch(cfg: DictConfig) -> None:
    print(f"Validating with {cfg.evaluation.metrics}.")

@hydra.main(
    config_path=str( MODELS_PROJECT_ROOT / "conf" ), config_name="default", version_base="1.1"
)
def run_pipeline(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(f"Fine-tuning a new {cfg.data.lightning_module.diffusion_module.model} on {cfg.data.data_module.train_dataset}...")
    print(MODELS_PROJECT_ROOT)

    train(cfg)
    # validate_batch(cfg)
    #if cfg.data.lightning_module.diffusion_module.model.fine_tune:
    #    train(cfg)
    #    validate_batch(cfg)
    #
    # if "perplexity" in cfg.evaluation.metrics:
    #     print("Perplexity is included in evaluation metrics.")

if __name__ == "__main__":
    run_pipeline()