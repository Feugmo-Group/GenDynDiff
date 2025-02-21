
import hydra
from omegaconf import DictConfig, OmegaConf
from gendyndiff.common.utils.globals import MODELS_PROJECT_ROOT
def train(cfg: DictConfig) -> None:
    print(f"Training {cfg.model.name} with {cfg.dataset.name}...")

def validate_batch(cfg: DictConfig) -> None:
    print(f"Validating with {cfg.evaluation.metrics}.")

@hydra.main(
    config_path=str(MODELS_PROJECT_ROOT / "conf"), config_name="default", version_base="1.1"
)
def run_pipeline(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(f"Fine-tuning a new {cfg.model.name} on {cfg.dataset.name}")

    # if cfg.model.fine_tune:
    #     train(cfg)
    #     validate_batch(cfg)
    #
    # if "perplexity" in cfg.evaluation.metrics:
    #     print("Perplexity is included in evaluation metrics.")

if __name__ == "__main__":
    run_pipeline()