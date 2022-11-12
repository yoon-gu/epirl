import hydra
from omegaconf import DictConfig


@hydra.main(config_path=".", config_name="conf.yaml")
def sphere(cfg: DictConfig) -> float:
    x: float = cfg.x
    y: float = cfg.y
    return x**2 + y**2


if __name__ == "__main__":
    sphere()