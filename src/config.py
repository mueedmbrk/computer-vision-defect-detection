from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 224
    num_classes: int = 2
    class_names: tuple[str, str] = ("Normal", "Defective")
    model_name: str = "efficientnet_b0_v1"
    checkpoint_name: str = "best_model.pt"


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = ROOT / "artifacts"
DEFAULT_CHECKPOINT = DEFAULT_ARTIFACTS / ModelConfig.checkpoint_name
