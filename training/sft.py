"""SFT pilot: train, save, and reload a LoRA adapter (todo Task 7)."""
import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SFTConfig:
    base_model: str            # exact checkpoint; record revision for provenance
    base_revision: str
    train_file: Path
    output_dir: Path
    lora_r: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2e-4
    epochs: int = 1
    max_seq_len: int = 4096
    seed: int = 0


def load_config(path: Path) -> SFTConfig:
    raise NotImplementedError("TODO(Task 7): parse configs/sft.yaml")


def train(cfg: SFTConfig) -> Path:
    """Run SFT and return the adapter directory (with base model + tokenizer info)."""
    raise NotImplementedError("TODO(Task 7): training loop")


def reload_adapter(adapter_dir: Path):
    """Smoke test: load the saved adapter on its base model."""
    raise NotImplementedError("TODO(Task 7): reload check")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sft.yaml"))
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
