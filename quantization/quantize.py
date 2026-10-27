"""Quantize one checkpoint to INT8/INT4 variants with provenance (todo Task 8)."""
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class QuantConfig:
    source_model: str          # same checkpoint for every precision in an ablation
    method: str                # e.g. "awq", "gptq", "int8"
    bits: int
    calibration_file: Path     # development data only; never held-out cases
    output_dir: Path
    runtime: str = "vllm"


def load_config(path: Path) -> list[QuantConfig]:
    raise NotImplementedError("TODO(Task 8): parse configs/quantization.yaml")


def quantize(cfg: QuantConfig) -> Path:
    """Produce the artifact and return its directory."""
    raise NotImplementedError("TODO(Task 8): run calibration + quantization")


def verify_artifact(path: Path) -> bool:
    """Reload the artifact and check format; report unsupported methods, don't swap models."""
    raise NotImplementedError("TODO(Task 8): artifact load check")


def write_provenance(cfg: QuantConfig, artifact: Path) -> None:
    (artifact / "provenance.json").write_text(
        json.dumps(asdict(cfg), default=str, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/quantization.yaml"))
    args = parser.parse_args()
    for cfg in load_config(args.config):
        artifact = quantize(cfg)
        verify_artifact(artifact)
        write_provenance(cfg, artifact)


if __name__ == "__main__":
    main()
