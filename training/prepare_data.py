"""Build SFT demonstrations from development cases only (todo Task 7)."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SFTExample:
    """One chat-format demonstration: messages end with the target assistant turn."""
    case_id: str
    messages: list[dict]
    metadata: dict = field(default_factory=dict)


def load_dev_cases(path: Path) -> list[dict]:
    """Read development cases; MUST refuse held-out cases and the unseen category."""
    raise NotImplementedError("TODO(Task 7): load and split-check cases")


def build_examples(cases: list[dict]) -> list[SFTExample]:
    """Turn cases into tool-use demonstrations (schema-valid tool calls + cited plans)."""
    raise NotImplementedError("TODO(Task 7): build demonstrations")


def write_jsonl(examples: list[SFTExample], out_path: Path) -> None:
    raise NotImplementedError("TODO(Task 7): write JSONL")
