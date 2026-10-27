"""Model client used by the agent. Wraps the TACC-hosted vLLM chat endpoint.

The existing standard-library client is deploy/tacc/qwen-public-client/qwen_client.py;
fold it in here (Task 4) so agent code has one import path.
"""
from dataclasses import dataclass, field


@dataclass
class ChatResult:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_s: float = 0.0


class ModelClient:
    def __init__(self, base_url: str, model: str = "qwen3.8-27b", timeout_s: float = 300):
        self.base_url, self.model, self.timeout_s = base_url, model, timeout_s

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int = 512, temperature: float = 0.0) -> ChatResult:
        raise NotImplementedError("TODO(Task 4): port qwen_client.chat, add tool support")
