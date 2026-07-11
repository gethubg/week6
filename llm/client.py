"""LLM client abstraction so Coach/Critic agent logic can be unit-tested
without an ANTHROPIC_API_KEY. Swap FakeLLMClient for AnthropicLLMClient once
real keys are available (see .env.example) -- the agent code never changes.
"""

from __future__ import annotations

import os
from typing import Protocol, TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMClient(Protocol):
    def generate_structured(self, prompt: str, schema: type[SchemaT], system: str | None = None) -> SchemaT:
        ...


class AnthropicLLMClient:
    """Real path: forces structured output via Anthropic tool-use, then
    validates the tool call's input against the requested Pydantic schema."""

    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 2048) -> None:
        self._model = model
        self._max_tokens = max_tokens

    def generate_structured(self, prompt: str, schema: type[SchemaT], system: str | None = None) -> SchemaT:
        import anthropic  # lazy import: keeps this module importable without the package/key

        client = anthropic.Anthropic()
        tool_name = schema.__name__
        response = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system or "",
            tools=[
                {
                    "name": tool_name,
                    "description": f"Emit a {tool_name} response.",
                    "input_schema": schema.model_json_schema(),
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return schema.model_validate(block.input)
        raise RuntimeError(f"Anthropic response had no {tool_name} tool_use block")


class NebiusLLMClient:
    """Real path via Nebius AI Studio's Token Factory: an OpenAI-compatible
    endpoint, so this reuses the `openai` SDK with a custom base_url and
    forces structured output the same way AnthropicLLMClient does -- a
    tool call constrained to the requested schema."""

    _BASE_URL = "https://api.studio.nebius.com/v1/"

    def __init__(self, model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507", max_tokens: int = 2048) -> None:
        self._model = model
        self._max_tokens = max_tokens

    def generate_structured(self, prompt: str, schema: type[SchemaT], system: str | None = None) -> SchemaT:
        import openai  # lazy import: keeps this module importable without the package/key

        client = openai.OpenAI(base_url=self._BASE_URL, api_key=os.environ["NEBIUS_API_KEY"])
        tool_name = schema.__name__
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": f"Emit a {tool_name} response.",
                        "parameters": schema.model_json_schema(),
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": tool_name}},
            temperature=0,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            raise RuntimeError(f"Nebius response had no {tool_name} tool call")
        return schema.model_validate_json(message.tool_calls[0].function.arguments)


class FakeLLMClient:
    """Test/offline stand-in: returns pre-scripted responses in order, or via
    a callback if you need the response to depend on the prompt."""

    def __init__(self, responses: list[BaseModel] | None = None) -> None:
        self._responses = list(responses or [])
        self.prompts_seen: list[str] = []

    def queue(self, response: BaseModel) -> None:
        self._responses.append(response)

    def generate_structured(self, prompt: str, schema: type[SchemaT], system: str | None = None) -> SchemaT:
        self.prompts_seen.append(prompt)
        if not self._responses:
            raise AssertionError("FakeLLMClient has no queued responses left")
        response = self._responses.pop(0)
        if not isinstance(response, schema):
            raise AssertionError(f"queued response is {type(response).__name__}, expected {schema.__name__}")
        return response
