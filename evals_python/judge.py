"""Claude as the LLM-as-judge for DeepEval metrics."""

import anthropic
from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

from app import config


class ClaudeJudge(DeepEvalBaseLLM):
    def __init__(self, model: str = config.JUDGE_MODEL):
        self.model_name = model
        self.client = anthropic.Anthropic()
        self.async_client = anthropic.AsyncAnthropic()

    def load_model(self):
        return self.client

    def get_model_name(self) -> str:
        return self.model_name

    def _kwargs(self, prompt: str) -> dict:
        return {
            "model": self.model_name,
            "max_tokens": 16000,
            "output_config": {"effort": "low"},
            "messages": [{"role": "user", "content": prompt}],
        }

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema is not None:
            return self.client.messages.parse(**self._kwargs(prompt), output_format=schema).parsed_output
        response = self.client.messages.create(**self._kwargs(prompt))
        return "".join(b.text for b in response.content if b.type == "text")

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema is not None:
            response = await self.async_client.messages.parse(**self._kwargs(prompt), output_format=schema)
            return response.parsed_output
        response = await self.async_client.messages.create(**self._kwargs(prompt))
        return "".join(b.text for b in response.content if b.type == "text")
