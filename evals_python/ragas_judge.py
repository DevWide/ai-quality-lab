"""Claude as the judge for Ragas metrics.

Ragas' own Anthropic adapter (via instructor) sends sampling parameters such as
temperature and top_p; this wrapper uses the SDK's structured outputs instead,
the same way evals_python/judge.py does for DeepEval.
"""

import anthropic
from pydantic import BaseModel
from ragas.llms.base import InstructorBaseRagasLLM

from app import config


class ClaudeRagasJudge(InstructorBaseRagasLLM):
    def __init__(self, model: str = config.JUDGE_MODEL):
        self.model = model
        self.client = anthropic.Anthropic()
        self.async_client = anthropic.AsyncAnthropic()

    def _kwargs(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "max_tokens": 16000,
            "output_config": {"effort": "low"},
            "messages": [{"role": "user", "content": prompt}],
        }

    def generate(self, prompt: str, response_model: type[BaseModel]):
        response = self.client.messages.parse(**self._kwargs(prompt), output_format=response_model)
        return response.parsed_output

    async def agenerate(self, prompt: str, response_model: type[BaseModel]):
        response = await self.async_client.messages.parse(**self._kwargs(prompt), output_format=response_model)
        return response.parsed_output
