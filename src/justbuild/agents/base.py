from __future__ import annotations

from dataclasses import dataclass

from ..llm import LLMClient
from ..models import BuildContext
from ..observability import BuildLogger


@dataclass(slots=True)
class AgentDependencies:
    context: BuildContext
    logger: BuildLogger
    llm: LLMClient


class BaseAgent:
    name = "base-agent"

    def __init__(self, deps: AgentDependencies) -> None:
        self.deps = deps

    @property
    def context(self) -> BuildContext:
        return self.deps.context

    @property
    def logger(self) -> BuildLogger:
        return self.deps.logger

    @property
    def llm(self) -> LLMClient:
        return self.deps.llm
