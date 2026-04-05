from __future__ import annotations

from dataclasses import dataclass

from ..models import BuildContext
from ..observability import BuildLogger


@dataclass(slots=True)
class AgentDependencies:
    context: BuildContext
    logger: BuildLogger


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
