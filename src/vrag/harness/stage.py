"""Typed pipeline stages. See AGENT_BUILD_SPEC.md §7.2 item 1.

Every stage is async, declares whether it's optional (droppable under deadline pressure), and its
minimum viable time budget. Stages are pure with respect to PipelineContext: they read and append,
never mutate history (see docs/CONVENTIONS.md).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from vrag.harness.pipeline import PipelineContext


class StageResult(BaseModel):
    stage_name: str
    skipped: bool = False
    skip_reason: str | None = None
    duration_ns: int = 0


class Stage(ABC):
    name: str
    min_viable_ms: float
    optional: bool = False

    @abstractmethod
    async def run(self, ctx: PipelineContext) -> StageResult: ...
