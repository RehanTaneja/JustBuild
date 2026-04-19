from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import BuildContext, BuildMemory, BuildMemoryEntry, PatternRecord

MAX_PAST_BUILDS = 10
MAX_PATTERN_EXAMPLES = 3
MAX_PROMPT_PATTERNS = 3
MEMORY_FILENAME = "build_memory.json"


def default_memory_path(output_root: Path) -> Path:
    return output_root / MEMORY_FILENAME


def load_build_memory(path: Path) -> BuildMemory:
    if not path.exists():
        return BuildMemory()

    payload = json.loads(path.read_text(encoding="utf-8"))
    return BuildMemory(
        past_builds=[
            BuildMemoryEntry(
                build_id=entry["build_id"],
                product_idea=entry["product_idea"],
                title=entry["title"],
                passed=bool(entry["passed"]),
                architecture_findings=list(entry.get("architecture_findings", [])),
                failure_groups=list(entry.get("failure_groups", [])),
                root_cause=entry.get("root_cause"),
                successful_patterns=list(entry.get("successful_patterns", [])),
            )
            for entry in payload.get("past_builds", [])
        ],
        failure_patterns=_load_pattern_groups(payload.get("failure_patterns", {})),
        successful_patterns=_load_pattern_groups(payload.get("successful_patterns", {})),
    )


def save_build_memory(path: Path, memory: BuildMemory) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(memory), indent=2), encoding="utf-8")
    return path


def update_build_memory(context: BuildContext) -> BuildMemory:
    memory = context.memory or BuildMemory()
    title = context.specification.title if context.specification else context.request.product_idea
    build_id = title.lower().replace(" ", "-")
    architecture_findings = context.architecture_review.findings if context.architecture_review else []
    failure_groups = context.debugging.failure_groups if context.debugging else []
    successful_patterns = _extract_successful_patterns(context)
    root_cause = context.debugging.root_cause if context.debugging else None

    memory.past_builds.append(
        BuildMemoryEntry(
            build_id=build_id,
            product_idea=context.request.product_idea,
            title=title,
            passed=bool(context.testing and context.testing.passed),
            architecture_findings=list(architecture_findings),
            failure_groups=list(failure_groups),
            root_cause=root_cause,
            successful_patterns=successful_patterns,
        )
    )
    memory.past_builds = memory.past_builds[-MAX_PAST_BUILDS:]

    for finding in architecture_findings:
        _record_pattern(memory.failure_patterns, "failed_architectures", finding)

    for report in context.testing.failure_reports if context.testing else []:
        _record_pattern(memory.failure_patterns, "repeated_bugs", report.summary, *report.details)

    if context.debugging:
        if context.debugging.root_cause:
            _record_pattern(memory.failure_patterns, "repeated_bugs", context.debugging.root_cause)
        for group in context.debugging.failure_groups:
            _record_pattern(memory.failure_patterns, "failure_groups", group)

    for pattern in successful_patterns:
        _record_pattern(memory.successful_patterns, "successful_patterns", pattern)

    context.memory = memory
    return memory


def build_memory_prompt_context(memory: BuildMemory | None) -> tuple[list[str], list[str]]:
    if memory is None:
        return [], []

    failures = _top_pattern_strings(memory.failure_patterns)
    successes = _top_pattern_strings(memory.successful_patterns)
    return failures[:MAX_PROMPT_PATTERNS], successes[:MAX_PROMPT_PATTERNS]


def _extract_successful_patterns(context: BuildContext) -> list[str]:
    patterns: list[str] = []
    if context.testing and context.testing.passed:
        patterns.append("Passing testing pipeline with execution, schema, and browser-aware checks.")
        if context.debugging:
            patterns.append(f"Debug-guided recovery strategy: {context.debugging.strategy}")
        if context.evaluation:
            patterns.extend(context.evaluation.code_quality[:1])
            patterns.extend(context.evaluation.maintainability[:1])
    return patterns[:MAX_PAST_BUILDS]


def _load_pattern_groups(payload: dict[str, list[dict[str, object]]]) -> dict[str, list[PatternRecord]]:
    groups: dict[str, list[PatternRecord]] = {}
    for group, records in payload.items():
        groups[group] = [
            PatternRecord(
                pattern=str(record["pattern"]),
                count=int(record.get("count", 1)),
                examples=[str(example) for example in record.get("examples", [])],
            )
            for record in records
        ]
    return groups


def _record_pattern(groups: dict[str, list[PatternRecord]], group: str, pattern: str, *examples: str) -> None:
    entries = groups.setdefault(group, [])
    for entry in entries:
        if entry.pattern == pattern:
            entry.count += 1
            for example in examples:
                if example and example not in entry.examples:
                    entry.examples.append(example)
            entry.examples = entry.examples[:MAX_PATTERN_EXAMPLES]
            return

    cleaned_examples = [example for example in examples if example][:MAX_PATTERN_EXAMPLES]
    entries.append(PatternRecord(pattern=pattern, count=1, examples=cleaned_examples))


def _top_pattern_strings(groups: dict[str, list[PatternRecord]]) -> list[str]:
    ranked: list[tuple[int, str]] = []
    for group, records in groups.items():
        for record in records:
            example_suffix = f" Example: {record.examples[0]}" if record.examples else ""
            ranked.append((record.count, f"{group}: {record.pattern} (seen {record.count}x).{example_suffix}"))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [item[1] for item in ranked]
