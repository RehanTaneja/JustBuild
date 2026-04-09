from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .llm import LLMClient
from .orchestrator import OrchestratorAgent
from .prototype import slugify

"""
This file is the CLI entry point which is how a user actually runs this. It takes a raw idea from the user and kicks off the whole pipeline.
"""

# defines how user talks to the system via terminal by making a argument parser.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JustBuild multi-agent prototype generator")
    parser.add_argument("idea", help="High-level product idea")
    parser.add_argument(
        "--output-root",
        default="build_output",
        help="Directory where generated builds should be written",
    )
    parser.add_argument("--provider", default=os.getenv("JUSTBUILD_LLM_PROVIDER"), help="LLM provider: openai, anthropic, gemini, openai_compatible")
    parser.add_argument("--model", default=os.getenv("JUSTBUILD_LLM_MODEL"), help="Cloud LLM model identifier")
    parser.add_argument("--base-url", default=os.getenv("JUSTBUILD_LLM_BASE_URL"), help="Base URL for provider or local OpenAI-compatible endpoint")
    parser.add_argument("--api-key", default=os.getenv("JUSTBUILD_LLM_API_KEY"), help="Cloud LLM API key")
    parser.add_argument("--local-model", default=os.getenv("JUSTBUILD_LLM_LOCAL_MODEL"), help="Local model name served by an OpenAI-compatible endpoint")
    parser.add_argument("--enable-playwright", action="store_true", default=os.getenv("JUSTBUILD_ENABLE_PLAYWRIGHT") == "1", help="Enable optional Playwright browser validation")
    parser.add_argument("--node-bin", default=os.getenv("JUSTBUILD_NODE_BIN", "node"), help="Node.js binary for JS runtime validation")
    parser.add_argument("--pytest-bin", default=os.getenv("JUSTBUILD_PYTEST_BIN", "pytest"), help="Pytest binary for Python execution validation")
    return parser

# This is the actual execution flow.
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv) # Parses all the input.

    output_root = Path(args.output_root).resolve() # Converts string to proper file path and makes it absolute.
    llm_client = LLMClient(
        api_key=args.api_key,
        local_model=args.local_model,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
    )
    orchestrator = OrchestratorAgent(
        product_idea=args.idea,
        output_root=output_root,
        llm_client=llm_client,
        enable_playwright=args.enable_playwright,
        node_bin=args.node_bin,
        pytest_bin=args.pytest_bin,
    ) # Initializes orchestrator.
    context = orchestrator.run() # Runs the entire system.

    payload = {
        "idea": args.idea,
        "llm_backend": {
            "provider": context.request.llm_provider,
            "model": context.request.llm_model,
            "base_url": context.request.llm_base_url,
            "backend_type": context.request.llm_backend_type,
        },
        "testing_backend": {
            "enable_playwright": context.request.enable_playwright,
            "node_bin": context.request.node_bin,
            "pytest_bin": context.request.pytest_bin,
        },
        "prototype_dir": str(context.implementation.prototype_dir) if context.implementation and context.implementation.prototype_dir else None,
        "summary_path": str(context.build_summary_path) if context.build_summary_path else None,
        "final_report_path": str(context.final_report_path) if context.final_report_path else None,
        "passed": context.testing.passed if context.testing else False,
        "iterations": len(context.iterations),
        "roadmap": [
            "Replace mocked API seams with real service implementations.",
            "Add persistent storage and authentication.",
            "Move execution into distributed workers with durable state.",
            "Introduce policy, security, and human approval gates for risky changes.",
        ],
        "slug": slugify(args.idea),
    } # This is the clean summary output.
    print(json.dumps(payload, indent=2))
    return 0
