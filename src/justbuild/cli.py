from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .llm import LLMClient, LLMConfigurationError
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
    parser.add_argument("--llm-timeout-s", type=int, default=int(os.getenv("JUSTBUILD_LLM_TIMEOUT_S", "60")), help="Timeout in seconds for each LLM provider request")
    parser.add_argument("--log-mode", default=os.getenv("JUSTBUILD_LOG_MODE", "progress"), choices=["quiet", "progress", "debug"], help="Logging verbosity for live CLI diagnostics")
    parser.add_argument("--enable-playwright", action="store_true", default=os.getenv("JUSTBUILD_ENABLE_PLAYWRIGHT") == "1", help="Enable optional Playwright browser validation")
    parser.add_argument("--node-bin", default=os.getenv("JUSTBUILD_NODE_BIN", "node"), help="Node.js binary for JS runtime validation")
    parser.add_argument("--pytest-bin", default=os.getenv("JUSTBUILD_PYTEST_BIN", "pytest"), help="Pytest binary for Python execution validation")
    parser.add_argument("--max-workers", type=int, default=int(os.getenv("JUSTBUILD_MAX_WORKERS", "4")), help="Maximum worker threads for parallel execution")
    parser.add_argument("--memory-path", default=os.getenv("JUSTBUILD_MEMORY_PATH"), help="Path to the persistent build memory JSON file")
    parser.add_argument("--publish-github", action="store_true", default=os.getenv("JUSTBUILD_PUBLISH_GITHUB") == "1", help="Publish completed builds to a GitHub repository")
    parser.add_argument("--github-repo-name", default=os.getenv("JUSTBUILD_GITHUB_REPO_NAME"), help="Optional GitHub repository name override for published builds")
    parser.add_argument("--github-visibility", default=os.getenv("JUSTBUILD_GITHUB_VISIBILITY", "public"), help="Visibility for published GitHub repositories")
    return parser


def _print_configuration_error(error: LLMConfigurationError) -> None:
    message = "\n".join(
        [
            f"Configuration error: {error}",
            "",
            "Provide one of these minimum LLM setups:",
            "1. Cloud API mode:",
            "   --provider openai --model <model> --api-key <key>",
            "   or set JUSTBUILD_LLM_PROVIDER, JUSTBUILD_LLM_MODEL, JUSTBUILD_LLM_API_KEY",
            "2. Local OpenAI-compatible mode:",
            "   --provider openai_compatible --local-model <model> --base-url <url>",
            "   or set JUSTBUILD_LLM_PROVIDER, JUSTBUILD_LLM_LOCAL_MODEL, JUSTBUILD_LLM_BASE_URL",
        ]
    )
    print(message, file=sys.stderr)

# This is the actual execution flow.
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv) # Parses all the input.

    output_root = Path(args.output_root).resolve() # Converts string to proper file path and makes it absolute.
    memory_path = Path(args.memory_path).resolve() if args.memory_path else None
    llm_client = LLMClient(
        api_key=args.api_key,
        local_model=args.local_model,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        timeout_s=args.llm_timeout_s,
    )
    try:
        orchestrator = OrchestratorAgent(
            product_idea=args.idea,
            output_root=output_root,
            llm_client=llm_client,
            enable_playwright=args.enable_playwright,
            node_bin=args.node_bin,
            pytest_bin=args.pytest_bin,
            max_workers=args.max_workers,
            memory_path=memory_path,
            publish_to_github=args.publish_github,
            github_repo_name=args.github_repo_name,
            github_repo_visibility=args.github_visibility,
            timeout_s=args.llm_timeout_s,
            log_mode=args.log_mode,
        ) # Initializes orchestrator.
        context = orchestrator.run() # Runs the entire system.
    except LLMConfigurationError as exc:
        _print_configuration_error(exc)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    payload = {
        "idea": args.idea,
        "llm_backend": {
            "provider": context.request.llm_provider,
            "model": context.request.llm_model,
            "base_url": context.request.llm_base_url,
            "backend_type": context.request.llm_backend_type,
            "structured_output_mode": context.request.llm_structured_output_mode,
            "timeout_s": context.request.llm_timeout_s,
        },
        "testing_backend": {
            "enable_playwright": context.request.enable_playwright,
            "node_bin": context.request.node_bin,
            "pytest_bin": context.request.pytest_bin,
            "max_workers": context.request.max_workers,
        },
        "memory_path": str(context.request.memory_path) if context.request.memory_path else None,
        "run_dir": str(context.run_dir) if context.run_dir else None,
        "events_log_path": str(context.events_log_path) if context.events_log_path else None,
        "text_log_path": str(context.text_log_path) if context.text_log_path else None,
        "partial_summary_path": str(context.partial_summary_path) if context.partial_summary_path else None,
        "github_publish": {
            "enabled": context.github_publish.enabled if context.github_publish else False,
            "published": context.github_publish.published if context.github_publish else False,
            "repo_url": context.github_publish.repo_url if context.github_publish else None,
            "repo_full_name": context.github_publish.repo_full_name if context.github_publish else None,
            "failure_reason": context.github_publish.failure_reason if context.github_publish else None,
        },
        "prototype_dir": str(context.implementation.prototype_dir) if context.implementation and context.implementation.prototype_dir else None,
        "summary_path": str(context.build_summary_path) if context.build_summary_path else None,
        "final_report_path": str(context.final_report_path) if context.final_report_path else None,
        "passed": context.testing.passed if context.testing else False,
        "iterations": len(context.iterations),
        "workflow_terminal_state": context.workflow_terminal_state,
        "slug": slugify(args.idea),
    } # This is the clean summary output.
    print(json.dumps(payload, indent=2))
    return 0
