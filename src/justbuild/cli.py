from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    return parser

# This is the actual execution flow.
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv) # Parses all the input.

    output_root = Path(args.output_root).resolve() # Converts string to proper file path and makes it absolute.
    orchestrator = OrchestratorAgent(product_idea=args.idea, output_root=output_root) # Initializes orchestrator.
    context = orchestrator.run() # Runs the entire system.

    payload = {
        "idea": args.idea,
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
