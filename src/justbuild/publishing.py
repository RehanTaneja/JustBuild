from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import BuildContext, GitHubPublishResult
from .prototype import slugify

DEFAULT_BRANCH = "main"


class GitHubPublishError(RuntimeError):
    pass


@dataclass(slots=True)
class GitHubRepoInfo:
    repo_name: str
    repo_full_name: str
    repo_url: str
    clone_url: str


class GhCliRepoClient:
    def __init__(self, command_runner=None) -> None:
        self.command_runner = command_runner or _run_command

    def create_repo(self, repo_name: str, description: str, visibility: str) -> GitHubRepoInfo:
        owner = self._get_authenticated_user()
        visibility_flag = visibility.lower()
        for attempt in range(1, 6):
            candidate = repo_name if attempt == 1 else f"{repo_name}-{attempt}"
            try:
                stdout = self.command_runner(
                    [
                        "gh",
                        "api",
                        "user/repos",
                        "--method",
                        "POST",
                        "-f",
                        f"name={candidate}",
                        "-f",
                        f"description={description}",
                        "-f",
                        "private=false" if visibility_flag == "public" else "private=true",
                    ]
                )
                payload = json.loads(stdout)
                return GitHubRepoInfo(
                    repo_name=candidate,
                    repo_full_name=payload["full_name"],
                    repo_url=payload["html_url"],
                    clone_url=payload["clone_url"],
                )
            except GitHubPublishError as exc:
                message = str(exc).lower()
                if "already exists" in message or "name already exists" in message:
                    continue
                raise
        raise GitHubPublishError(f"Unable to create a unique GitHub repository for {owner}/{repo_name}.")

    def _get_authenticated_user(self) -> str:
        stdout = self.command_runner(["gh", "api", "user"])
        payload = json.loads(stdout)
        login = payload.get("login")
        if not login:
            raise GitHubPublishError("Unable to determine authenticated GitHub user.")
        return str(login)


class GitHubPublisher:
    def __init__(self, repo_client: GhCliRepoClient | None = None, command_runner=None) -> None:
        self.command_runner = command_runner or _run_command
        self.repo_client = repo_client or GhCliRepoClient(command_runner=self.command_runner)

    def publish(self, context: BuildContext) -> GitHubPublishResult:
        implementation = context.implementation
        if implementation is None or implementation.prototype_dir is None:
            raise GitHubPublishError("Publishing requires generated implementation artifacts.")
        if context.build_summary_path is None or context.final_report_path is None:
            raise GitHubPublishError("Publishing requires build summary and final report files.")

        repo_name = context.request.github_repo_name or f"{slugify(context.specification.title if context.specification else context.request.product_idea)}-prototype"
        description = context.request.product_idea
        repo_info = self.repo_client.create_repo(repo_name, description, context.request.github_repo_visibility)

        build_root = implementation.prototype_dir.parent
        publish_dir = build_root / "github_publish"
        self._prepare_publish_directory(context, publish_dir, repo_info.repo_url)
        commit_messages = self._create_commit_history(context, publish_dir)
        self._push_publish_directory(publish_dir, repo_info.clone_url)
        return GitHubPublishResult(
            enabled=True,
            published=True,
            repo_name=repo_info.repo_name,
            repo_full_name=repo_info.repo_full_name,
            repo_url=repo_info.repo_url,
            branch=DEFAULT_BRANCH,
            local_publish_dir=publish_dir,
            commits=commit_messages,
        )

    def _prepare_publish_directory(self, context: BuildContext, publish_dir: Path, repo_url: str) -> None:
        if publish_dir.exists():
            shutil.rmtree(publish_dir)
        publish_dir.mkdir(parents=True, exist_ok=True)

        prototype_dir = context.implementation.prototype_dir
        shutil.copytree(prototype_dir, publish_dir / "prototype")
        shutil.copy2(context.build_summary_path, publish_dir / "build_summary.json")
        shutil.copy2(context.final_report_path, publish_dir / "final_report.md")
        (publish_dir / "iterations").mkdir(parents=True, exist_ok=True)
        (publish_dir / "README.md").write_text(self._render_repo_readme(context, repo_url), encoding="utf-8")

    def _create_commit_history(self, context: BuildContext, publish_dir: Path) -> list[str]:
        commit_messages: list[str] = []
        self.command_runner(["git", "init"], cwd=publish_dir)
        self.command_runner(["git", "checkout", "-b", DEFAULT_BRANCH], cwd=publish_dir)
        self.command_runner(["git", "config", "user.name", "JustBuild Bot"], cwd=publish_dir)
        self.command_runner(["git", "config", "user.email", "justbuild-bot@example.com"], cwd=publish_dir)

        numeric_iterations = [entry for entry in context.iterations if isinstance(entry.get("iteration"), int)]
        if not numeric_iterations:
            numeric_iterations = [{"iteration": 1, "events": []}]
        if not context.request.github_commit_per_iteration:
            numeric_iterations = [numeric_iterations[0]]

        iteration_history_path = publish_dir / "ITERATION_HISTORY.md"
        history_lines = ["# Iteration History", ""]
        for index, iteration in enumerate(numeric_iterations, start=1):
            iteration_number = int(iteration["iteration"])
            iteration_path = publish_dir / "iterations" / f"iteration-{iteration_number:02d}.json"
            iteration_path.write_text(json.dumps(iteration, indent=2), encoding="utf-8")
            history_lines.extend(self._iteration_history_lines(iteration_number, iteration))
            iteration_history_path.write_text("\n".join(history_lines).strip() + "\n", encoding="utf-8")
            if index == 1:
                self.command_runner(
                    ["git", "add", "README.md", "prototype", "iterations", "ITERATION_HISTORY.md"],
                    cwd=publish_dir,
                )
            else:
                self.command_runner(
                    ["git", "add", str(iteration_path.relative_to(publish_dir)), "ITERATION_HISTORY.md"],
                    cwd=publish_dir,
                )
            message = self._commit_message(iteration_number, iteration, first=index == 1)
            self.command_runner(["git", "commit", "-m", message], cwd=publish_dir)
            commit_messages.append(message)

        self.command_runner(["git", "add", "build_summary.json", "final_report.md"], cwd=publish_dir)
        final_message = "docs: add final build summary and report"
        self.command_runner(["git", "commit", "-m", final_message], cwd=publish_dir)
        commit_messages.append(final_message)
        return commit_messages

    def _push_publish_directory(self, publish_dir: Path, clone_url: str) -> None:
        self.command_runner(["git", "remote", "add", "origin", clone_url], cwd=publish_dir)
        try:
            self.command_runner(["gh", "auth", "setup-git"], cwd=publish_dir)
        except GitHubPublishError:
            pass
        self.command_runner(["git", "push", "-u", "origin", DEFAULT_BRANCH], cwd=publish_dir)

    def _render_repo_readme(self, context: BuildContext, repo_url: str) -> str:
        specification = context.specification
        testing = context.testing
        evaluation = context.evaluation
        title = specification.title if specification else context.request.product_idea
        risk_line = evaluation.risk_assessment[0] if evaluation and evaluation.risk_assessment else "Risk assessment unavailable."
        debt_line = evaluation.technical_debt[0] if evaluation and evaluation.technical_debt else "Technical debt summary unavailable."
        return f"""# {title}

Generated by JustBuild from the idea:

> {context.request.product_idea}

- Published repository: {repo_url}
- Prototype directory: `prototype/`
- Test status: {"PASS" if testing and testing.passed else "FAIL"}
- Iterations: {len(context.iterations)}

## Included Artifacts

- `prototype/`
- `build_summary.json`
- `final_report.md`
- `iterations/`

## Build Snapshot

- Risk assessment: {risk_line}
- Technical debt: {debt_line}
"""

    def _iteration_history_lines(self, iteration_number: int, iteration: dict[str, object]) -> list[str]:
        lines = [f"## Iteration {iteration_number}"]
        for event in iteration.get("events", []):
            lines.append(f"- {json.dumps(event, default=str)}")
        if len(lines) == 1:
            lines.append("- No events recorded.")
        lines.append("")
        return lines

    def _commit_message(self, iteration_number: int, iteration: dict[str, object], first: bool) -> str:
        event_blob = json.dumps(iteration.get("events", []), default=str).lower()
        if first:
            return "feat: initial prototype generation"
        if "debugging" in event_blob or "fix_plan" in event_blob:
            return f"fix: iteration {iteration_number} debug-guided refinement"
        if "failure" in event_blob:
            return f"fix: iteration {iteration_number} failure-driven update"
        return f"chore: record iteration {iteration_number} build progress"


def _run_command(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise GitHubPublishError(f"Required command not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or exc.stdout.strip()
        raise GitHubPublishError(stderr or f"Command failed: {' '.join(cmd)}") from exc
    return completed.stdout.strip()
