"""CLI entry point for the build loop.

Two modes:
  template_first (default): Productized, narrow. python_cli and fastapi_service.
  freeform: Experimental, broad. Any project type, best-effort.

Supports --resume to restart from a specific phase using saved state.
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from rich.console import Console

from build_loop.modes import BuildMode
from build_loop.agents.architect import ArchitectAgent

console = Console()


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="build-loop: autonomous multi-agent project builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  template_first  Productized build for python_cli and fastapi_service (default)
  freeform        Experimental generalist loop for any project type

Examples:
  build-loop "a CLI to convert CSV to SQL inserts"
  build-loop "a REST API for book reviews" --mode template_first
  build-loop --resume setup -o ~/my-project    # resume from setup phase
  build-loop -f idea.txt -o ./my-project
        """,
    )
    parser.add_argument("idea", nargs="?", help="What to build (plain english)")
    parser.add_argument("--file", "-f", help="Read idea from a text file")
    parser.add_argument("--output", "-o", default="./output", help="Output directory (default: ./output)")
    parser.add_argument(
        "--mode", "-m",
        choices=[m.value for m in BuildMode],
        default=BuildMode.TEMPLATE_FIRST.value,
        help="Build mode: template_first (default) or freeform (experimental)",
    )
    parser.add_argument(
        "--resume",
        choices=["setup", "test", "verify", "accept"],
        help="Resume from a phase using saved state (skips research/contract/plan/build)",
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="Run optimizer pass (off by default)",
    )
    args = parser.parse_args()

    # Get the idea
    if args.resume:
        # Resume doesn't need an idea — uses saved state
        idea = args.resume  # placeholder
    elif args.file:
        with open(args.file) as f:
            idea = f.read().strip()
    elif args.idea:
        idea = args.idea
    else:
        console.print("[bold]What do you want to build?[/bold]")
        idea = input("> ").strip()

    if not idea and not args.resume:
        console.print("[red]No idea provided.[/red]")
        sys.exit(1)

    mode = BuildMode(args.mode)
    if mode == BuildMode.FREEFORM:
        console.print("[bold yellow]Running in freeform (experimental) mode[/bold yellow]")

    architect = ArchitectAgent(
        output_dir=args.output,
        mode=mode,
        run_optimizer=args.optimize,
    )

    if args.resume:
        architect.resume(args.resume)
    else:
        architect.run(idea)

    # Exit non-zero if the pipeline did not succeed.
    state = architect.state
    if state.acceptance is None:
        sys.exit(1)
    if hasattr(state.acceptance.verdict, "value"):
        verdict = state.acceptance.verdict.value
    else:
        verdict = str(state.acceptance.verdict)
    if verdict != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
