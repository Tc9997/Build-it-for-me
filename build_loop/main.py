"""CLI entry point for the build loop."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

from build_loop.agents.architect import ArchitectAgent

console = Console()


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="build-loop: multi-agent project builder")
    parser.add_argument("idea", nargs="?", help="Project idea (or pass via --file)")
    parser.add_argument("--file", "-f", help="Read idea from a file")
    parser.add_argument("--output", "-o", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode (answer clarifying questions)")
    args = parser.parse_args()

    # Get the idea
    if args.file:
        with open(args.file) as f:
            idea = f.read().strip()
    elif args.idea:
        idea = args.idea
    else:
        idea = Prompt.ask("[bold]What do you want to build?[/bold]")

    if not idea:
        console.print("[red]No idea provided.[/red]")
        sys.exit(1)

    architect = ArchitectAgent(output_dir=args.output)

    if args.interactive:
        # Interactive: ask clarifying questions, wait for answers
        questions = architect.clarify(idea)
        clarifications = ""
        if questions:
            console.print("\n[bold yellow]The architect has some questions:[/bold yellow]")
            answers = []
            for q in questions:
                ans = Prompt.ask(f"  [cyan]?[/cyan] {q}")
                answers.append(f"Q: {q}\nA: {ans}")
            clarifications = "\n\n".join(answers)

        architect.plan(clarifications)

        console.print("\n[bold]Build plan ready. Proceed? (y/n)[/bold]")
        if Prompt.ask("", choices=["y", "n"], default="y") != "y":
            console.print("[dim]Aborted.[/dim]")
            sys.exit(0)

        architect.build()
        architect.integrate()
        output = architect.write_project()
    else:
        # Non-interactive: run everything end-to-end
        output = architect.run(idea)

    console.print(f"\n[bold green]Project written to: {output}[/bold green]")


if __name__ == "__main__":
    main()
