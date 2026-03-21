"""CLI entry point for the build loop."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from rich.console import Console

from build_loop.agents.architect import ArchitectAgent

console = Console()


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="build-loop: autonomous multi-agent project builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  build-loop "a CLI to convert CSV to SQL inserts"
  build-loop "scrape wine auctions and recommend based on my taste" -o ~/wine-bot
  build-loop "find smart devices on my LAN and control via WhatsApp" -o ~/home-control
  build-loop -f idea.txt -o ./my-project
        """,
    )
    parser.add_argument("idea", nargs="?", help="What to build (plain english)")
    parser.add_argument("--file", "-f", help="Read idea from a text file")
    parser.add_argument("--output", "-o", default="./output", help="Output directory (default: ./output)")
    args = parser.parse_args()

    # Get the idea
    if args.file:
        with open(args.file) as f:
            idea = f.read().strip()
    elif args.idea:
        idea = args.idea
    else:
        console.print("[bold]What do you want to build?[/bold]")
        idea = input("> ").strip()

    if not idea:
        console.print("[red]No idea provided.[/red]")
        sys.exit(1)

    architect = ArchitectAgent(output_dir=args.output)
    architect.run(idea)


if __name__ == "__main__":
    main()
