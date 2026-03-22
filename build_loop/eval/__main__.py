"""Eval harness CLI.

Usage:
  python -m build_loop.eval                          # Run all tasks in both modes
  python -m build_loop.eval --mode template_first    # One mode only
  python -m build_loop.eval --task python_cli_01     # Single task
  python -m build_loop.eval --archetype python_cli   # Filter by archetype
  python -m build_loop.eval --output results.json    # Save results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from build_loop.eval import corpus_loader
from build_loop.eval.models import EvalSuiteResult
from build_loop.eval.reporter import print_comparison, save_results
from build_loop.eval.runner import run_suite
from build_loop.modes import BuildMode


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="build-loop eval harness")
    parser.add_argument("--mode", "-m", choices=["template_first", "freeform", "both"],
                        default="both", help="Which mode(s) to eval")
    parser.add_argument("--task", "-t", help="Run a single task by ID")
    parser.add_argument("--archetype", "-a", help="Filter tasks by archetype")
    parser.add_argument("--output", "-o", default="eval_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--output-dir", "-d", default="/tmp/build-loop-eval",
                        help="Base directory for task outputs")
    args = parser.parse_args()

    # Load tasks
    if args.task:
        task = corpus_loader.load_by_id(args.task)
        if not task:
            print(f"Task not found: {args.task}")
            sys.exit(1)
        tasks = [task]
    elif args.archetype:
        tasks = corpus_loader.load_by_archetype(args.archetype)
    else:
        tasks = corpus_loader.load_all()

    if not tasks:
        print("No tasks to run.")
        sys.exit(1)

    print(f"Loaded {len(tasks)} eval task(s)")

    # Determine modes
    modes = []
    if args.mode in ("template_first", "both"):
        modes.append(BuildMode.TEMPLATE_FIRST)
    if args.mode in ("freeform", "both"):
        modes.append(BuildMode.FREEFORM)

    # Run
    output_base = Path(args.output_dir)
    suites: list[EvalSuiteResult] = []

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  EVAL: {mode.value} ({len(tasks)} tasks)")
        print(f"{'='*60}")
        suite = run_suite(tasks, mode, output_base / mode.value)
        suites.append(suite)

    # Report
    print_comparison(suites)
    save_results(suites, Path(args.output))


if __name__ == "__main__":
    main()
