"""CLI entry point — builder fills in commands and argument parsing."""

import argparse
import sys


def main():
    """Main entry point for {{project_name}}."""
    parser = argparse.ArgumentParser(description="{{summary}}")
    # TODO: builder adds arguments and subcommands here
    args = parser.parse_args()
    return 0


if __name__ == "__main__":
    sys.exit(main())
