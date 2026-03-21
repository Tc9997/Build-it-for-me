"""Build modes: template_first and freeform.

template_first: Productized, narrow. Two archetypes (python_cli, fastapi_service).
  Typed contract, deterministic policy, pinned templates, ownership manifest,
  verifier authority. Reliability-focused.

freeform: Experimental, broad. Autonomous generalist loop for any project type.
  Best-effort. Useful for exploration, fallback research, and benchmarking.
  Not the product promise.
"""

from enum import Enum


class BuildMode(str, Enum):
    TEMPLATE_FIRST = "template_first"
    FREEFORM = "freeform"
