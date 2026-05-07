"""Quick local linter: parse new workflow YAML + validate the keys we expect.

Used only by the verification step in the agent's session — not part of CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / ".github" / "workflows" / "weekly-discovery.yml",
    ROOT / ".github" / "workflows" / "hf-space-deploy.yml",
]


def main() -> int:
    rc = 0
    for path in TARGETS:
        with path.open(encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        # PyYAML decodes the bare-key `on:` as Python True.
        triggers = doc.get(True, doc.get("on"))
        jobs = list(doc.get("jobs", {}).keys())
        print(f"{path.name}: jobs={jobs}, triggers={triggers}")
        if not jobs:
            print(f"  ERROR: {path.name} has no jobs")
            rc = 1
        if triggers is None:
            print(f"  ERROR: {path.name} has no 'on' triggers")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
