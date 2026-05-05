#!/usr/bin/env python3
"""Pre-commit hook: reject any source file containing common secret patterns.

Patterns checked (case-insensitive where appropriate):
    hf_<32-40 chars>           HuggingFace token
    sl\\.u\\.[A-Za-z0-9_-]{60,} Dropbox long-lived token
    AKIA[0-9A-Z]{16}           AWS access key id
    ghp_[A-Za-z0-9]{36}        GitHub personal access token
    gho_[A-Za-z0-9]{36}        GitHub OAuth token
    sk-[A-Za-z0-9]{32,}        OpenAI / Anthropic-style API key
    AIza[0-9A-Za-z_-]{35}      Google API key
    -----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----  any PEM private key

Allowed locations:
    SECURITY.md         (incident-history references — must use prefix only)
    .env.example        (placeholder values)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("HuggingFace token", re.compile(r"\bhf_[A-Za-z0-9]{32,40}\b")),
    ("Dropbox long token", re.compile(r"\bsl\.u\.[A-Za-z0-9_-]{60,}")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub PAT", re.compile(r"\bghp_[A-Za-z0-9]{36}\b")),
    ("GitHub OAuth", re.compile(r"\bgho_[A-Za-z0-9]{36}\b")),
    ("OpenAI/Anthropic key", re.compile(r"\bsk-[A-Za-z0-9_-]{32,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("PEM private key", re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----")),
]

ALLOWLIST = {
    "SECURITY.md",
    ".env.example",
    "scripts/check_no_secrets.py",  # This very file references the patterns.
}


def main(argv: list[str]) -> int:
    failures: list[tuple[Path, str, str]] = []
    for arg in argv[1:]:
        path = Path(arg)
        if not path.is_file():
            continue
        rel = path.as_posix()
        if any(rel.endswith(allow) for allow in ALLOWLIST):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in SECRET_PATTERNS:
            match = pattern.search(text)
            if match:
                failures.append((path, label, match.group(0)[:40] + "..."))
    if failures:
        print("\n[secret-scan] Possible secrets detected — refusing commit:\n")
        for path, label, snippet in failures:
            print(f"  {path}: {label}: {snippet}")
        print("\nIf this is a false positive, add the path to ALLOWLIST in")
        print("scripts/check_no_secrets.py and document the exception.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
