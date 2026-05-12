"""One-shot smoke: pull 1 FiveK row to verify mirror + token + decode path."""

from __future__ import annotations

import itertools
import sys

from pps_data.loaders.fivek import stream_fivek


def main() -> int:
    try:
        it = stream_fivek(expert="c")
        rows = list(itertools.islice(it, 1))
    except Exception as exc:  # noqa: BLE001
        print("FAIL:", type(exc).__name__, str(exc)[:400])
        return 1
    print(f"OK rows fetched: {len(rows)}")
    for row in rows:
        print(f"  keys: {list(row.keys())}")
        for k, v in row.items():
            tname = type(v).__name__
            extra = ""
            for attr in ("size", "shape", "mode", "format"):
                if hasattr(v, attr):
                    extra += f" {attr}={getattr(v, attr)}"
            if not extra:
                extra = f" val={str(v)[:80]}"
            print(f"    {k}: {tname}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
