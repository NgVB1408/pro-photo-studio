"""Entry point cho PyInstaller bundle. Multiplex CLI subcommands.

Usage:
    wincei-stack.exe                          → start API server
    wincei-stack.exe api                      → start API server
    wincei-stack.exe wincei foto.jpg          → window+ceiling fix
    wincei-stack.exe hdr --inputs ./raw       → HDR fuse
    wincei-stack.exe masks foto.jpg           → segmentation
    wincei-stack.exe regions foto.jpg         → JSON regions
"""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("api", "serve"):
        # Default: start API server
        from .server import run
        # Strip subcommand arg
        if len(sys.argv) >= 2 and sys.argv[1] in ("api", "serve"):
            sys.argv.pop(1)
        run()
        return 0

    cmd = sys.argv.pop(1)

    if cmd == "wincei":
        from pps_wincei.cli import main as wincei_main
        return wincei_main()
    if cmd == "hdr":
        from pps_wincei_hdr.cli import main as hdr_main
        return hdr_main()
    if cmd in ("masks", "folder"):
        if cmd == "folder":
            from pps_wincei.folder_cli import main as folder_main
            return folder_main()
        from pps_wincei_masks.cli import main as masks_main
        return masks_main()
    if cmd == "regions":
        from pps_wincei_masks.cli_regions import main as regions_main
        return regions_main()

    print(f"Unknown subcommand: {cmd}", file=sys.stderr)
    print("Usage: wincei-stack.exe [api|wincei|hdr|masks|folder|regions] ...", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
