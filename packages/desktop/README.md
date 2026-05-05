# pps-desktop

PySide6 desktop client for Pro Photo Studio.

Ports the existing watermark-toolkit desktop app to the new monorepo. Calls
into `pps-core` directly for offline / single-machine workflows; can also
be configured to delegate to a remote `pps-api` instance.

## Run

```bash
uv pip install -e packages/core
uv pip install -e packages/desktop
uv run pps-desktop
```

## Build standalone executable

```bash
uv pip install pyinstaller
pyinstaller deploy/pyinstaller.spec
```

> Status: ported as-is from watermark-toolkit. Phase 4 will refactor the
> UI to talk to `pps-api` so multiple users can share a single GPU
> backend.
