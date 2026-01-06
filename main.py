from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if src.exists():
        # Append (not prepend) to avoid shadowing third-party packages.
        sys.path.append(str(src))


def main() -> int:
    _ensure_src_on_path()
    from runtime.lifecycle import main as runtime_main

    runtime_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

