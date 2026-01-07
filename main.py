from __future__ import annotations

def main() -> int:
    from vrchat_eidolon.runtime.lifecycle import main as runtime_main

    runtime_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

