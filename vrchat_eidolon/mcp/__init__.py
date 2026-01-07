"""Compatibility shim for the third-party `mcp` package.

This repository depends on the upstream `mcp` SDK. Earlier iterations of the
project also used a local `src/mcp` package for internal integration code, which
causes import-name collisions.

The internal integration layer now lives under `eidolon_mcp`.

To minimize surprises when `src/` ends up early on `sys.path` (e.g. developer
workflows, ad-hoc scripts), this module forwards `import mcp` to the third-party
package.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path


def _load_third_party_mcp() -> None:
	"""Load the third-party `mcp` package, bypassing this local source tree."""

	pkg_root = Path(__file__).resolve().parent
	src_root = pkg_root.parent

	search_path: list[str] = []
	for p in sys.path:
		# Exclude our local `src/` root, so we don't resolve back to `src/mcp`.
		try:
			if Path(p).resolve() == src_root:
				continue
		except Exception:
			# If a path entry can't be resolved (rare), keep it.
			pass
		search_path.append(p)

	spec = importlib.machinery.PathFinder.find_spec(__name__, search_path)
	if spec is None or spec.loader is None:
		raise ImportError(
			"Third-party 'mcp' package not found. "
			"Install dependencies (e.g. 'uv sync') and avoid using local 'src/mcp'."
		)

	module = importlib.util.module_from_spec(spec)
	# Replace ourselves in sys.modules before executing to prevent recursion.
	sys.modules[__name__] = module
	spec.loader.exec_module(module)
	globals().update(module.__dict__)


_load_third_party_mcp()

