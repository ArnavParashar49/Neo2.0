"""The old name of NEO's own package, kept for a checkout: it moved to ``neo2.core`` in 2.1.

neo2-codemod: shim (in the repository only: the ``neo2`` wheel never ships it)

``import neo`` gives ``neo2.core`` itself, and ``import neo.tools.web`` the very module
``neo2.core.tools.web`` (one module under both names, never a second copy), so an old script, a
scratch test or a branch not yet rewritten keeps working. ``python -m neo …`` starts NEO.app's core
like ``python -m neo2 …``, and ``python -m neo.<module>`` runs ``neo2.core.<module>``. The first
import warns once (DeprecationWarning). ``python scripts/codemod_neo2.py`` rewrites old imports.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import warnings
from types import CodeType, ModuleType

_OLD, _NEW = __name__, "neo2.core"
_HERE = os.path.dirname(os.path.abspath(__file__))


class _Alias(importlib.abc.InspectLoader):
    """Loads ``neo.x.y`` as the module ``neo2.core.x.y`` itself (imported under its real name)."""

    def __init__(self, target: str) -> None:
        self.target = target
        self._spec: importlib.machinery.ModuleSpec | None = None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = importlib.import_module(self.target)
        self._spec = getattr(module, "__spec__", None)
        return module

    def exec_module(self, module: ModuleType) -> None:
        # The import system gave the module the alias's spec: it keeps its own (its name, reloads).
        module.__spec__ = self._spec

    # runpy (python -m neo.x.y): the real module's code, run as __main__
    def _real(self) -> importlib.machinery.ModuleSpec:
        spec = importlib.util.find_spec(self.target)
        if spec is None or spec.loader is None:
            raise ImportError(f"No module named {self.target!r}", name=self.target)
        return spec

    def get_code(self, fullname: str) -> CodeType | None:
        return self._real().loader.get_code(self.target)  # type: ignore[union-attr]

    def get_source(self, fullname: str) -> str | None:
        return self._real().loader.get_source(self.target)  # type: ignore[union-attr]

    def is_package(self, fullname: str) -> bool:
        return self._real().submodule_search_locations is not None


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(
        self, fullname: str, path: object = None, target: object = None
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(_OLD + "."):
            return None
        if fullname == _OLD + ".__main__":  # python -m neo: this folder's __main__.py
            return importlib.util.spec_from_file_location(fullname, os.path.join(_HERE, "__main__.py"))
        new = _NEW + fullname[len(_OLD) :]
        if new in sys.modules:  # imported already (or stood in for, or blocked with None: an ImportError)
            module = sys.modules[new]
            if module is None:
                raise ModuleNotFoundError(f"import of {new} halted; None in sys.modules", name=fullname)
            origin, package = getattr(module, "__file__", None), hasattr(module, "__path__")
        else:
            real = importlib.util.find_spec(new)
            if real is None:
                return None
            origin, package = real.origin, real.submodule_search_locations is not None
        return importlib.machinery.ModuleSpec(fullname, _Alias(new), origin=origin, is_package=package)


_message = (
    "the `neo` package is now `neo2.core` (and `python -m neo` is `python -m neo2`): "
    "`neo` works only from a checkout; run scripts/codemod_neo2.py to rewrite old imports"
)
if sys.argv[:1] == ["-m"]:  # python -m neo…: shown, as for a script's own warnings
    warnings.warn_explicit(_message, DeprecationWarning, __file__, 1, module="__main__")
else:
    warnings.warn(_message, DeprecationWarning, stacklevel=2)

if not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())  # before the path finder: never a second copy of a module
sys.modules[_OLD] = importlib.import_module(_NEW)  # `neo` is neo2.core itself from here on
