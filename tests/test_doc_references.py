"""Check conventions for Sphinx cross-reference roles used in docstrings and ``.rst`` files.

This scans ``src/pykeen`` and ``docs/source`` for roles like ``:class:`pykeen.models.TransE``` and checks two
things for every fully-qualified ``pykeen.*`` target:

1. It uses the short-form ``~`` prefix (``:class:`~pykeen.models.TransE```), so rendered prose shows just
   ``TransE`` instead of the full dotted path. ``:mod:`` is exempt from this: module paths are kept in their
   long form (``:mod:`pykeen.triples.splitting```) since the short name alone (``splitting``) is often
   ambiguous or uninformative out of context.
2. The target actually resolves via a plain Python import + attribute lookup.

The second check is a cheap proxy, *not* a substitute for an actual ``sphinx -n`` (nitpicky) build. A target can
import fine in Python and still have no anchor in the rendered docs -- e.g. bare ``__init__``/``__repr__``
methods (folded into their class' docs via ``autoclass_content = "both"``), or ``ClassVar`` attributes that are
never individually documented. Those cases belong in ``nitpick_ignore``/``nitpick_ignore_regex`` in
``docs/source/conf.py``, or as a plain double-backtick literal in the source -- not something this test can
distinguish from a genuine typo. Treat a pass here as "not an obvious typo", and an actual nitpicky doc build as
the authoritative check.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src" / "pykeen"
DOCS = ROOT / "docs" / "source"

#: matches e.g. ``:class:`~pykeen.models.TransE``` or ``:py:attr:`pykeen.models.Model.loss_default```
ROLE_PATTERN = re.compile(
    r":(?:py:)?(?P<role>class|func|meth|attr|data|const|mod|obj|exc):`(?P<tilde>~?)(?P<target>[\w.]+)`"
)


def _iter_role_usages():
    """Yield ``(path, lineno, role, tilde, target)`` for every Sphinx role found in ``src`` and ``docs``."""
    for pattern, base in [("*.py", SRC), ("*.rst", DOCS)]:
        for path in sorted(base.rglob(pattern)):
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                for match in ROLE_PATTERN.finditer(line):
                    yield (
                        path.relative_to(ROOT),
                        lineno,
                        match.group("role"),
                        match.group("tilde"),
                        match.group("target"),
                    )


def _collect_pykeen_targets():
    """Collect fully-qualified ``pykeen.*`` role usages, deduplicated by (target, tilde)."""
    seen = {}
    for path, lineno, role, tilde, target in _iter_role_usages():
        if not target.startswith("pykeen.") or "." not in target[len("pykeen.") :]:
            # skip bare `pykeen` and single-segment targets like `pykeen.env`, which are already short
            continue
        key = (target, role)
        seen.setdefault(key, (path, lineno, tilde))
    return sorted((target, role, *info) for (target, role), info in seen.items())


def _resolves(target: str) -> bool:
    """Return whether ``target`` (dotted path, no leading ``~``) resolves via import + getattr."""
    parts = target.split(".")
    for split in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split])
        try:
            obj = importlib.import_module(module_name)
        except ImportError:
            continue
        try:
            for attr in parts[split:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return False
        return True
    return False


PYKEEN_TARGETS = _collect_pykeen_targets()


@pytest.mark.parametrize(
    ("target", "role", "path", "lineno", "tilde"),
    PYKEEN_TARGETS,
    ids=[f"{path}:{lineno}:{target}" for target, role, path, lineno, tilde in PYKEEN_TARGETS],
)
def test_pykeen_reference_prefers_short_form(target, role, path, lineno, tilde):
    """A fully-qualified pykeen.* role should use ``~`` so prose renders the short name (except ``:mod:``)."""
    if role == "mod":
        assert tilde == "", f"{path}:{lineno}: :mod:`~{target}` should be :mod:`{target}` (long form)"
    else:
        assert tilde == "~", (
            f"{path}:{lineno}: :{role}:`{target}` should be :{role}:`~{target}` so it renders as the short name"
        )


@pytest.mark.parametrize(
    ("target", "role", "path", "lineno", "tilde"),
    PYKEEN_TARGETS,
    ids=[f"{path}:{lineno}:{target}" for target, role, path, lineno, tilde in PYKEEN_TARGETS],
)
def test_pykeen_reference_import_resolves(target, role, path, lineno, tilde):
    """A fully-qualified pykeen.* role should point at something importable.

    This does NOT guarantee Sphinx can generate a link to it -- see the module docstring.
    """
    assert _resolves(target), f"{path}:{lineno}: :{role}:`{target}` does not resolve via import + getattr"
