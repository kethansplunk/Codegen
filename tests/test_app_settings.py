"""
Phase 20A demo: sidebar settings must not force a pipeline rebuild.

strategy / max_retries / n_candidates are runtime parameters -- the LangGraph
graph reads them when a node executes, not at construction. Keying app.py's
@st.cache_resource on them made every slider nudge reload the 7B generator, so
this pins that they are applied in place instead.
"""

from __future__ import annotations

import types

import app


class _StubGenerator:
    def __init__(self):
        self.n_candidates = 1


def _stub_router():
    r = types.SimpleNamespace(
        strategy="balanced", max_retries=3, generator=_StubGenerator())
    return r


def test_apply_settings_mutates_in_place():
    r = _stub_router()
    before = r.generator
    app._apply_settings(r, "schema_priority", 1, 5)
    assert r.strategy == "schema_priority"
    assert r.max_retries == 1
    assert r.generator.n_candidates == 5
    # Same generator object -- nothing was rebuilt.
    assert r.generator is before


def test_no_bare_expressions_for_streamlit_magic():
    """No standalone expression statements anywhere in app.py.

    Streamlit rewrites the app script's AST and wraps standalone expressions in
    st.write(). The lazy-property warm-up in _get_router() was written as bare
    `router.linker` / `router.sar` / `router.generator` lines, which dumped the
    full ApiSchemaLinker and SARRetriever reprs into the page above the results.
    Calls and docstrings are legitimate; anything else is almost certainly an
    accidental render.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(app.__file__).read_text())
    bare = [n for n in ast.walk(tree)
            if isinstance(n, ast.Expr)
            and not isinstance(n.value, (ast.Call, ast.Constant, ast.Await))]
    assert not bare, ("bare expressions in app.py will be rendered by Streamlit "
                      "magic: " + ", ".join(f"line {n.lineno}: {ast.unparse(n)}"
                                            for n in bare))


def test_get_router_is_keyed_on_track_only():
    # The cache key is the function signature; anything beyond `track` would
    # evict the cached 7B model whenever a slider moved.
    import inspect
    params = list(inspect.signature(app._get_router.__wrapped__).parameters)
    assert params == ["track"], f"_get_router must take only `track`, got {params}"
