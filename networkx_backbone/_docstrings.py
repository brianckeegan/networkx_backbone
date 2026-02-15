"""Helpers for enriching public API docstrings."""

import inspect


def append_complexity_docstrings(namespace, complexity_map):
    """Append a standardized Complexity section to function docstrings.

    Parameters
    ----------
    namespace : dict
        Module globals() dictionary containing function objects.
    complexity_map : dict
        Mapping ``{function_name: {"time": ..., "space": ..., "notes": ...}}``.
    """
    for name, spec in complexity_map.items():
        func = namespace.get(name)
        if func is None or not callable(func):
            continue

        raw_doc = getattr(func, "__doc__", None)
        if not raw_doc:
            continue

        # Normalize indentation first so appending new sections does not
        # accidentally change the minimum indentation and break RST parsing.
        doc = inspect.cleandoc(raw_doc)
        if "Complexity:" in doc:
            continue

        def _rst_safe(text):
            return str(text).replace("|", r"\|")

        lines = [
            "",
            (
                "Computational complexity estimate is "
                f"``{_rst_safe(spec['time'])}`` time and "
                f"``{_rst_safe(spec['space'])}`` space."
            ),
        ]

        func.__doc__ = doc.rstrip() + "\n" + "\n".join(lines)
