"""Helpers for enriching public API docstrings."""


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

        doc = getattr(func, "__doc__", None)
        if not doc:
            continue
        if "Complexity\n----------" in doc:
            continue

        lines = [
            "",
            "Complexity",
            "----------",
            f"Time: {spec['time']}",
            f"Space: {spec['space']}",
        ]
        notes = spec.get("notes")
        if notes:
            lines.append(f"Notes: {notes}")

        func.__doc__ = doc.rstrip() + "\n" + "\n".join(lines)
