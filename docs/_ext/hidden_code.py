"""Sphinx extension: drop the empty code blocks of the executed tutorials.

Sphinx-Gallery removes ``# sphinx_gallery_start_ignore`` / ``end_ignore``
blocks and config comments (``# sphinx_gallery_thumbnail_path = ...``) from
the code it shows, but still writes a ``.. code-block:: Python`` for a code
cell that has nothing left, which renders as an empty box.  A tutorial's
figure cells hide their plotting code entirely ("Writing docs figures" in
docs/contributing.rst), so this hook removes such blocks from the generated
pages before Sphinx reads them; the cell's output (the figure) stays.
"""

from __future__ import annotations

import re
from typing import Any

from sphinx.application import Sphinx

GALLERY_DIR = "auto_examples/"

# A code-block directive, its options, then only blank lines up to the next
# unindented line: the block has no content.
_EMPTY = re.compile(
    r"^\.\. code-block:: Python\n(?:   :[^\n]*\n)*(?:[ \t]*\n)+(?=\S)",
    re.MULTILINE,
)


def _drop_empty_code_blocks(app: Sphinx, docname: str, source: list[str]) -> None:
    if docname.startswith(GALLERY_DIR):
        source[0] = _EMPTY.sub("", source[0])


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("source-read", _drop_empty_code_blocks)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
