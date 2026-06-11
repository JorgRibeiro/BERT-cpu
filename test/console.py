"""Shared console styling for the didactic test walkthroughs.

The whole test suite uses a single colour scheme, organised by *content
category* (not by colour) so that restyling a category is a one-line change.
Everything is encapsulated in the :class:`Console` class and exposed as the
module-level singleton :data:`console`, which any test file can import::

    from test.console import console

    print(console.label("name"), console.fmt(some_array))

Categories
----------
``text``   prose / explanatory sentences
``math``   variables and equations (x, w, y = tanh(w . x), dy/dz, ...)
``label``  reserved keywords (name, value, grad, STEP)
``value``  numeric values and matrices (the actual numbers)
"""

from __future__ import annotations

import os

import numpy as np


class Console:
    """Content-category colour scheme plus small formatting helpers.

    To restyle a category, change only its ANSI SGR code in :attr:`COLOR`
    (https://en.wikipedia.org/wiki/ANSI_escape_code#SGR). An empty code leaves
    the text in the terminal's default colour. ``NO_COLOR`` disables all colour.
    """

    #: ANSI SGR code per content category (the single source of truth).
    COLOR = {
        "text": "",                       # default / white
        "math": "1;31",                   # bold red
        "label": "38;2;217;119;87",       # Anthropic orange (#D97757, truecolor)
        "value": "1;38;2;189;147;249",    # bold Dracula violet (#BD93F9, truecolor)
    }

    # ------------------------------------------------------------------ #
    # Colouring
    # ------------------------------------------------------------------ #
    def paint(self, category: str, text: str) -> str:
        """Wrap ``text`` in the ANSI code registered for ``category``.

        No-op when the category has no code or when ``NO_COLOR`` is set, so the
        output stays free of escape codes when redirected to a file.
        """
        code = self.COLOR[category]
        if not code or os.environ.get("NO_COLOR"):
            return text
        return f"\033[{code}m{text}\033[0m"

    def text(self, s: str) -> str:
        """Colour prose / explanatory text."""
        return self.paint("text", s)

    def math(self, s: str) -> str:
        """Colour maths: variables and equations."""
        return self.paint("math", s)

    def label(self, s: str) -> str:
        """Colour the reserved keywords (``name``, ``value``, ``grad``, ``STEP``)."""
        return self.paint("label", s)

    def value(self, s: str) -> str:
        """Colour numeric values and matrices."""
        return self.paint("value", s)

    # ------------------------------------------------------------------ #
    # Number formatting
    # ------------------------------------------------------------------ #
    def fmt_plain(self, value: np.ndarray) -> str:
        """Compact, sign-aligned formatting of a scalar or small array (no colour).

        Returning the uncoloured string lets callers measure its real width for
        table alignment before wrapping it in ANSI codes.
        """
        arr = np.asarray(value)
        if arr.ndim == 0:
            return f"{float(arr):+.3f}"
        return "[" + ", ".join(f"{v:+.3f}" for v in arr.ravel()) + "]"

    def fmt(self, value: np.ndarray) -> str:
        """Like :meth:`fmt_plain`, but coloured as the ``value`` category."""
        return self.value(self.fmt_plain(value))

    # ------------------------------------------------------------------ #
    # Printing
    # ------------------------------------------------------------------ #
    def kv(self, label: str, value: str, note: str = "", color=None) -> None:
        """Print one ``label: value`` line in the colour scheme.

        ``label`` is printed with ``color`` (the ``text`` category by default,
        or :meth:`math` for equations) and any trailing ``note`` with ``text``,
        while ``value`` is expected to already carry its own colour (e.g. from
        :meth:`fmt`). Reproduces the pattern::

            print(color(label), end="")
            print(value)
        """
        color = color or self.text
        print(color(label), end="")
        print(value, end="")
        print(self.text(note) if note else "")


#: Module-level singleton shared by every test file.
console = Console()
