"""Make the `lane_perception` package importable without installation.

Scripts `import _bootstrap` (first thing) to add ../src to sys.path.
"""

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
