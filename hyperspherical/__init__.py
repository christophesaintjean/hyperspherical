import sys

MIN_PYTHON = (3, 9)
MAX_PYTHON = (3, 13)
CURRENT_PYTHON = sys.version_info[:2]

if not (MIN_PYTHON <= CURRENT_PYTHON <= MAX_PYTHON):
    raise RuntimeError(
        f"Hyperspherical requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
        f"to {MAX_PYTHON[0]}.{MAX_PYTHON[1]}, "
        f"but you are using Python {sys.version_info[0]}.{sys.version_info[1]}."
    )

__version__ = "0.1.0"
