"""The list of training libraries kiln can use.

To add one: write a Backend subclass in its own file here, then add it to BACKENDS.
"""

from __future__ import annotations

from kiln import KilnError
from kiln.train.backends.base import Backend
from kiln.train.backends.rfdetr import RFDETRBackend

# config `backend = "..."` -> class that does the work
BACKENDS: dict[str, type[Backend]] = {
    "rfdetr": RFDETRBackend,
}


def get_backend(name: str) -> Backend:
    if name not in BACKENDS:
        raise KilnError(f"Unknown backend '{name}'. Available: {', '.join(BACKENDS)}")
    return BACKENDS[name]()
