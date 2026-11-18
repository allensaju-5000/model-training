"""kiln: dataset and model workflow CLI (load, review, export; training later)."""


class KilnError(Exception):
    """An expected failure with a message meant for the user (no traceback)."""
