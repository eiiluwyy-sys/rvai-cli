"""User-facing errors for local language-model generation."""


class GenerationError(RuntimeError):
    """Raised when generation setup, execution, or output is invalid."""
