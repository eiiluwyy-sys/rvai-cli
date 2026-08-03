"""User-facing errors for optional model inference features."""


class InferenceError(RuntimeError):
    """Base error for inference setup, input, runtime, or output failures."""


class InferenceDependencyError(InferenceError):
    """Raised when the optional ONNX inference dependencies are unavailable."""


class InferenceInputError(InferenceError):
    """Raised when an inference input cannot be decoded or prepared."""


class InferenceOutputError(InferenceError):
    """Raised when runtime output cannot be interpreted safely."""
