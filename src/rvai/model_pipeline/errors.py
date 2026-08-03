"""Errors raised by the offline MobileNetV2 model-production pipeline."""


class ModelPipelineError(RuntimeError):
    """Base error for model-pipeline configuration and deterministic I/O."""


class PipelineIOError(ModelPipelineError):
    """Raised when a deterministic record cannot be loaded or published."""


class PipelineConfigError(ModelPipelineError):
    """Raised when committed pipeline configuration is invalid or inconsistent."""


class PipelinePathError(ModelPipelineError):
    """Raised when a pipeline path violates the repository path policy."""
