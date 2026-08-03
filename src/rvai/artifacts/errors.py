"""User-facing artifact cache and download exceptions."""


class ArtifactError(Exception):
    """Base exception for model artifact operations."""


class ArtifactNotDeclaredError(ArtifactError):
    """Raised when a Manifest does not declare an artifact."""


class ArtifactNotCachedError(ArtifactError):
    """Raised when a declared artifact is not present in the cache."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when cached or downloaded bytes fail verification."""


class ArtifactDownloadError(ArtifactError):
    """Raised when an HTTP download cannot be completed."""


class ArtifactCacheError(ArtifactError):
    """Raised when cache paths or metadata cannot be used safely."""
