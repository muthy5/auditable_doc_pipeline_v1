from __future__ import annotations


class PipelineError(RuntimeError):
    """Raised when pipeline orchestration fails."""


class BackendError(RuntimeError):
    """Raised when a backend cannot produce a valid response."""


class SchemaLoadError(FileNotFoundError):
    """Raised when a schema file cannot be found."""


class ChunkingError(ValueError):
    """Raised when chunk generation cannot proceed."""


class PassSchemaValidationError(RuntimeError):
    """Raised when a pass output fails schema validation in strict mode."""

    def __init__(self, pass_name: str, message: str) -> None:
        """Initialize validation error details.

        Args:
            pass_name: Name of the failing pass.
            message: Human-readable error message.
        """
        super().__init__(message)
        self.pass_name = pass_name
