from __future__ import annotations


class PipelineError(Exception):
    """Base exception for all pipeline-specific failures."""


class PassValidationError(PipelineError):
    """Raised when data fails JSON schema or structural validation."""


class SchemaLoadError(PipelineError):
    """Raised when a schema file cannot be loaded or parsed."""


class BackendError(PipelineError):
    """Raised when a configured model backend fails or is misconfigured."""


class ChunkingError(PipelineError):
    """Raised when input cannot be chunked into valid document chunks."""
