"""Container backends: how the harness reaches an instance's container.

``get_backend("docker")`` is the historical behaviour; ``get_backend("apptainer")``
runs the same instance image with Apptainer on a machine that has no Docker daemon.
"""

from __future__ import annotations

from swebench.harness.container_backends.base import (
    Container,
    ContainerBackend,
    ExecResult,
)
from swebench.harness.container_backends.docker import DockerBackend

BACKENDS = ("docker", "apptainer")


def get_backend(name: str = "docker", **kwargs):
    """Build a backend by name (``docker`` | ``apptainer``)."""
    name = (name or "docker").lower()
    if name == "docker":
        return DockerBackend(**kwargs)
    if name == "apptainer":
        from swebench.harness.container_backends.apptainer import ApptainerBackend

        kwargs.pop("client", None)
        return ApptainerBackend(**kwargs)
    raise ValueError(
        f"unknown container backend {name!r}; choose one of {', '.join(BACKENDS)}"
    )


__all__ = [
    "BACKENDS",
    "ApptainerBackend",
    "Container",
    "ContainerBackend",
    "DockerBackend",
    "ExecResult",
    "get_backend",
]
