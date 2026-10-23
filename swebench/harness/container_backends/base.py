"""A container backend interface for the evaluation harness.

``run_evaluation`` used to talk to ``docker-py`` directly: create a container,
copy the patch and the eval script into it, run ``/eval.sh``, copy the log out,
delete the container. Everything that decides a *score* -- ``grading``,
``make_test_spec``, the report format -- is already backend agnostic, so the only
thing that has to change for HPC clusters (Apptainer, no daemon) is this
transport layer.

The interface is deliberately the smallest set of operations ``run_instance``
needs, and each one keeps the semantics the Docker implementation had, so the
Docker backend is a thin adapter over the existing ``docker_utils`` helpers and
behaves exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable


@dataclass
class ExecResult:
    """The subset of ``docker.models.containers.ExecResult`` the harness uses."""

    exit_code: int
    output: bytes


@runtime_checkable
class Container(Protocol):
    """One running (or startable) instance container."""

    #: Human-readable identifier, for logs only.
    id: str

    def start(self) -> None:
        """Start the container (Docker creates it stopped; Apptainer is stateless)."""


class ContainerBackend(Protocol):
    """Where an instance's tests run."""

    #: ``docker`` or ``apptainer`` -- recorded in logs and the report.
    name: str

    def create(self, test_spec, run_id: str, logger) -> Container:
        """Return a container for ``test_spec`` (started later by ``.start()``)."""

    def copy(self, container: Container, source, target: PurePosixPath) -> None:
        """Copy a host file to ``target`` inside the container."""

    def exec_with_timeout(self, container: Container, command: str, timeout: int | None):
        """Run ``command``; return ``(output, timed_out, seconds)``."""

    def cleanup(self, container: Container | None, logger) -> None:
        """Remove the container (and, for Docker, its image)."""

    def close(self) -> None:
        """Release backend-wide resources (a Docker client, a cache lock)."""
