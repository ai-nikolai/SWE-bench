"""The Docker backend: the behaviour the harness always had, behind the interface.

Everything here forwards to the pre-existing ``docker_utils`` helpers, so a default
run is byte-for-byte the run it was before the interface existed. That is also what
the regression test asserts: the same helpers, in the same order, with the same
arguments.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from swebench.harness.container_backends.base import Container, ExecResult
from swebench.harness.docker_utils import (
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
)


class _DockerContainer:
    """Wrap a ``docker`` container so the harness sees our ``ExecResult``."""

    def __init__(self, container):
        self._container = container
        self.id = container.id

    def start(self) -> None:
        self._container.start()

    def exec_run(self, command, workdir=None, user=None) -> ExecResult:
        result = self._container.exec_run(command, workdir=workdir, user=user)
        return ExecResult(exit_code=result.exit_code, output=result.output)

    def __getattr__(self, item):
        # `create_container` and the log path still want `.short_id`, `.reload()`
        # and friends; forward everything we do not wrap.
        return getattr(self._container, item)


class DockerBackend:
    name = "docker"

    def __init__(self, client=None):
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from swebench.harness.run_evaluation import _docker_client

            self._client = _docker_client()
        return self._client

    def create(self, test_spec, run_id: str, logger) -> Container:
        # Imported lazily: `run_evaluation` imports this module, and importing it
        # back at module load would be circular.
        from swebench.harness.run_evaluation import create_container

        return _DockerContainer(
            create_container(test_spec, self.client, run_id, logger)
        )

    def copy(self, container: Container, source, target: PurePosixPath) -> None:
        copy_to_container(_unwrap(container), source, target)

    def exec_with_timeout(self, container: Container, command: str, timeout):
        return exec_run_with_timeout(_unwrap(container), command, timeout)

    def cleanup(self, container: Container | None, logger) -> None:
        cleanup_container(
            self.client, _unwrap(container) if container is not None else None, logger
        )

    def close(self) -> None:
        # The client is shared with the reporting code, which still uses it.
        pass


def _unwrap(container):
    return container._container if isinstance(container, _DockerContainer) else container
