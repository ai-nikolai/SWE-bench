"""Run SWE-bench instance containers with Apptainer instead of a Docker daemon.

An instance image is public OCI (``swebench/sweb.eval.x86_64.<id>``, with ``__``
escaped to ``_1776_`` in the tag), so ``apptainer`` can pull or run it with no
daemon and no root. The mapping from the harness's container operations is:

===================================  ==========================================
``create``                            resolve/pull ``<id>.sif`` in the SIF cache
``copy(src, dest)``                   write into the bind-mounted scratch dir and
                                      record ``cp`` into the container's prelude
``exec*``                             ``apptainer exec --fakeroot --writable-tmpfs``
``cleanup``                           drop the scratch dir (the ``.sif`` is reused)
===================================  ==========================================

**Why a "prelude" and not just commands.** Apptainer has no long-lived container:
every ``apptainer exec`` starts from the image again, so a ``git apply`` in one
exec would be invisible to the next. The harness, written against Docker, expects
persistence (copy the patch, apply it, run ``/eval.sh``). The backend therefore
records every state-changing command it has run and replays them before the next
one, inside a single ``--writable-tmpfs`` overlay. Replay costs a second or two and
keeps ``run_instance`` unchanged; the alternative (a writable sandbox per instance)
costs a 1 GB extraction per instance on a network filesystem.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath

from swebench.harness.container_backends.base import Container, ExecResult

#: Where the harness's per-instance scratch dir appears inside the container.
SCRATCH_DIR = "/out"

#: Commands whose effect must survive into the next `apptainer exec`.
_MUTATING_PREFIXES = (
    "git apply",
    "git checkout",
    "git add",
    "git clean",
    "patch ",
    "cp ",
    "mv ",
    "mkdir",
    "rm ",
)


def escaped_image(instance_id: str, image: str | None = None) -> str:
    """The OCI reference for an instance (dataset-provided ``image`` wins)."""
    if image:
        return image
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}"


def image_ref(image: str) -> str:
    if image.endswith(".sif") or "://" in image:
        return image
    return f"docker://{image}"


def sif_path(cache_dir, instance_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in instance_id)
    return Path(cache_dir) / f"swe-{safe}.sif"


def _as_shell(command) -> str:
    if isinstance(command, (list, tuple)):
        return " ".join(str(part) for part in command)
    return str(command)


class _ApptainerContainer:
    def __init__(self, backend: "ApptainerBackend", test_spec, logger):
        self._backend = backend
        self._spec = test_spec
        self._logger = logger
        self.id = f"apptainer-{test_spec.instance_id}"
        self.sif = backend.ensure_image(test_spec)
        self.scratch = Path(tempfile.mkdtemp(prefix=f"swe-{test_spec.instance_id}-"))
        self.scratch.chmod(0o777)
        # (workdir, command) pairs whose effect must survive into the next exec.
        self._prelude: list[tuple[str | None, str]] = []
        self._workdir: str | None = None

    # -- the interface ---------------------------------------------------

    def start(self) -> None:
        """Prove the image boots before anything is applied to it.

        Without this, a broken image would surface as "the patch did not apply",
        which is the wrong diagnosis.
        """
        result = self.exec_run("true")
        if result.exit_code != 0:
            raise RuntimeError(
                f"apptainer could not start {self.sif}: "
                f"{result.output.decode(errors='replace').strip()[:2000]}"
            )

    def exec_run(self, command, workdir=None, user=None) -> ExecResult:
        shell = _as_shell(command)
        if workdir:
            self._workdir = str(workdir)
        result = self._run(self._script(shell, workdir=workdir), workdir=workdir)
        if result.exit_code == 0 and shell.lstrip().startswith(_MUTATING_PREFIXES):
            # Remember it (with its directory: `git apply` is relative to /testbed)
            # so the next exec starts from the same tree.
            self._prelude.append((self._workdir, shell))
        return result

    # -- internals -------------------------------------------------------

    def _script(self, command: str, *, workdir=None) -> str:
        lines = []
        for directory, recorded in self._prelude:
            if directory:
                lines.append(f"cd {directory} 2>/dev/null || exit 1")
            lines.append(recorded)
        if workdir:
            lines.append(f"cd {workdir} 2>/dev/null || exit 1")
        lines.append(command)
        return "\n".join(lines)

    def _argv(self, *, workdir=None) -> list[str]:
        """The flags every exec needs.

        ``--fakeroot``: SWE-bench images are built to run as root and the eval
        script installs test dependencies. ``--writable-tmpfs``: a .sif is
        read-only, and applying a patch necessarily writes to /testbed.
        """
        argv = [
            self._backend.binary,
            "exec",
            "--cleanenv",
            "--fakeroot",
            "--writable-tmpfs",
            "--bind",
            f"{self.scratch}:{SCRATCH_DIR}",
        ]
        if workdir:
            argv += ["--pwd", str(workdir)]
        return argv

    def _run(self, script: str, *, workdir=None) -> ExecResult:
        argv = [*self._argv(workdir=workdir), str(self.sif), "bash", "-lc", script]
        try:
            result = subprocess.run(
                argv, capture_output=True, timeout=self._backend.exec_timeout
            )
        except subprocess.TimeoutExpired as e:
            return ExecResult(
                exit_code=124, output=(e.output or b"") + b"\n(timeout)"
            )
        except OSError as e:
            return ExecResult(exit_code=127, output=str(e).encode())
        return ExecResult(
            exit_code=result.returncode,
            output=(result.stdout or b"") + (result.stderr or b""),
        )


class ApptainerBackend:
    name = "apptainer"

    def __init__(self, sif_cache=None, binary=None, exec_timeout: int = 1800):
        self.sif_cache = Path(
            sif_cache
            or os.environ.get("SWEBENCH_SIF_CACHE")
            or Path.home() / ".cache" / "swebench" / "sif"
        )
        self.sif_cache.mkdir(parents=True, exist_ok=True)
        self.binary = (
            binary or shutil.which("apptainer") or shutil.which("singularity")
        )
        if not self.binary:
            raise RuntimeError(
                "the apptainer backend needs an `apptainer` or `singularity` binary "
                "on PATH (or pass --apptainer-bin)"
            )
        self.exec_timeout = exec_timeout
        self._env = dict(os.environ)

    # -- images ----------------------------------------------------------

    def ensure_image(self, test_spec) -> Path:
        """Return the instance's ``.sif``, pulling it once if it is not cached."""
        target = sif_path(self.sif_cache, test_spec.instance_id)
        if target.exists():
            return target
        reference = image_ref(escaped_image(test_spec.instance_id, test_spec.image))
        tmp = target.with_suffix(f".sif.tmp.{os.getpid()}")
        result = subprocess.run(
            [self.binary, "pull", "--force", str(tmp), reference],
            capture_output=True,
            env=self._env,
        )
        if result.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"apptainer pull failed for {reference}: "
                f"{result.stderr.decode(errors='replace').strip()[-2000:]}"
            )
        tmp.rename(target)
        return target

    # -- the interface ---------------------------------------------------

    def create(self, test_spec, run_id: str, logger) -> _ApptainerContainer:
        return _ApptainerContainer(self, test_spec, logger)

    def copy(self, container: _ApptainerContainer, source, target: PurePosixPath) -> None:
        """Publish ``source`` at ``target`` inside the container.

        The bytes land in the bind-mounted scratch dir; the ``cp`` into place is
        recorded in the prelude so it happens inside the same exec as whatever
        needs the file.
        """
        staged = container.scratch / Path(target).name
        shutil.copyfile(source, staged)
        staged.chmod(0o644)
        container._prelude.append(
            (None, f"cp {SCRATCH_DIR}/{Path(target).name} {target}")
        )

    def exec_with_timeout(self, container: _ApptainerContainer, command: str, timeout):
        started = time.monotonic()
        argv = [*container._argv(), str(container.sif), "bash", "-lc",
                container._script(command)]
        try:
            result = subprocess.run(argv, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            output = (e.output or b"") + (e.stderr or b"")
            return output.decode(errors="replace"), True, time.monotonic() - started
        elapsed = time.monotonic() - started
        output = (result.stdout or b"") + (result.stderr or b"")
        return output.decode(errors="replace"), False, elapsed

    def cleanup(self, container: _ApptainerContainer | None, logger) -> None:
        if container is not None:
            shutil.rmtree(container.scratch, ignore_errors=True)

    def close(self) -> None:
        pass
