"""Tests for the container backend abstraction added on our fork.

Two things are worth pinning down:

1. the Docker backend is a pure adapter -- the same ``docker_utils`` helpers, in the
   same order, with the same arguments, so a default run cannot change behaviour;
2. the Apptainer backend's *replay* semantics, which exist because Apptainer has no
   long-lived container: a `git apply` in one exec must be re-applied for the next
   one, or the eval script runs against an unpatched tree (which silently scores
   gold patches as unresolved).
"""

import subprocess
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from swebench.harness import container_backends
from swebench.harness.container_backends import apptainer as apptainer_module
from swebench.harness.container_backends import docker as docker_module
from swebench.harness.container_backends import get_backend


def test_get_backend_selects_and_validates(tmp_path):
    assert get_backend("docker").name == "docker"
    backend = get_backend(
        "apptainer", binary="/usr/bin/apptainer", sif_cache=str(tmp_path / "sif")
    )
    assert backend.name == "apptainer"
    with pytest.raises(ValueError, match="unknown container backend"):
        get_backend("podman")


def test_exported_names():
    assert set(container_backends.BACKENDS) == {"docker", "apptainer"}
    assert container_backends.ContainerBackend is not None


def test_escaped_image_matches_the_published_tags():
    # SWE-bench tags cannot contain "__"; the dataset uses _1776_ instead.
    assert (
        apptainer_module.escaped_image("astropy__astropy-12907")
        == "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907"
    )
    assert (
        apptainer_module.escaped_image("a__b-1", "someone/else:tag")
        == "someone/else:tag"
    )
    assert apptainer_module.image_ref("alpine:3.20") == "docker://alpine:3.20"
    assert apptainer_module.image_ref("/x/y.sif") == "/x/y.sif"


def test_sif_path_is_filesystem_safe(tmp_path):
    path = apptainer_module.sif_path(tmp_path, "astropy__astropy-12907")
    assert path.name == "swe-astropy__astropy-12907.sif"
    assert path.parent == tmp_path


def test_docker_backend_forwards_to_the_original_helpers(monkeypatch):
    """A default run must still call exactly the docker_utils helpers."""
    calls = []

    class FakeContainer(SimpleNamespace):
        id = "abc"

        def start(self):
            calls.append(("start",))

        def exec_run(self, command, workdir=None, user=None):
            calls.append(("exec_run", command, workdir, user))
            return SimpleNamespace(exit_code=0, output=b"ok")

    container = FakeContainer()
    monkeypatch.setattr(
        docker_module,
        "copy_to_container",
        lambda c, src, dst: calls.append(("copy_to_container", dst)),
    )
    monkeypatch.setattr(
        docker_module,
        "exec_run_with_timeout",
        lambda c, cmd, timeout: calls.append(("exec_run_with_timeout", cmd)) or (
            "out",
            False,
            1.0,
        ),
    )
    monkeypatch.setattr(
        docker_module,
        "cleanup_container",
        lambda client, c, logger: calls.append(("cleanup_container",)),
    )

    backend = get_backend("docker", client=object())
    wrapped = docker_module._DockerContainer(container)
    wrapped.start()
    backend.copy(wrapped, "/host/patch.diff", PurePosixPath("/tmp/patch.diff"))
    backend.exec_with_timeout(wrapped, "bash /eval.sh", 10)
    backend.cleanup(wrapped, None)

    assert [call[0] for call in calls] == [
        "start",
        "copy_to_container",
        "exec_run_with_timeout",
        "cleanup_container",
    ]
    assert calls[1][1] == PurePosixPath("/tmp/patch.diff")


def _fake_backend(tmp_path, monkeypatch, results):
    """An ApptainerBackend whose execs return scripted results."""
    backend = get_backend(
        "apptainer", binary="/usr/bin/apptainer", sif_cache=str(tmp_path)
    )
    sif = tmp_path / "swe-x.sif"
    sif.write_text("sif")
    monkeypatch.setattr(backend, "ensure_image", lambda spec: sif)
    seen = []

    def fake_run(argv, capture_output=True, timeout=None, **kwargs):
        script = argv[-1]
        seen.append(script)
        code, out = results.pop(0) if results else (0, b"")
        return subprocess.CompletedProcess(argv, code, out, b"")

    monkeypatch.setattr(apptainer_module.subprocess, "run", fake_run)
    return backend, seen


def test_apptainer_replays_mutating_commands(tmp_path, monkeypatch):
    backend, seen = _fake_backend(tmp_path, monkeypatch, [(0, b""), (0, b""), (0, b"")])
    spec = SimpleNamespace(instance_id="x", image=None)
    container = backend.create(spec, "run", None)

    container.exec_run("git apply -v /tmp/patch.diff", workdir="/testbed")
    backend.copy(container, __file__, PurePosixPath("/eval.sh"))
    backend.exec_with_timeout(container, "/bin/bash /eval.sh", 60)

    final = seen[-1]
    # The patch is re-applied in the testbed, from the bind-mounted scratch dir...
    assert "cd /testbed" in final
    assert "git apply -v /tmp/patch.diff" in final
    # ...the eval script is copied in, and only then does the script run.
    assert f"cp {apptainer_module.SCRATCH_DIR}/eval.sh /eval.sh" in final
    assert final.rstrip().endswith("/bin/bash /eval.sh")


def test_apptainer_does_not_replay_failed_commands(tmp_path, monkeypatch):
    backend, seen = _fake_backend(tmp_path, monkeypatch, [(1, b"boom")])
    spec = SimpleNamespace(instance_id="x", image=None)
    container = backend.create(spec, "run", None)

    result = container.exec_run("git apply -v /tmp/patch.diff", workdir="/testbed")
    assert result.exit_code == 1
    container.exec_run("true")
    assert "git apply" not in seen[-1], "a failed apply must not be replayed"


def test_apptainer_exec_uses_fakeroot_and_writable_tmpfs(tmp_path, monkeypatch):
    backend, seen_argv = _fake_backend(tmp_path, monkeypatch, [])
    spec = SimpleNamespace(instance_id="x", image=None)
    container = backend.create(spec, "run", None)
    argv_seen = []
    original = apptainer_module.subprocess.run

    def capture(argv, **kwargs):
        argv_seen.append(argv)
        return original(argv, **kwargs)

    monkeypatch.setattr(apptainer_module.subprocess, "run", capture)
    container.exec_run("true")
    argv = argv_seen[-1]
    assert "--fakeroot" in argv
    assert "--writable-tmpfs" in argv, "a .sif is read-only; patches need an overlay"
    assert "--bind" in argv
