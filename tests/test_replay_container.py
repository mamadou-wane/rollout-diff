import copy
import importlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module():
    return importlib.import_module("rolloutdiff._replay_container")


IMAGE = "sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de"
IMAGE_ENV = [
    "PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LANG=C.UTF-8",
    "GPG_KEY=7169605F62C751356D054A26A821E680E5FA6305",
    "PYTHON_VERSION=3.12.13",
    "PYTHON_SHA256=" + "b" * 64,
]


def _result(value=None, *, returncode=0, status="completed", stderr=b""):
    stdout = json.dumps(value).encode() if value is not None else b""
    return SimpleNamespace(
        returncode=returncode, stdout=stdout, stderr=stderr, status=status
    )


class Docker:
    def __init__(self):
        self.calls = []
        self.created = False
        self.create_visible = True
        self.started = False
        self.removed = False
        self.start_result = _result({"tests_passed": 999})
        self.create_result = _result()
        self.bad_config = False
        self.missing_image = False
        self.cleanup_failure = False
        self.exit_code = 0
        self.oom = False
        self.info = {
            "OSType": "linux",
            "ServerVersion": "29.9.2",
            "MemoryLimit": True,
            "SwapLimit": True,
            "PidsLimit": True,
            "CpuCfsQuota": True,
            "CpuCfsPeriod": True,
            "CgroupVersion": "2",
            "SecurityOptions": ["name=seccomp,profile=builtin", "name=cgroupns"],
            "Runtimes": {"runc": {}},
        }
        self.image = {
            "Id": IMAGE,
            "Os": "linux",
            "Architecture": "arm64",
            "Config": {"Env": IMAGE_ENV, "Entrypoint": None, "Volumes": None},
        }

    def invoke(self, args, *, env, input_data=b"", timeout, output_limit):
        config = Path(args[args.index("--config") + 1])
        assert json.loads((config / "config.json").read_text()) == {"auths": {}}
        assert set(env) == {"HOME", "PATH"}
        assert args[args.index("--host") + 1] == "unix:///tmp/replay-test.sock"
        command = args[5:]
        self.calls.append(command)
        if command[0] == "info":
            return _result(self.info)
        if command[:2] == ["image", "inspect"]:
            if self.missing_image:
                return _result(returncode=1, stderr=b"No such image")
            return _result(self.image)
        if command[0] == "create":
            assert timeout == 10
            assert "--pull=never" in command
            assert "--network=none" in command
            assert "--read-only" in command
            assert "--log-driver=none" in command
            assert not any(part in command for part in ("-v", "--volume", "--mount"))
            self.name = command[command.index("--name") + 1]
            self.label = command[command.index("--label") + 1]
            self.script = command[-1]
            self.created = self.create_visible
            return self.create_result
        if command[:2] == ["container", "inspect"]:
            if not self.created or self.removed:
                return _result(returncode=1, stderr=b"Error: No such container")
            return _result(self.container())
        if command[0] == "start":
            assert self.created
            assert input_data == b"payload"
            self.started = True
            return self.start_result
        if command[0] == "rm":
            assert command[-1] == "c" * 64
            if self.cleanup_failure:
                return _result(returncode=1)
            self.removed = True
            return _result()
        raise AssertionError(command)

    def container(self):
        key, value = self.label.split("=", 1)
        return {
            "Id": "c" * 64,
            "Image": IMAGE,
            "Mounts": [],
            "Config": {
                "User": "65534:65534",
                "WorkingDir": "/tmp",
                "Env": IMAGE_ENV,
                "Entrypoint": None,
                "Volumes": None,
                "Cmd": ["/usr/local/bin/python3", "-I", "-S", "-c", self.script],
                "Labels": {key: value},
                "OpenStdin": True,
                "Tty": False,
                "Healthcheck": {"Test": ["NONE"]},
            },
            "HostConfig": {
                "NetworkMode": "host" if self.bad_config else "none",
                "ReadonlyRootfs": True,
                "Privileged": False,
                "CapAdd": None,
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges"],
                "IpcMode": "none",
                "PidMode": "",
                "UTSMode": "",
                "CgroupnsMode": "private",
                "Memory": 1073741824,
                "MemorySwap": 1073741824,
                "NanoCpus": 1000000000,
                "PidsLimit": 16,
                "Tmpfs": {"/tmp": "rw,noexec,nosuid,nodev,size=16777216"},
                "Binds": None,
                "Mounts": None,
                "Devices": [],
                "DeviceRequests": None,
                "VolumesFrom": None,
                "Links": None,
                "ExtraHosts": None,
                "PortBindings": {},
                "PublishAllPorts": False,
                "AutoRemove": False,
                "Runtime": "runc",
                "LogConfig": {"Type": "none", "Config": {}},
                "Ulimits": [
                    {"Name": "cpu", "Soft": 3, "Hard": 3},
                    {"Name": "core", "Soft": 0, "Hard": 0},
                    {"Name": "nofile", "Soft": 64, "Hard": 64},
                    {"Name": "fsize", "Soft": 1048576, "Hard": 1048576},
                ],
            },
            "State": {
                "Status": "exited" if self.started else "created",
                "Running": False,
                "ExitCode": self.exit_code,
                "OOMKilled": self.oom,
                "Error": "",
            },
        }


def _run(monkeypatch, docker, **kwargs):
    module = _module()
    monkeypatch.setattr(module, "_invoke", docker.invoke)
    monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/local/bin/docker")
    config = module.SandboxConfig(
        image=IMAGE, docker_host="unix:///tmp/replay-test.sock"
    )
    return module.run_container(
        "pass", b"payload", config, execution_authorized=True, **kwargs
    )


def test_requires_explicit_authorization_before_any_process(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "_invoke", lambda *a, **kw: pytest.fail("launched"))
    with pytest.raises(ValueError, match="authorization"):
        module.run_container(
            "pass", b"", module.SandboxConfig(), execution_authorized=False
        )


def test_rejects_unreviewed_image_identity_before_any_process(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "_invoke", lambda *a, **kw: pytest.fail("launched"))
    with pytest.raises(ValueError, match="unsupported image"):
        module.run_container(
            "pass",
            b"",
            module.SandboxConfig(image="sha256:" + "a" * 64),
            execution_authorized=True,
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "tcp://localhost:2375",
        "ssh://host",
        "unix://host/x",
        "unix:///x?y",
        "unix:///x\n",
    ],
)
def test_rejects_nonlocal_or_malformed_endpoint_before_any_process(
    monkeypatch, endpoint
):
    module = _module()
    monkeypatch.setattr(module, "_invoke", lambda *a, **kw: pytest.fail("launched"))
    with pytest.raises(ValueError, match="endpoint"):
        module.run_container(
            "pass",
            b"",
            module.SandboxConfig(docker_host=endpoint),
            execution_authorized=True,
        )


def test_verified_container_result_uses_host_state_and_keeps_stdout_untrusted(
    monkeypatch,
):
    docker = Docker()
    docker.exit_code = 7
    docker.oom = True
    result = _run(monkeypatch, docker)
    assert result.status == "nonzero_exit"
    assert result.exit_code == 7
    assert result.oom_killed is True
    assert result.stdout == b'{"tests_passed": 999}'
    assert result.config_verified and result.cleanup_verified and docker.removed
    assert result.metadata["image_id"] == IMAGE
    assert result.metadata["python_version"] == "3.12.13"


@pytest.mark.parametrize("status", ["timeout", "output_limit"])
def test_execution_limits_remove_entire_owned_container(monkeypatch, status):
    docker = Docker()
    docker.start_result = _result(status=status)
    result = _run(monkeypatch, docker)
    assert result.status == status
    assert result.cleanup_verified and docker.removed
    assert result.exit_code is None
    start_index = next(
        i for i, command in enumerate(docker.calls) if command[0] == "start"
    )
    assert docker.calls[start_index + 1][0] == "rm"


def test_infrastructure_timeout_cleans_up_ambiguous_create(monkeypatch):
    docker = Docker()
    docker.create_result = _result(status="timeout")
    result = _run(monkeypatch, docker)
    assert result.status == "infrastructure_error"
    assert not docker.started
    assert docker.removed and result.cleanup_verified


@pytest.mark.parametrize("status,returncode", [("timeout", 0), ("completed", 1)])
def test_unconfirmed_create_cannot_claim_cleanup_from_absence(
    monkeypatch, status, returncode
):
    docker = Docker()
    docker.create_result = _result(status=status, returncode=returncode)
    docker.create_visible = False
    result = _run(monkeypatch, docker)
    assert result.status == "infrastructure_error"
    assert not result.cleanup_verified
    assert result.metadata["container_name"].startswith("rolloutdiff-replay-")


def test_rejects_unverified_configuration_before_start(monkeypatch):
    docker = Docker()
    docker.bad_config = True
    result = _run(monkeypatch, docker)
    assert result.status == "infrastructure_error"
    assert not result.config_verified and not docker.started
    assert docker.removed and result.cleanup_verified


def test_missing_image_never_pulls_or_creates(monkeypatch):
    docker = Docker()
    docker.missing_image = True
    result = _run(monkeypatch, docker)
    assert result.status == "infrastructure_error"
    assert not docker.created
    assert not any("pull" in command for command in docker.calls)


@pytest.mark.parametrize(
    "field,value",
    [("MemoryLimit", False), ("PidsLimit", False), ("SecurityOptions", [])],
)
def test_missing_daemon_enforcement_blocks_creation(monkeypatch, field, value):
    docker = Docker()
    docker.info[field] = value
    result = _run(monkeypatch, docker)
    assert result.status == "infrastructure_error"
    assert not docker.created


@pytest.mark.parametrize(
    "field,value",
    [
        ("Entrypoint", ["sh"]),
        ("Volumes", {"/secret": {}}),
        ("Env", [*IMAGE_ENV, "AWS_SECRET_ACCESS_KEY=private"]),
    ],
)
def test_unexpected_image_authority_blocks_creation(monkeypatch, field, value):
    docker = Docker()
    docker.image = copy.deepcopy(docker.image)
    docker.image["Config"][field] = value
    result = _run(monkeypatch, docker)
    assert result.status == "infrastructure_error"
    assert not docker.created


def test_cleanup_failure_cannot_be_reported_as_completed(monkeypatch):
    docker = Docker()
    docker.cleanup_failure = True
    result = _run(monkeypatch, docker)
    assert result.status == "infrastructure_error"
    assert not result.cleanup_verified
    assert result.error == "container cleanup could not be verified"


@pytest.mark.parametrize("status", ["completed", "timeout"])
@pytest.mark.parametrize("phase", ["remove", "absence"])
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_cleanup_interrupt_preserves_cancellation_and_recovery_record(
    monkeypatch, status, phase, cleanup_failure
):
    interruption = KeyboardInterrupt("controlled cleanup interruption")

    class InterruptedDocker(Docker):
        fired = False
        removing = False

        def invoke(self, args, **kwargs):
            command = args[5:]
            if command[0] == "rm":
                self.removing = True
            hit = (
                command[0] == "rm"
                if phase == "remove"
                else (self.removing and command[:2] == ["container", "inspect"])
            )
            if hit and not self.fired:
                self.fired = True
                raise interruption
            return super().invoke(args, **kwargs)

    docker = InterruptedDocker()
    docker.start_result = _result(status=status)
    docker.cleanup_failure = cleanup_failure
    with pytest.raises(KeyboardInterrupt) as caught:
        _run(monkeypatch, docker)
    assert caught.value is interruption
    result = caught.value.replay_result
    assert result.cleanup_verified is not cleanup_failure
    assert docker.removed is not cleanup_failure
    assert result.metadata["container_id"] == "c" * 64
    assert result.metadata["container_name"] == docker.name
    assert result.status == ("infrastructure_error" if cleanup_failure else status)
    assert docker.calls[-1][:2] == ["container", "inspect"]
    assert sum(call[0] == "start" for call in docker.calls) == 1


def test_repeated_cleanup_interrupts_are_bounded_and_recorded(monkeypatch):
    interruption = KeyboardInterrupt("repeated cleanup interruption")

    class InterruptedDocker(Docker):
        removals = 0

        def invoke(self, args, **kwargs):
            if args[5] == "rm":
                self.removals += 1
                raise interruption
            return super().invoke(args, **kwargs)

    docker = InterruptedDocker()
    with pytest.raises(KeyboardInterrupt) as caught:
        _run(monkeypatch, docker)
    result = caught.value.replay_result
    assert caught.value is interruption
    assert docker.removals == 2
    assert not result.cleanup_verified
    assert result.status == "infrastructure_error"
    assert result.metadata["container_id"] == "c" * 64
    assert docker.calls[-1][:2] == ["container", "inspect"]


def test_execution_interrupt_reaches_caller_after_cleanup(monkeypatch):
    interruption = KeyboardInterrupt("controlled execution interruption")

    class InterruptedDocker(Docker):
        def invoke(self, args, **kwargs):
            result = super().invoke(args, **kwargs)
            if args[5] == "start":
                raise interruption
            return result

    docker = InterruptedDocker()
    with pytest.raises(KeyboardInterrupt) as caught:
        _run(monkeypatch, docker)
    assert caught.value is interruption
    assert caught.value.replay_result.cleanup_verified
    assert docker.removed


def test_cleanup_transport_failure_retains_identity_and_checks_absence(monkeypatch):
    module = _module()

    class FailedTransport(Docker):
        def invoke(self, args, **kwargs):
            if args[5] == "rm":
                error = module._InfrastructureError("controlled reap failure")
                error.process_cleanup_failure = {"pid": 123, "reaped": False}
                raise error
            return super().invoke(args, **kwargs)

    docker = FailedTransport()
    result = _run(monkeypatch, docker)
    assert result.status == "infrastructure_error"
    assert not result.cleanup_verified
    assert result.metadata["container_id"] == "c" * 64
    assert result.metadata["process_cleanup_failure"] == {"pid": 123, "reaped": False}
    assert docker.calls[-1][:2] == ["container", "inspect"]


def test_lost_create_recovery_preserves_verified_identity(monkeypatch):
    docker = Docker()
    docker.create_result = _result(status="timeout")
    docker.cleanup_failure = True
    result = _run(monkeypatch, docker)
    assert result.status == "infrastructure_error"
    assert not result.cleanup_verified
    assert result.metadata["container_id"] == "c" * 64


def test_interrupt_between_cleanup_commands_is_deferred(monkeypatch):
    module = _module()
    original = module._cleanup
    interruption = KeyboardInterrupt("between cleanup commands")
    calls = []

    def cleanup(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise interruption
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_cleanup", cleanup)
    docker = Docker()
    with pytest.raises(KeyboardInterrupt) as caught:
        _run(monkeypatch, docker)
    assert caught.value is interruption
    assert caught.value.replay_result.cleanup_verified
    assert docker.removed
    assert len(calls) == 2


@pytest.mark.parametrize("kind", ["timeout", "output_limit"])
def test_transport_bounds_output_and_wall_time(kind):
    module = _module()
    script = (
        "import time; time.sleep(10)" if kind == "timeout" else "print('x' * 100000)"
    )
    result = module._invoke(
        [sys.executable, "-I", "-c", script],
        env={"PATH": "/usr/bin:/bin"},
        timeout=0.1 if kind == "timeout" else 2,
        output_limit=100,
    )
    assert result.status == kind
    assert len(result.stdout) + len(result.stderr) <= 100


@pytest.mark.parametrize("exception_type", [OSError, KeyboardInterrupt])
@pytest.mark.parametrize("phase", ["selector", "blocking", "registration"])
def test_transport_setup_failure_reaps_owned_process(
    monkeypatch, exception_type, phase
):
    module = _module()
    children, selectors = [], []
    original_popen = subprocess.Popen
    original_selector = module.selectors.DefaultSelector
    error = exception_type("controlled stream setup fault")

    def spawn(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        children.append(process)
        return process

    def fail(*args, **kwargs):
        raise error

    def selector():
        if phase == "selector":
            raise error
        value = original_selector()
        selectors.append(value)
        if phase == "registration":
            monkeypatch.setattr(value, "register", fail)
        return value

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    monkeypatch.setattr(module.selectors, "DefaultSelector", selector)
    if phase == "blocking":
        monkeypatch.setattr(module.os, "set_blocking", fail)
    try:
        with pytest.raises(exception_type) as caught:
            module._invoke(
                [sys.executable, "-I", "-S", "-c", "import time; time.sleep(5)"],
                env={"PATH": os.defpath},
                timeout=0.05,
                output_limit=100,
            )
        assert caught.value is error
        process = children[0]
        assert process.returncode is not None
        with pytest.raises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)
        assert all(s.closed for s in (process.stdin, process.stdout, process.stderr))
        assert all(s.get_map() is None for s in selectors)
    finally:
        for process in children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=1)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()


def test_transport_normal_completion():
    result = _module()._invoke(
        [sys.executable, "-I", "-S", "-c", "print('controlled')"],
        env={"PATH": os.defpath},
        timeout=2,
        output_limit=100,
    )
    assert result.status == "completed"
    assert result.returncode == 0
    assert result.stdout == b"controlled\n"


def test_transport_reap_failure_preserves_original_error_and_recovery(monkeypatch):
    module = _module()
    children = []
    original_popen = subprocess.Popen
    original_killpg = os.killpg
    error = KeyboardInterrupt("controlled setup cancellation")

    def spawn(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        children.append(process)
        return process

    def fail(*args, **kwargs):
        raise error

    def cannot_kill(*args):
        raise PermissionError("controlled kill failure")

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    monkeypatch.setattr(module.os, "set_blocking", fail)
    monkeypatch.setattr(module.os, "killpg", cannot_kill)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            module._invoke(
                [sys.executable, "-I", "-S", "-c", "import time; time.sleep(5)"],
                env={"PATH": os.defpath},
                timeout=0.05,
                output_limit=100,
            )
        assert caught.value is error
        assert error.process_cleanup_failure == {
            "pid": children[0].pid,
            "reaped": False,
        }
        assert isinstance(error.__cause__, PermissionError)
        assert all(
            s.closed
            for s in (children[0].stdin, children[0].stdout, children[0].stderr)
        )
    finally:
        for process in children:
            if process.poll() is None:
                original_killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=1)


@pytest.mark.parametrize("phase", ["kill", "wait"])
@pytest.mark.parametrize("interrupts", [1, 2])
def test_transport_reap_interrupt_is_bounded_and_preserved(
    monkeypatch, phase, interrupts
):
    module = _module()
    children = []
    original_popen, original_killpg = subprocess.Popen, os.killpg
    interruption = KeyboardInterrupt("controlled reap interruption")
    faults = []

    def fault():
        if len(faults) < interrupts:
            faults.append(True)
            raise interruption

    def spawn(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        children.append(process)
        original_wait = process.wait
        if phase == "wait":

            def wait(*args, **kwargs):
                fault()
                return original_wait(*args, **kwargs)

            monkeypatch.setattr(process, "wait", wait)
        return process

    def killpg(*args):
        fault()
        return original_killpg(*args)

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    if phase == "kill":
        monkeypatch.setattr(module.os, "killpg", killpg)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            module._invoke(
                [sys.executable, "-I", "-S", "-c", "import time; time.sleep(5)"],
                env={"PATH": os.defpath},
                timeout=0.02,
                output_limit=100,
            )
        assert caught.value is interruption
        assert len(faults) == interrupts
        process = children[0]
        if interrupts == 1:
            assert process.returncode is not None
            assert not hasattr(interruption, "process_cleanup_failure")
            with pytest.raises(ChildProcessError):
                os.waitpid(process.pid, os.WNOHANG)
        else:
            assert interruption.process_cleanup_failure == {
                "pid": process.pid,
                "reaped": False,
            }
        assert all(s.closed for s in (process.stdin, process.stdout, process.stderr))
    finally:
        for process in children:
            if phase == "kill" and process.poll() is None:
                original_killpg(process.pid, signal.SIGKILL)
            original_popen.wait(process, timeout=1)
