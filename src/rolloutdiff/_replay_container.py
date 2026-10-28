"""Bounded local Docker transport; container output has no verdict authority."""

from __future__ import annotations

import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_IMAGE = (
    "sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de"
)
_IMAGE_PATH = (
    "/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
_OUTPUT_LIMIT = 1024 * 1024
_OWNER_LABEL = "org.rolloutdiff.replay-owner"
_ULIMITS = {"cpu": 3, "core": 0, "nofile": 64, "fsize": 1048576}
_HOST_CONFIG = {
    "NetworkMode": "none",
    "ReadonlyRootfs": True,
    "Privileged": False,
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
    "PublishAllPorts": False,
    "AutoRemove": False,
    "Runtime": "runc",
    "LogConfig": {"Type": "none", "Config": {}},
}


@dataclass(frozen=True)
class SandboxConfig:
    image: str = DEFAULT_IMAGE
    docker_host: str = "unix:///var/run/docker.sock"


@dataclass(frozen=True)
class ContainerResult:
    status: str
    exit_code: int | None = None
    oom_killed: bool | None = None
    elapsed_seconds: float = 0.0
    stdout: bytes = b""
    stderr: bytes = b""
    config_verified: bool = False
    cleanup_verified: bool = False
    error: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _ProcessResult:
    returncode: int | None
    stdout: bytes
    stderr: bytes
    status: str = "completed"


class _InfrastructureError(Exception):
    pass


def _invoke(args, *, env, input_data=b"", timeout, output_limit):
    """Drain both pipes incrementally and stop before retaining excess output."""
    process = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        close_fds=True,
        start_new_session=True,
    )
    try:
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        status, written, retained = "completed", 0, 0
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            for stream, event, name in (
                (process.stdin, selectors.EVENT_WRITE, "stdin"),
                (process.stdout, selectors.EVENT_READ, "stdout"),
                (process.stderr, selectors.EVENT_READ, "stderr"),
            ):
                os.set_blocking(stream.fileno(), False)
                if name == "stdin" and not input_data:
                    stream.close()
                else:
                    selector.register(stream, event, name)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = "timeout"
                    break
                for key, _ in selector.select(min(remaining, 0.1)):
                    if key.data == "stdin":
                        try:
                            written += os.write(
                                key.fd,
                                memoryview(input_data)[written : written + 65536],
                            )
                        except BrokenPipeError:
                            written = len(input_data)
                        if written == len(input_data):
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                    else:
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            continue
                        allowed = output_limit - retained
                        buffers[key.data].extend(chunk[:allowed])
                        retained += min(allowed, len(chunk))
                        if len(chunk) > allowed:
                            status = "output_limit"
                            break
                if status != "completed":
                    break
            if status == "completed":
                try:
                    process.wait(timeout=max(0.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    status = "timeout"
    finally:
        original_error = sys.exception()
        interruption = cleanup_failure = None
        killed = False
        try:
            for _ in range(2):
                try:
                    if not killed and process.poll() is None:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                        killed = True
                    process.wait(timeout=1)
                except (KeyboardInterrupt, SystemExit) as exc:
                    if interruption is None:
                        interruption = exc
                    cleanup_failure = exc
                except (OSError, subprocess.SubprocessError) as exc:
                    cleanup_failure = exc
                    break
                else:
                    cleanup_failure = None
                    break
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
        if cleanup_failure is not None:
            error = (
                original_error
                or interruption
                or _InfrastructureError(
                    "owned subprocess cleanup could not be verified"
                )
            )
            error.process_cleanup_failure = {"pid": process.pid, "reaped": False}
            error.add_note(
                f"Owned subprocess cleanup is unverified (pid {process.pid})"
            )
            if error is cleanup_failure:
                raise error
            raise error from cleanup_failure
        if interruption is not None and original_error is None:
            raise interruption
    return _ProcessResult(
        process.returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"]), status
    )


def _checked_json(result, operation):
    if result.status != "completed" or result.returncode != 0:
        raise _InfrastructureError(f"Docker {operation} failed")
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise _InfrastructureError(f"Docker {operation} returned invalid metadata")
    return value


def _validate_image(image, config):
    settings = image["Config"]
    values = settings.get("Env", [])
    env = dict(item.split("=", 1) for item in values)
    if (
        image["Id"] != config.image
        or image["Os"] != "linux"
        or image["Architecture"] not in {"arm64", "amd64"}
        or settings.get("Entrypoint")
        or settings.get("Volumes")
        or len(env) != len(values)
        or set(env) != {"PATH", "LANG", "GPG_KEY", "PYTHON_VERSION", "PYTHON_SHA256"}
        or env["PATH"] != _IMAGE_PATH
        or env["LANG"] != "C.UTF-8"
        or not re.fullmatch(r"3\.12\.\d+", env["PYTHON_VERSION"])
        or not re.fullmatch(r"[A-F0-9]{40}", env["GPG_KEY"])
        or not re.fullmatch(r"[a-f0-9]{64}", env["PYTHON_SHA256"])
    ):
        raise _InfrastructureError("local image configuration is unsupported")
    return env


def _validate_daemon(info):
    if (
        info.get("OSType") != "linux"
        or info.get("CgroupVersion") != "2"
        or "name=seccomp,profile=builtin" not in info.get("SecurityOptions", [])
        or "runc" not in info.get("Runtimes", {})
        or any(
            info.get(key) is not True
            for key in (
                "MemoryLimit",
                "SwapLimit",
                "PidsLimit",
                "CpuCfsQuota",
                "CpuCfsPeriod",
            )
        )
    ):
        raise _InfrastructureError("Docker isolation prerequisites are unavailable")


def _verify_container(container, image, command, owner):
    host, settings = container["HostConfig"], container["Config"]
    ulimits = {item["Name"]: (item["Soft"], item["Hard"]) for item in host["Ulimits"]}
    if (
        any(host.get(key) != value for key, value in _HOST_CONFIG.items())
        or any(
            host.get(key)
            for key in (
                "CapAdd",
                "Binds",
                "Mounts",
                "Devices",
                "DeviceRequests",
                "VolumesFrom",
                "Links",
                "ExtraHosts",
                "PortBindings",
            )
        )
        or ulimits != {name: (limit, limit) for name, limit in _ULIMITS.items()}
        or container.get("Mounts")
        or not re.fullmatch(r"[a-f0-9]{64}", container["Id"])
        or container["Image"] != image["Id"]
        or settings.get("User") != "65534:65534"
        or settings.get("WorkingDir") != "/tmp"
        or settings.get("Env") != image["Config"]["Env"]
        or settings.get("Entrypoint")
        or settings.get("Volumes")
        or settings.get("Cmd") != command
        or settings.get("Labels", {}).get(_OWNER_LABEL) != owner
        or settings.get("OpenStdin") is not True
        or settings.get("Tty") is not False
        or settings.get("Healthcheck", {}).get("Test") != ["NONE"]
        or container["State"].get("Status") != "created"
    ):
        raise _InfrastructureError("created container failed isolation verification")


def _cleanup(command, name, owner, *, creation_resolved, known_id, recovery):
    if known_id is not None:
        with suppress(OSError, subprocess.SubprocessError, _InfrastructureError):
            command(["rm", "--force", "--volumes", known_id])
    for remove in (known_id is None, False):
        result = command(["container", "inspect", "--format", "{{json .}}", name])
        if (
            result.status == "completed"
            and result.returncode == 1
            and (
                b"No such container" in result.stderr
                or b"No such object" in result.stderr
            )
        ):
            return creation_resolved
        try:
            container = _checked_json(result, "cleanup inspection")
            identity = container["Id"]
            if container["Config"].get("Labels", {}).get(
                _OWNER_LABEL
            ) != owner or not re.fullmatch(r"[a-f0-9]{64}", identity):
                return False
            recovery["container_id"] = identity
        except (OSError, ValueError, KeyError, TypeError, _InfrastructureError):
            return False
        if remove:
            creation_resolved = True
            with suppress(OSError, subprocess.SubprocessError, _InfrastructureError):
                command(["rm", "--force", "--volumes", identity])
    return False


def run_container(
    script: str, payload: bytes, config: SandboxConfig, *, execution_authorized: bool
) -> ContainerResult:
    """Run an authorized payload; only daemon state supplies execution status.

    Cancellation propagates after bounded cleanup, with its ContainerResult
    attached as ``replay_result`` for recovery when absence is unverified.
    """
    if execution_authorized is not True:
        raise ValueError("container execution requires explicit authorization")
    endpoint = urlsplit(config.docker_host)
    if (
        endpoint.scheme != "unix"
        or endpoint.netloc
        or not endpoint.path.startswith("/")
        or endpoint.query
        or endpoint.fragment
        or "%" in config.docker_host
        or any(ord(char) < 33 for char in config.docker_host)
    ):
        raise ValueError("Docker endpoint must be an absolute local unix socket")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", config.image):
        raise ValueError("image must be an immutable local sha256 identity")
    if config.image != DEFAULT_IMAGE:
        raise ValueError(
            "unsupported image: this slice requires the reviewed Python image"
        )
    if not isinstance(payload, bytes) or len(payload) > 2 * 1024 * 1024:
        raise ValueError("container input exceeds the 2 MiB limit")
    if not isinstance(script, str) or len(script.encode()) > 65536 or "\0" in script:
        raise ValueError("container runner must be at most 64 KiB without NUL bytes")
    started = time.monotonic()
    status, error = "infrastructure_error", None
    stdout, stderr, exit_code, oom = b"", b"", None, None
    verified, cleaned, creation_attempted = False, True, False
    creation_resolved = False
    container_id = None
    interruption = None
    metadata = {}
    owner = uuid.uuid4().hex
    name = "rolloutdiff-replay-" + owner
    python_command = ["/usr/local/bin/python3", "-I", "-S", "-c", script]
    with tempfile.TemporaryDirectory(prefix="rolloutdiff-docker-") as directory:
        Path(directory, "config.json").write_text('{"auths": {}}')
        docker = shutil.which("docker")
        if docker is None:
            return ContainerResult(
                "infrastructure_error",
                error="Docker CLI is unavailable",
                cleanup_verified=True,
            )
        base = [docker, "--config", directory, "--host", config.docker_host]
        env = {"HOME": directory, "PATH": os.defpath}

        def command(args):
            return _invoke(
                base + args,
                env=env,
                timeout=10 if args[0] == "create" else 3,
                output_limit=_OUTPUT_LIMIT,
            )

        def cleanup_command(args):
            nonlocal interruption
            # A cancelled CLI command has an unknown daemon outcome. Retry once;
            # the final inspection still decides whether cleanup is established.
            for _ in range(2):
                try:
                    return command(args)
                except (KeyboardInterrupt, SystemExit) as exc:
                    if interruption is None:
                        interruption = exc
                    if hasattr(exc, "process_cleanup_failure"):
                        metadata["process_cleanup_failure"] = (
                            exc.process_cleanup_failure
                        )
                except (
                    OSError,
                    subprocess.SubprocessError,
                    _InfrastructureError,
                ) as exc:
                    if hasattr(exc, "process_cleanup_failure"):
                        metadata["process_cleanup_failure"] = (
                            exc.process_cleanup_failure
                        )
                    return _ProcessResult(None, b"", b"", "infrastructure_error")
            return _ProcessResult(None, b"", b"", "interrupted")

        try:
            info = _checked_json(command(["info", "--format", "{{json .}}"]), "info")
            _validate_daemon(info)
            image = _checked_json(
                command(["image", "inspect", "--format", "{{json .}}", config.image]),
                "image inspection",
            )
            image_env = _validate_image(image, config)
            metadata = {
                "image_id": image["Id"],
                "architecture": image["Architecture"],
                "python_version": image_env["PYTHON_VERSION"],
                "daemon_version": info.get("ServerVersion"),
                "cgroup_version": info["CgroupVersion"],
                "runtime": "runc",
                "limits": {
                    "memory_bytes": 1073741824,
                    "cpus": 1,
                    "pids": 16,
                    "wall_seconds": 6,
                    "output_bytes": _OUTPUT_LIMIT,
                    "stdin_bytes": 2097152,
                    "tmp_bytes": 16777216,
                },
            }
            create = [
                "create",
                "--name",
                name,
                "--label",
                f"{_OWNER_LABEL}={owner}",
                "--pull=never",
                "--interactive",
                "--network=none",
                "--read-only",
                "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16777216",
                "--ipc=none",
                "--user=65534:65534",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--cgroupns=private",
                "--memory=1073741824",
                "--memory-swap=1073741824",
                "--cpus=1",
                "--pids-limit=16",
                "--log-driver=none",
                "--runtime=runc",
                "--no-healthcheck",
                "--workdir=/tmp",
            ]
            for key, limit in _ULIMITS.items():
                create.append(f"--ulimit={key}={limit}:{limit}")
            metadata["container_name"] = name
            creation_attempted = True
            created = command([*create, config.image, *python_command])
            creation_resolved = (
                created.status == "completed" and created.returncode == 0
            )
            if created.status != "completed" or created.returncode != 0:
                raise _InfrastructureError("Docker create failed")
            container = _checked_json(
                command(["container", "inspect", "--format", "{{json .}}", name]),
                "configuration inspection",
            )
            _verify_container(container, image, python_command, owner)
            verified = True
            container_id = container["Id"]
            metadata["container_id"] = container_id
            result = _invoke(
                [*base, "start", "--attach", "--interactive", name],
                env=env,
                input_data=payload,
                timeout=6,
                output_limit=_OUTPUT_LIMIT,
            )
            stdout, stderr = result.stdout, result.stderr
            if result.status in {"timeout", "output_limit"}:
                status = result.status
            else:
                state = _checked_json(
                    command(["container", "inspect", "--format", "{{json .}}", name]),
                    "outcome inspection",
                )["State"]
                if (
                    state.get("Status") != "exited"
                    or state.get("Running") is not False
                    or state.get("Error")
                    or type(state.get("ExitCode")) is not int
                    or type(state.get("OOMKilled")) is not bool
                ):
                    raise _InfrastructureError(
                        "container outcome could not be verified"
                    )
                exit_code, oom = state["ExitCode"], state["OOMKilled"]
                status = "completed" if exit_code == 0 else "nonzero_exit"
        except _InfrastructureError as exc:
            error = str(exc)
            if hasattr(exc, "process_cleanup_failure"):
                metadata["process_cleanup_failure"] = exc.process_cleanup_failure
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
        ) as exc:
            error = "Docker transport or metadata validation failed"
            if hasattr(exc, "process_cleanup_failure"):
                metadata["process_cleanup_failure"] = exc.process_cleanup_failure
        except (KeyboardInterrupt, SystemExit) as exc:
            interruption = exc
            error = "Docker execution interrupted"
            if hasattr(exc, "process_cleanup_failure"):
                metadata["process_cleanup_failure"] = exc.process_cleanup_failure
        finally:
            if creation_attempted:
                cleaned = False
                for _ in range(2):
                    try:
                        cleaned = _cleanup(
                            cleanup_command,
                            name,
                            owner,
                            creation_resolved=creation_resolved,
                            known_id=container_id,
                            recovery=metadata,
                        )
                        break
                    except (KeyboardInterrupt, SystemExit) as exc:
                        if interruption is None:
                            interruption = exc
                    except (
                        OSError,
                        ValueError,
                        KeyError,
                        TypeError,
                        subprocess.SubprocessError,
                        _InfrastructureError,
                    ):
                        break
            if "process_cleanup_failure" in metadata:
                cleaned = False
            if not cleaned:
                status, error = (
                    "infrastructure_error",
                    "container cleanup could not be verified",
                )
    result = ContainerResult(
        status,
        exit_code,
        oom,
        time.monotonic() - started,
        stdout,
        stderr,
        verified,
        cleaned,
        error,
        metadata,
    )
    if interruption is not None:
        interruption.replay_result = result
        if not result.cleanup_verified:
            interruption.add_note(
                "Container cleanup is unverified; recovery identity is in replay_result"
            )
        raise interruption
    return result
