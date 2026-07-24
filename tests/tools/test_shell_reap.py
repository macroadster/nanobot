"""Tests for subprocess zombie reaping in ExecTool / exec sessions."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.tools.exec_session import _ExecSession
from nanobot.agent.tools.shell import (
    ExecTool,
    _reap_pid,
    _reap_process_group,
    _signal_process_tree,
)


def test_reap_pid_noops_without_waitpid():
    """On platforms (or test stubs) without waitpid, reaping is a no-op."""
    with patch("nanobot.agent.tools.shell.os") as mock_os:
        mock_os.waitpid = None
        mock_os.WNOHANG = None
        _reap_pid(12345)


def test_reap_pid_calls_waitpid_wnohang():
    with patch("nanobot.agent.tools.shell.os") as mock_os:
        mock_os.waitpid = MagicMock(return_value=(12345, 0))
        mock_os.WNOHANG = 1
        _reap_pid(12345)
        mock_os.waitpid.assert_called_once_with(12345, 1)


def test_reap_pid_swallows_already_reaped_errors():
    with patch("nanobot.agent.tools.shell.os") as mock_os:
        mock_os.WNOHANG = 1
        mock_os.waitpid = MagicMock(side_effect=ChildProcessError("no child"))
        _reap_pid(99)

        mock_os.waitpid = MagicMock(side_effect=ProcessLookupError("gone"))
        _reap_pid(99)


def test_signal_process_tree_killpg_when_group_leader():
    """Child is session leader (pgid == pid) → kill the whole group."""
    process = MagicMock()
    process.pid = 5001
    process.returncode = None
    process.kill = MagicMock()

    with (
        patch("nanobot.agent.tools.shell._IS_WINDOWS", False),
        patch("nanobot.agent.tools.shell.os.getpgid", return_value=5001) as getpgid,
        patch("nanobot.agent.tools.shell.os.killpg") as killpg,
    ):
        pgid = _signal_process_tree(process)

    assert pgid == 5001
    getpgid.assert_called_once_with(5001)
    killpg.assert_called_once_with(5001, signal.SIGKILL)
    process.kill.assert_not_called()


def test_signal_process_tree_never_killpg_foreign_group():
    """If child shares another pgid, do not killpg (would hit nanobot itself)."""
    process = MagicMock()
    process.pid = 5002
    process.returncode = None
    process.kill = MagicMock()

    with (
        patch("nanobot.agent.tools.shell._IS_WINDOWS", False),
        patch("nanobot.agent.tools.shell.os.getpgid", return_value=1),
        patch("nanobot.agent.tools.shell.os.killpg") as killpg,
    ):
        pgid = _signal_process_tree(process)

    assert pgid is None
    killpg.assert_not_called()
    process.kill.assert_called_once()


def test_signal_process_tree_falls_back_when_getpgid_fails():
    process = MagicMock()
    process.pid = 5003
    process.returncode = None
    process.kill = MagicMock()

    with (
        patch("nanobot.agent.tools.shell._IS_WINDOWS", False),
        patch(
            "nanobot.agent.tools.shell.os.getpgid",
            side_effect=ProcessLookupError("gone"),
        ),
        patch("nanobot.agent.tools.shell.os.killpg") as killpg,
    ):
        pgid = _signal_process_tree(process)

    assert pgid is None
    killpg.assert_not_called()
    process.kill.assert_called_once()


def test_reap_process_group_uses_waitid_p_pgid():
    with patch("nanobot.agent.tools.shell.os") as mock_os:
        mock_os.waitid = MagicMock(side_effect=[object(), None])
        mock_os.P_PGID = 2
        mock_os.WEXITED = 4
        mock_os.WNOHANG = 1
        _reap_process_group(4242)
        assert mock_os.waitid.call_count == 2
        mock_os.waitid.assert_called_with(2, 4242, 4 | 1)


def test_reap_process_group_noops_without_waitid():
    with patch("nanobot.agent.tools.shell.os") as mock_os:
        mock_os.waitid = None
        mock_os.P_PGID = 2
        mock_os.WEXITED = 4
        mock_os.WNOHANG = 1
        _reap_process_group(1)  # must not raise


@pytest.mark.asyncio
async def test_kill_process_skips_kill_when_already_exited():
    """Already-dead processes must not raise ProcessLookupError from kill()."""
    process = AsyncMock()
    process.pid = 4242
    process.returncode = 0
    process.kill = MagicMock(side_effect=ProcessLookupError("already dead"))

    with (
        patch("nanobot.agent.tools.shell._reap_pid") as reap,
        patch("nanobot.agent.tools.shell._signal_process_tree") as signal_tree,
    ):
        await ExecTool._kill_process(process)

    signal_tree.assert_not_called()
    process.kill.assert_not_called()
    process.wait.assert_not_called()
    reap.assert_called_once_with(4242)


@pytest.mark.asyncio
async def test_kill_process_kills_and_reaps_live_process():
    process = AsyncMock()
    process.pid = 77
    process.returncode = None
    process.kill = MagicMock()
    process.wait = AsyncMock(return_value=0)

    with (
        patch("nanobot.agent.tools.shell._reap_pid") as reap,
        patch("nanobot.agent.tools.shell._reap_process_group") as reap_group,
        patch(
            "nanobot.agent.tools.shell._signal_process_tree",
            return_value=77,
        ) as signal_tree,
    ):
        await ExecTool._kill_process(process)

    signal_tree.assert_called_once_with(process)
    process.wait.assert_awaited()
    reap_group.assert_called_once_with(77)
    reap.assert_called_once_with(77)


@pytest.mark.asyncio
async def test_kill_process_reaps_even_if_signal_is_noop():
    """Signal path may find the child already gone; wait + reap still run."""
    process = AsyncMock()
    process.pid = 88
    process.returncode = None
    process.wait = AsyncMock(return_value=0)

    with (
        patch("nanobot.agent.tools.shell._reap_pid") as reap,
        patch("nanobot.agent.tools.shell._reap_process_group") as reap_group,
        patch(
            "nanobot.agent.tools.shell._signal_process_tree",
            return_value=None,
        ) as signal_tree,
    ):
        await ExecTool._kill_process(process)

    signal_tree.assert_called_once_with(process)
    process.wait.assert_awaited()
    reap_group.assert_not_called()
    reap.assert_called_once_with(88)


@pytest.mark.asyncio
async def test_execute_reaps_after_normal_completion():
    mock_proc = AsyncMock()
    mock_proc.pid = 1001
    mock_proc.communicate.return_value = (b"ok\n", b"")
    mock_proc.returncode = 0

    with (
        patch.object(ExecTool, "_spawn", return_value=mock_proc),
        patch.object(ExecTool, "_guard_command", return_value=None),
        patch("nanobot.agent.tools.shell._reap_pid") as reap,
    ):
        tool = ExecTool()
        result = await tool.execute(command="echo ok")

    assert "ok" in result
    assert "Exit code: 0" in result
    reap.assert_called_with(1001)


@pytest.mark.asyncio
async def test_execute_timeout_kills_and_reaps():
    mock_proc = AsyncMock()
    mock_proc.pid = 1002
    mock_proc.returncode = None
    mock_proc.communicate.side_effect = asyncio.TimeoutError()
    mock_proc.wait = AsyncMock(return_value=-9)

    with (
        patch.object(ExecTool, "_spawn", return_value=mock_proc),
        patch.object(ExecTool, "_guard_command", return_value=None),
        patch("nanobot.agent.tools.shell._reap_pid") as reap,
        patch("nanobot.agent.tools.shell._reap_process_group"),
        patch(
            "nanobot.agent.tools.shell._signal_process_tree",
            return_value=1002,
        ) as signal_tree,
    ):
        tool = ExecTool(timeout=1)
        result = await tool.execute(command="sleep 99", timeout=1)

    assert "timed out" in result.lower()
    signal_tree.assert_called_once_with(mock_proc)
    reap.assert_called_with(1002)


@pytest.mark.asyncio
async def test_execute_exception_after_success_does_not_raise_on_dead_process():
    """Generic except must return ToolResult.error, not ProcessLookupError."""
    mock_proc = AsyncMock()
    mock_proc.pid = 1003
    mock_proc.communicate = AsyncMock(return_value=(b"out\n", b""))
    mock_proc.returncode = 0
    mock_proc.kill = MagicMock(side_effect=ProcessLookupError("already dead"))

    with (
        patch.object(ExecTool, "_spawn", return_value=mock_proc),
        patch.object(ExecTool, "_guard_command", return_value=None),
        patch("nanobot.agent.tools.shell._reap_pid") as reap,
        patch(
            "nanobot.agent.tools.shell.clamp_session_int",
            side_effect=RuntimeError("boom after exit"),
        ),
    ):
        tool = ExecTool()
        result = await tool.execute(command="echo out")

    assert "Error executing command" in result
    assert "boom after exit" in result
    mock_proc.kill.assert_not_called()
    assert reap.called


@pytest.mark.asyncio
async def test_execute_exception_during_communicate_kills_live_process():
    mock_proc = AsyncMock()
    mock_proc.pid = 1004
    mock_proc.returncode = None
    mock_proc.communicate.side_effect = OSError("pipe broken")
    mock_proc.wait = AsyncMock(return_value=-1)

    with (
        patch.object(ExecTool, "_spawn", return_value=mock_proc),
        patch.object(ExecTool, "_guard_command", return_value=None),
        patch("nanobot.agent.tools.shell._reap_pid") as reap,
        patch("nanobot.agent.tools.shell._reap_process_group"),
        patch(
            "nanobot.agent.tools.shell._signal_process_tree",
            return_value=1004,
        ) as signal_tree,
    ):
        tool = ExecTool()
        result = await tool.execute(command="broken")

    assert "Error executing command" in result
    assert "pipe broken" in result
    signal_tree.assert_called_once_with(mock_proc)
    reap.assert_called_with(1004)


def _mock_session_process(*, pid: int, returncode: int | None):
    process = AsyncMock()
    process.pid = pid
    process.returncode = returncode
    process.kill = MagicMock()
    process.wait = AsyncMock(return_value=returncode if returncode is not None else -9)
    # Constructor starts stream readers; closed streams exit immediately.
    process.stdout = None
    process.stderr = None
    process.stdin = None
    return process


@pytest.mark.asyncio
async def test_exec_session_kill_reaps():
    process = _mock_session_process(pid=2001, returncode=None)
    session = _ExecSession(
        session_id="abc",
        process=process,
        command="sleep 1",
        cwd="/tmp",
        timeout=30,
        owner_session_key=None,
    )
    try:
        with (
            patch("nanobot.agent.tools.shell._reap_pid") as reap,
            patch("nanobot.agent.tools.shell._reap_process_group"),
            patch(
                "nanobot.agent.tools.shell._signal_process_tree",
                return_value=2001,
            ) as signal_tree,
        ):
            await session.kill()
        signal_tree.assert_called_once_with(process)
        reap.assert_called_once_with(2001)
    finally:
        await asyncio.gather(session._stdout_task, session._stderr_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_exec_session_kill_reaps_if_process_exits_before_kill():
    process = _mock_session_process(pid=2002, returncode=None)
    process.kill.side_effect = ProcessLookupError("raced exit")
    session = _ExecSession(
        session_id="raced",
        process=process,
        command="sleep 1",
        cwd="/tmp",
        timeout=30,
        owner_session_key=None,
    )
    try:
        with patch("nanobot.agent.tools.shell._reap_pid") as reap:
            await session.kill()
        reap.assert_called_once_with(2002)
    finally:
        await asyncio.gather(session._stdout_task, session._stderr_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_exec_session_kill_waits_for_reader_tasks():
    process = _mock_session_process(pid=2003, returncode=None)
    session = _ExecSession(
        session_id="readers",
        process=process,
        command="sleep 1",
        cwd="/tmp",
        timeout=30,
        owner_session_key=None,
    )
    await asyncio.gather(session._stdout_task, session._stderr_task)

    release = asyncio.Event()

    async def reader() -> None:
        await release.wait()

    async def wait_for_process() -> int:
        release.set()
        process.returncode = -9
        return -9

    session._stdout_task = asyncio.create_task(reader())
    session._stderr_task = asyncio.create_task(reader())
    process.wait = AsyncMock(side_effect=wait_for_process)
    try:
        await session.kill()

        assert session._stdout_task.done()
        assert session._stderr_task.done()
    finally:
        release.set()
        await asyncio.gather(session._stdout_task, session._stderr_task)


@pytest.mark.asyncio
async def test_exec_session_poll_reaps_after_exit():
    process = _mock_session_process(pid=2002, returncode=0)
    session = _ExecSession(
        session_id="def",
        process=process,
        command="echo hi",
        cwd="/tmp",
        timeout=30,
        owner_session_key=None,
    )
    try:
        with patch("nanobot.agent.tools.shell._reap_pid") as reap:
            poll = await session.poll(yield_time_ms=0, max_output_chars=1000)
        assert poll.done is True
        assert poll.exit_code == 0
        reap.assert_called_once_with(2002)
    finally:
        await asyncio.gather(session._stdout_task, session._stderr_task, return_exceptions=True)


def _direct_children() -> list[tuple[int, str, str]]:
    """Return (pid, name, state) for processes whose parent is this test process."""
    mypid = os.getpid()
    children: list[tuple[int, str, str]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return children
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        meta: dict[str, str] = {}
        for line in status.splitlines():
            if line.startswith(("Name:", "State:", "PPid:")):
                key, _, value = line.partition(":")
                meta[key] = value.strip()
        try:
            ppid = int(meta.get("PPid", "").split()[0])
        except (ValueError, IndexError):
            continue
        if ppid == mypid:
            name = meta.get("Name", "?")
            state = meta.get("State", "?")
            children.append((int(entry.name), name, state))
    return children


def _harvest_zombies() -> list[tuple[int, int]]:
    reaped: list[tuple[int, int]] = []
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            break
        reaped.append((pid, status))
    return reaped


def _enable_child_subreaper() -> bool:
    """Become a subreaper so orphans reparent to us (Docker-like). Best-effort."""
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        # PR_SET_CHILD_SUBREAPER = 36 on Linux
        return libc.prctl(36, 1, 0, 0, 0) == 0
    except Exception:
        return False


@pytest.mark.skipif(sys.platform == "win32", reason="zombie reaping is a Unix waitpid concern")
@pytest.mark.asyncio
async def test_real_timeout_leaves_no_zombie(tmp_path):
    """Integration: simple timed-out sleep must not leave a defunct child."""
    tool = ExecTool(working_dir=str(tmp_path), timeout=1)
    result = await tool.execute(command="sleep 30", timeout=1)
    assert "timed out" in result.lower()

    await asyncio.sleep(0.05)
    assert _harvest_zombies() == []
    leftovers = [c for c in _direct_children() if c[1] in {"sleep", "bash", "sh", "cat"}]
    assert leftovers == [], f"leftover children after simple timeout: {leftovers}"


@pytest.mark.skipif(sys.platform == "win32", reason="process-group kill is a Unix concern")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "sleep 30 | cat",
        "sleep 30 & wait",
        "(sleep 30)",
        "bash -c 'sleep 30; true'",
    ],
)
async def test_real_timeout_kills_pipeline_descendants(tmp_path, command):
    """Timeout must kill the whole process group, not just the outer shell.

    Pipelines and background jobs leave grandchildren when only the shell
    is SIGKILL'd. With start_new_session + killpg those descendants die
    with the shell and never reparent (or zombie under a subreaper).
    """
    _enable_child_subreaper()
    # Drain any pre-existing zombies from earlier tests in this process.
    _harvest_zombies()

    tool = ExecTool(working_dir=str(tmp_path), timeout=1)
    result = await tool.execute(command=command, timeout=1)
    assert "timed out" in result.lower()

    # Give the kernel a beat to reap / reparent.
    await asyncio.sleep(0.1)

    zombies = _harvest_zombies()
    assert zombies == [], f"unreaped zombies after {command!r}: {zombies}"

    leftovers = [
        c
        for c in _direct_children()
        if c[1] in {"sleep", "bash", "sh", "cat", "dash"}
    ]
    assert leftovers == [], f"orphan descendants after {command!r}: {leftovers}"
