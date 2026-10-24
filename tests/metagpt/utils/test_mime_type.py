"""Offline regressions for MIME detection of untrusted filenames."""

import asyncio
import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from metagpt.tools.libs import shell
from metagpt.utils.common import get_mime_type
from metagpt.utils.repo_to_markdown import is_text_file


@pytest.mark.parametrize(
    "name",
    [
        "note'; echo MIME_CANARY_EXECUTED; #.txt",
        "note' & echo MIME_CANARY_EXECUTED & rem '.txt",
        "$(echo MIME_CANARY_EXECUTED).txt",
        "`echo MIME_CANARY_EXECUTED`.txt",
        "--files-from=other.txt",
        "folder with spaces/notes.txt",
        "résumé 平.txt",
        "notes\nnext-line.txt",
    ],
)
def test_filename_is_data_for_the_native_process(monkeypatch, name):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return type("Result", (), {"stdout": "synthetic: text/plain\n", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(shell.subprocess, "run", run)

    assert asyncio.run(get_mime_type(Path(name), force_read=True)) == "text/plain"
    command, options = calls[0]
    assert options["shell"] is False
    assert command[-2:] == ["--", str(Path(name))]
    assert command[0] == "file"


def test_known_extension_does_not_launch_a_process(monkeypatch):
    execute = AsyncMock()
    monkeypatch.setattr(shell, "shell_execute", execute)

    assert asyncio.run(get_mime_type(Path("photo.png"))) == "image/png"
    execute.assert_not_awaited()


@pytest.mark.parametrize("suffix, expected", [("json", "application/json"), ("yaml", "text/yaml")])
def test_preserves_text_extension_override(monkeypatch, suffix, expected):
    monkeypatch.setattr(shell, "shell_execute", AsyncMock(return_value=("synthetic: text/plain\n", "", 0)))

    assert asyncio.run(get_mime_type(Path(f"note.{suffix}"), force_read=True)) == expected


def test_preserves_stderr_fallback(monkeypatch):
    monkeypatch.setattr(shell, "shell_execute", AsyncMock(return_value=("", "unavailable", 1)))

    assert asyncio.run(get_mime_type(Path("photo.png"), force_read=True)) == "image/png"


def test_missing_file_utility_preserves_extension_fallback(monkeypatch):
    monkeypatch.setattr(shell, "shell_execute", AsyncMock(side_effect=FileNotFoundError("file utility missing")))

    assert asyncio.run(get_mime_type(Path("photo.png"), force_read=True)) == "image/png"


@pytest.mark.parametrize("name", ["note", "--help", "note.unknown_mime_type"])
def test_missing_utility_and_unknown_type_remain_safe_for_textuality_callers(monkeypatch, name):
    monkeypatch.setattr(shell, "shell_execute", AsyncMock(side_effect=FileNotFoundError("file utility missing")))

    assert asyncio.run(is_text_file(Path(name))) == (False, "unknown")


def test_preserves_unknown_on_other_execution_errors(monkeypatch):
    monkeypatch.setattr(shell, "shell_execute", AsyncMock(side_effect=ValueError("synthetic failure")))

    assert asyncio.run(get_mime_type(Path("photo.png"), force_read=True)) == "unknown"


def test_preserves_unknown_extension_stderr_fallback(monkeypatch):
    monkeypatch.setattr(shell, "shell_execute", AsyncMock(return_value=("", "unavailable", 1)))

    assert asyncio.run(get_mime_type(Path("note.unknown_mime_type"), force_read=True)) is None


def test_preserves_reported_mime_and_return_code_behavior(monkeypatch):
    monkeypatch.setattr(
        shell, "shell_execute", AsyncMock(return_value=("synthetic: application/octet-stream\n", "", 7))
    )

    assert asyncio.run(is_text_file(Path("note.txt"))) == (False, "application/octet-stream")


@pytest.mark.skipif(shutil.which("file") is None, reason="Native file utility is not installed")
@pytest.mark.parametrize(
    "name",
    [
        "ordinary notes.txt",
        "note'; echo MIME_CANARY_EXECUTED; #.txt",
        "note' & echo MIME_CANARY_EXECUTED & rem '.txt",
        "--help",
    ],
)
def test_native_file_utility_reads_only_the_literal_filename(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    filename = Path(name)
    filename.write_text("synthetic invoice notes\n", encoding="utf-8")

    assert asyncio.run(get_mime_type(filename, force_read=True)) == "text/plain"


@pytest.mark.skipif(shutil.which("file") is None, reason="Native file utility is not installed")
def test_native_filename_cannot_launch_a_second_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    marker = Path("MIME_CANARY_CREATED")
    name = (
        "note' & copy nul MIME_CANARY_CREATED & rem '.txt"
        if os.name == "nt"
        else "note'; touch MIME_CANARY_CREATED; #.txt"
    )
    filename = Path(name)
    filename.write_text("synthetic invoice notes\n", encoding="utf-8")

    mime = asyncio.run(get_mime_type(filename, force_read=True))

    assert not marker.exists()
    assert mime == "text/plain"
