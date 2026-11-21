#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

"""
@Time    : 2023/10/09 18:40:34
@Author  : Stitch-z
@File    : test_invoice_ocr.py
"""

import asyncio
import os
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path, PureWindowsPath

import aiofiles
import pytest

from metagpt.actions import invoice_ocr
from metagpt.actions.invoice_ocr import GenerateTable, InvoiceOCR, ReplyQuestion
from metagpt.const import TEST_DATA_PATH


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invoice_path",
    [
        Path("invoices/invoice-3.jpg"),
        Path("invoices/invoice-4.zip"),
    ],
)
async def test_invoice_ocr(invoice_path: Path, context):
    invoice_path = TEST_DATA_PATH / invoice_path
    resp = await InvoiceOCR(context=context).run(file_path=Path(invoice_path))
    assert isinstance(resp, list)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("invoice_path", "expected_result"),
    [
        (Path("invoices/invoice-1.pdf"), {"收款人": "小明", "城市": "深圳", "总费用/元": 412.00, "开票日期": "2023年02月03日"}),
    ],
)
async def test_generate_table(invoice_path: Path, expected_result: dict):
    invoice_path = TEST_DATA_PATH / invoice_path
    filename = invoice_path.name
    ocr_result = await InvoiceOCR().run(file_path=Path(invoice_path))
    table_data = await GenerateTable().run(ocr_results=ocr_result, filename=filename)
    assert isinstance(table_data, list)
    table_data = table_data[0]
    assert expected_result["收款人"] == table_data["收款人"]
    assert expected_result["城市"] in table_data["城市"]
    assert float(expected_result["总费用/元"]) == float(table_data["总费用/元"])
    assert expected_result["开票日期"] == table_data["开票日期"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("invoice_path", "query", "expected_result"),
    [(Path("invoices/invoice-1.pdf"), "Invoicing date", "2023年02月03日")],
)
async def test_reply_question(invoice_path: Path, query: dict, expected_result: str):
    invoice_path = TEST_DATA_PATH / invoice_path
    ocr_result = await InvoiceOCR().run(file_path=Path(invoice_path))
    result = await ReplyQuestion().run(query=query, ocr_result=ocr_result)
    assert expected_result in result


def _invoice_zip(tmp_path, entries, name="invoices.zip"):
    archive_path = tmp_path / name
    with zipfile.ZipFile(archive_path, "w") as archive:
        for member, data in entries:
            archive.writestr(member, data)
    return archive_path


@pytest.fixture
def confined_invoice_writes(tmp_path, monkeypatch):
    """Even a failing baseline must never write beyond the disposable fixture."""
    mkdir = Path.mkdir
    open_file = aiofiles.open

    def check(path):
        absolute = Path(os.path.abspath(path))
        try:
            contained = os.path.commonpath([str(absolute), str(tmp_path)]) == str(tmp_path)
        except ValueError:
            contained = False
        if not contained or PureWindowsPath(absolute.name).is_reserved() or ":" in absolute.name:
            raise AssertionError("Archive attempted a write outside the safe fixture")

    def safe_mkdir(path, *args, **kwargs):
        check(path)
        return mkdir(path, *args, **kwargs)

    def safe_open(path, *args, **kwargs):
        check(path)
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", safe_mkdir)
    monkeypatch.setattr(aiofiles, "open", safe_open)


@pytest.mark.parametrize(
    "name",
    [
        "../outside.pdf",
        "nested/../../outside.pdf",
        "..\\outside.pdf",
        "nested\\..\\..\\outside.pdf",
        "/outside.pdf",
        "\\outside.pdf",
        "C:/outside.pdf",
        "C:outside.pdf",
        "\\\\example.invalid\\share\\outside.pdf",
        "invoice.pdf:stream",
        "nested/NUL.pdf",
        "CON.pdf",
        "nested/.. /outside.pdf",
        "nested./receipt.pdf",
        "receipt.pdf ",
    ],
)
def test_unzip_rejects_unsafe_member_names(tmp_path, confined_invoice_writes, name):
    archive = _invoice_zip(tmp_path, [("first.pdf", b"safe"), (name, b"unsafe")])

    with pytest.raises(ValueError):
        asyncio.run(InvoiceOCR._unzip(archive))

    assert not (tmp_path / "unzip_invoices").exists()


def test_unzip_does_not_overwrite_a_victim_outside_its_directory(tmp_path, confined_invoice_writes):
    victim = tmp_path / "victim.pdf"
    victim.write_bytes(b"KEEP")
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    archive = _invoice_zip(uploads, [("../../../victim.pdf", b"CHANGED")])
    rejected = False
    try:
        asyncio.run(InvoiceOCR._unzip(archive))
    except ValueError:
        rejected = True

    assert victim.read_bytes() == b"KEEP"
    assert rejected


def test_unzip_ordinary_nested_invoice_control(tmp_path, confined_invoice_writes):
    archive = _invoice_zip(tmp_path, [("nested/receipt.pdf", b"ordinary invoice")])

    output = asyncio.run(InvoiceOCR._unzip(archive))

    assert (output / "nested/receipt.pdf").read_bytes() == b"ordinary invoice"


def test_unzip_preserves_nested_and_gbk_names_and_suffix_behavior(tmp_path, confined_invoice_writes):
    legacy_name = "票据/发票.pdf".encode("gbk")
    placeholder = "L" * len(legacy_name)
    archive = _invoice_zip(
        tmp_path,
        [
            ("nested/receipt.pdf", b"invoice"),
            (placeholder, b"legacy"),
            ("nested\\second.jpg", b"second"),
            ("other.txt", b"text"),
            ("other.exe", b"data only"),
            ("extensionless", b"ignored"),
            ("empty.pdf/", b""),
        ],
    )
    encoded_archive = archive.read_bytes()
    assert encoded_archive.count(placeholder.encode("ascii")) == 2
    archive.write_bytes(encoded_archive.replace(placeholder.encode("ascii"), legacy_name))

    output = asyncio.run(InvoiceOCR._unzip(archive))

    assert output.parent == tmp_path / "unzip_invoices"
    assert (output / "nested/receipt.pdf").read_bytes() == b"invoice"
    assert (output / "票据/发票.pdf").read_bytes() == b"legacy"
    assert (output / "nested/second.jpg").read_bytes() == b"second"
    assert (output / "other.txt").read_bytes() == b"text"
    assert (output / "other.exe").read_bytes() == b"data only"
    assert not (output / "extensionless").exists()
    assert not (output / "empty.pdf").is_file()


def test_unzip_accepts_standard_utf8_zip_filename(tmp_path, confined_invoice_writes):
    archive = _invoice_zip(tmp_path, [("发票.pdf", b"utf8 invoice")])
    with zipfile.ZipFile(archive) as source:
        assert source.infolist()[0].flag_bits & 0x800

    output = asyncio.run(InvoiceOCR._unzip(archive))

    assert (output / "发票.pdf").read_bytes() == b"utf8 invoice"


@pytest.mark.parametrize(
    "names",
    [
        ["receipt.pdf", "receipt.pdf"],
        ["nested/receipt.pdf", "nested\\receipt.pdf"],
        ["nested/./receipt.pdf", "nested/receipt.pdf"],
        ["parent.pdf", "parent.pdf/receipt.pdf"],
        ["parent.pdf/receipt.pdf", "parent.pdf"],
        ["parent.pdf/", "parent.pdf"],
    ],
)
def test_unzip_rejects_duplicate_or_colliding_destinations_before_writes(tmp_path, confined_invoice_writes, names):
    archive = _invoice_zip(tmp_path, [(name, b"synthetic") for name in names])

    with pytest.raises(ValueError):
        asyncio.run(InvoiceOCR._unzip(archive))

    assert not (tmp_path / "unzip_invoices").exists()


def test_unzip_uses_a_unique_directory_for_each_archive(tmp_path, confined_invoice_writes):
    first = _invoice_zip(tmp_path, [("receipt.pdf", b"first")], "first.zip")
    second = _invoice_zip(tmp_path, [("receipt.pdf", b"second")], "second.zip")

    first_output = asyncio.run(InvoiceOCR._unzip(first))
    second_output = asyncio.run(InvoiceOCR._unzip(second))

    assert first_output != second_output
    assert (first_output / "receipt.pdf").read_bytes() == b"first"
    assert (second_output / "receipt.pdf").read_bytes() == b"second"


def test_unzip_rejects_nul_metadata_before_writes(tmp_path, confined_invoice_writes):
    archive = _invoice_zip(tmp_path, [("receipt.pdfZ", b"synthetic")])
    archive.write_bytes(archive.read_bytes().replace(b"receipt.pdfZ", b"receipt.pdf\x00"))

    with pytest.raises(ValueError):
        asyncio.run(InvoiceOCR._unzip(archive))

    assert not (tmp_path / "unzip_invoices").exists()


def test_unzip_rejects_archive_symlink_metadata(tmp_path, confined_invoice_writes):
    member = zipfile.ZipInfo("link.pdf")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = _invoice_zip(tmp_path, [(member, b"../outside.pdf")])

    with pytest.raises(ValueError):
        asyncio.run(InvoiceOCR._unzip(archive))

    assert not (tmp_path / "unzip_invoices").exists()


def test_unzip_encoding_failure_does_not_leave_a_partial_extraction(tmp_path, confined_invoice_writes):
    archive = _invoice_zip(tmp_path, [("first.pdf", b"first"), ("X.pdf", b"invalid legacy encoding")])
    encoded_archive = archive.read_bytes()
    assert encoded_archive.count(b"X.pdf") == 2
    archive.write_bytes(encoded_archive.replace(b"X.pdf", b"\xff.pdf"))

    with pytest.raises(UnicodeError):
        asyncio.run(InvoiceOCR._unzip(archive))

    assert not (tmp_path / "unzip_invoices").exists()


def _invoice_directory_link(link, target):
    if os.name == "nt":
        executable = shutil.which("powershell.exe")
        if not executable:
            pytest.skip("PowerShell is required for the Windows junction fixture")
        link_literal = str(link).replace("'", "''")
        target_literal = str(target).replace("'", "''")
        script = f"New-Item -ItemType Junction -Path '{link_literal}' -Value '{target_literal}' -ErrorAction Stop"
        result = subprocess.run([executable, "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True)
        if result.returncode:
            pytest.skip("Windows junction creation is unavailable")
    else:
        link.symlink_to(target, target_is_directory=True)


def test_unzip_rejects_a_preexisting_extraction_root_link(tmp_path, confined_invoice_writes):
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "receipt.pdf"
    victim.write_bytes(b"KEEP")
    _invoice_directory_link(tmp_path / "unzip_invoices", outside)
    archive = _invoice_zip(tmp_path, [("receipt.pdf", b"CHANGED")])

    with pytest.raises(ValueError):
        asyncio.run(InvoiceOCR._unzip(archive))

    assert victim.read_bytes() == b"KEEP"
    assert list(outside.iterdir()) == [victim]


def test_unzip_rejects_a_seeded_parent_link(tmp_path, confined_invoice_writes, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "receipt.pdf"
    victim.write_bytes(b"KEEP")
    create_directory = invoice_ocr.tempfile.mkdtemp

    def seeded_directory(*args, **kwargs):
        output = Path(create_directory(*args, **kwargs))
        _invoice_directory_link(output / "nested", outside)
        return str(output)

    monkeypatch.setattr(invoice_ocr.tempfile, "mkdtemp", seeded_directory)
    archive = _invoice_zip(tmp_path, [("nested/receipt.pdf", b"CHANGED")])

    with pytest.raises(ValueError):
        asyncio.run(InvoiceOCR._unzip(archive))

    assert victim.read_bytes() == b"KEEP"


def test_unzip_refuses_to_truncate_a_seeded_hardlink(tmp_path, confined_invoice_writes, monkeypatch):
    victim = tmp_path / "victim.pdf"
    victim.write_bytes(b"KEEP")
    create_directory = invoice_ocr.tempfile.mkdtemp

    def seeded_directory(*args, **kwargs):
        output = Path(create_directory(*args, **kwargs))
        os.link(victim, output / "receipt.pdf")
        return str(output)

    monkeypatch.setattr(invoice_ocr.tempfile, "mkdtemp", seeded_directory)
    archive = _invoice_zip(tmp_path, [("receipt.pdf", b"CHANGED")])

    with pytest.raises(FileExistsError):
        asyncio.run(InvoiceOCR._unzip(archive))

    assert victim.read_bytes() == b"KEEP"


def test_unzip_propagates_write_failure(tmp_path, confined_invoice_writes, monkeypatch):
    open_file = aiofiles.open

    def fail_open(path, *args, **kwargs):
        if Path(path).name == "receipt.pdf":
            raise PermissionError("Synthetic extraction write failure")
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(aiofiles, "open", fail_open)
    archive = _invoice_zip(tmp_path, [("first.pdf", b"first"), ("receipt.pdf", b"synthetic")])

    with pytest.raises(PermissionError, match="Synthetic extraction"):
        asyncio.run(InvoiceOCR._unzip(archive))

    assert not list((tmp_path / "unzip_invoices").glob("invoice-*"))


def test_unzip_reads_corrupt_member_before_creating_a_file(tmp_path, confined_invoice_writes, monkeypatch):
    archive = _invoice_zip(tmp_path, [("receipt.pdf", b"AAAA")])
    archive.write_bytes(archive.read_bytes().replace(b"AAAA", b"BBBB", 1))
    open_file = aiofiles.open
    opened = []

    def record_open(path, *args, **kwargs):
        opened.append(path)
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(aiofiles, "open", record_open)
    with pytest.raises(zipfile.BadZipFile):
        asyncio.run(InvoiceOCR._unzip(archive))

    assert opened == []
    assert not list((tmp_path / "unzip_invoices").glob("invoice-*"))


def test_unzip_preserves_platform_case_semantics(tmp_path, confined_invoice_writes):
    archive = _invoice_zip(tmp_path, [("receipt.pdf", b"lower"), ("RECEIPT.pdf", b"upper")])
    probe = tmp_path / "case_probe"
    probe.write_bytes(b"probe")
    case_insensitive = (tmp_path / "CASE_PROBE").exists()
    probe.unlink()

    if case_insensitive:
        with pytest.raises((ValueError, FileExistsError)):
            asyncio.run(InvoiceOCR._unzip(archive))
        assert not list((tmp_path / "unzip_invoices").glob("invoice-*"))
    else:
        output = asyncio.run(InvoiceOCR._unzip(archive))
        assert (output / "receipt.pdf").read_bytes() == b"lower"
        assert (output / "RECEIPT.pdf").read_bytes() == b"upper"
