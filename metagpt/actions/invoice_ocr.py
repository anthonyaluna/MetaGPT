#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

"""
@Time    : 2023/9/21 18:10:20
@Author  : Stitch-z
@File    : invoice_ocr.py
@Describe : Actions of the invoice ocr assistant.
"""

import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional

import aiofiles
import pandas as pd
from paddleocr import PaddleOCR

from metagpt.actions import Action
from metagpt.const import INVOICE_OCR_TABLE_PATH
from metagpt.logs import logger
from metagpt.prompts.invoice_ocr import (
    EXTRACT_OCR_MAIN_INFO_PROMPT,
    REPLY_OCR_QUESTION_PROMPT,
)
from metagpt.utils.common import OutputParser


class InvoiceOCR(Action):
    """Action class for performing OCR on invoice files, including zip, PDF, png, and jpg files.

    Args:
        name: The name of the action. Defaults to an empty string.
        language: The language for OCR output. Defaults to "ch" (Chinese).

    """

    name: str = "InvoiceOCR"
    i_context: Optional[str] = None

    @staticmethod
    async def _check_file_type(file_path: Path) -> str:
        """Check the file type of the given filename.

        Args:
            file_path: The path of the file.

        Returns:
            The file type based on FileExtensionType enum.

        Raises:
            Exception: If the file format is not zip, pdf, png, or jpg.
        """
        ext = file_path.suffix
        if ext not in [".zip", ".pdf", ".png", ".jpg"]:
            raise Exception("The invoice format is not zip, pdf, png, or jpg")

        return ext

    @staticmethod
    async def _unzip(file_path: Path) -> Path:
        """Unzip a file and return the path to the unzipped directory.

        Args:
            file_path: The path to the zip file.

        Returns:
            The path to the unzipped directory.
        """
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            members = []
            files = set()
            directories = set()
            for zip_info in zip_ref.infolist():
                # Preserve UTF-8 names; only legacy ZIP names need the CP437-to-GBK workaround.
                if zip_info.flag_bits & 0x800:
                    name = zip_info.filename
                else:
                    name = zip_info.filename.encode("cp437").decode("gbk")
                windows_name = PureWindowsPath(name)
                parts = PurePosixPath(name.replace("\\", "/")).parts
                if (
                    "\x00" in zip_info.orig_filename
                    or windows_name.drive
                    or windows_name.root
                    or any(
                        part == ".." or ":" in part or part.endswith((" ", ".")) or PureWindowsPath(part).is_reserved()
                        for part in parts
                    )
                ):
                    raise ValueError("Unsafe invoice ZIP member path")
                file_type = stat.S_IFMT(zip_info.external_attr >> 16)
                if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ValueError("Unsupported invoice ZIP member type")
                if not parts:
                    continue
                relative_name = Path(*parts)
                key = os.path.normcase(str(relative_name))
                parents = {os.path.normcase(str(parent)) for parent in relative_name.parents if parent != Path(".")}
                is_directory = zip_info.is_dir() or name.endswith("\\") or file_type == stat.S_IFDIR
                if key in files or parents.intersection(files) or (not is_directory and key in directories):
                    raise ValueError("Colliding invoice ZIP member paths")
                if is_directory:
                    directories.update(parents | {key})
                elif relative_name.suffix:
                    files.add(key)
                    directories.update(parents)
                    members.append((zip_info, relative_name))

            # Validate the whole archive before creating any extraction output.
            extraction_root = file_path.parent.resolve() / "unzip_invoices"
            if extraction_root.resolve() != extraction_root:
                raise ValueError("Invoice extraction root must not be a link")
            extraction_root.mkdir(exist_ok=True)
            if extraction_root.resolve() != extraction_root:
                raise ValueError("Invoice extraction root must not be a link")
            file_directory = Path(tempfile.mkdtemp(prefix="invoice-", dir=extraction_root))
            try:
                for zip_info, relative_name in members:
                    content = zip_ref.read(zip_info)
                    current = file_directory
                    if current.resolve() != current:
                        raise ValueError("Invoice extraction directory must not be a link")
                    for part in relative_name.parts[:-1]:
                        current = current / part
                        current.mkdir(exist_ok=True)
                        if current.resolve() != current:
                            raise ValueError("Invoice extraction parent must not be a link")
                    # Exclusive creation refuses existing files, symlinks and hardlinks.
                    async with aiofiles.open(current / relative_name.name, mode="xb") as writer:
                        await writer.write(content)
            except BaseException:
                shutil.rmtree(file_directory)
                raise

        logger.info(f"unzip_path: {file_directory}")
        return file_directory

    @staticmethod
    async def _ocr(invoice_file_path: Path):
        ocr = PaddleOCR(use_angle_cls=True, lang="ch", page_num=1)
        ocr_result = ocr.ocr(str(invoice_file_path), cls=True)
        for result in ocr_result[0]:
            result[1] = (result[1][0], round(result[1][1], 2))  # round long confidence scores to reduce token costs
        return ocr_result

    async def run(self, file_path: Path, *args, **kwargs) -> list:
        """Execute the action to identify invoice files through OCR.

        Args:
            file_path: The path to the input file.

        Returns:
            A list of OCR results.
        """
        file_ext = await self._check_file_type(file_path)

        if file_ext == ".zip":
            # OCR recognizes zip batch files
            unzip_path = await self._unzip(file_path)
            ocr_list = []
            for root, _, files in os.walk(unzip_path):
                for filename in files:
                    invoice_file_path = Path(root) / Path(filename)
                    # Identify files that match the type
                    if Path(filename).suffix in [".zip", ".pdf", ".png", ".jpg"]:
                        ocr_result = await self._ocr(str(invoice_file_path))
                        ocr_list.append(ocr_result)
            return ocr_list

        else:
            #  OCR identifies single file
            ocr_result = await self._ocr(file_path)
            return [ocr_result]


class GenerateTable(Action):
    """Action class for generating tables from OCR results.

    Args:
        name: The name of the action. Defaults to an empty string.
        language: The language used for the generated table. Defaults to "ch" (Chinese).

    """

    name: str = "GenerateTable"
    i_context: Optional[str] = None
    language: str = "ch"

    async def run(self, ocr_results: list, filename: str, *args, **kwargs) -> dict[str, str]:
        """Processes OCR results, extracts invoice information, generates a table, and saves it as an Excel file.

        Args:
            ocr_results: A list of OCR results obtained from invoice processing.
            filename: The name of the output Excel file.

        Returns:
            A dictionary containing the invoice information.

        """
        table_data = []
        pathname = INVOICE_OCR_TABLE_PATH
        pathname.mkdir(parents=True, exist_ok=True)

        for ocr_result in ocr_results:
            # Extract invoice OCR main information
            prompt = EXTRACT_OCR_MAIN_INFO_PROMPT.format(ocr_result=ocr_result, language=self.language)
            ocr_info = await self._aask(prompt=prompt)
            invoice_data = OutputParser.extract_struct(ocr_info, dict)
            if invoice_data:
                table_data.append(invoice_data)

        # Generate Excel file
        filename = f"{filename.split('.')[0]}.xlsx"
        full_filename = f"{pathname}/{filename}"
        df = pd.DataFrame(table_data)
        df.to_excel(full_filename, index=False)
        return table_data


class ReplyQuestion(Action):
    """Action class for generating replies to questions based on OCR results.

    Args:
        name: The name of the action. Defaults to an empty string.
        language: The language used for generating the reply. Defaults to "ch" (Chinese).

    """

    language: str = "ch"

    async def run(self, query: str, ocr_result: list, *args, **kwargs) -> str:
        """Reply to questions based on ocr results.

        Args:
            query: The question for which a reply is generated.
            ocr_result: A list of OCR results.

        Returns:
            A reply result of string type.
        """
        prompt = REPLY_OCR_QUESTION_PROMPT.format(query=query, ocr_result=ocr_result, language=self.language)
        resp = await self._aask(prompt=prompt)
        return resp
