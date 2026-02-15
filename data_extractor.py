"""
Data Extraction Module for DocMind AI

Extracts structured data from PDFs:
- Tables → pandas DataFrames → CSV
- Charts/images → GPT-4o Vision analysis → numerical data
"""

import os
import base64
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import pdfplumber
import fitz  # PyMuPDF
import pandas as pd


@dataclass
class ExtractedTable:
    """A table extracted from a PDF page."""
    page_number: int
    table_index: int
    dataframe: pd.DataFrame
    raw_data: List[List]

    @property
    def label(self) -> str:
        return f"Page {self.page_number} - Table {self.table_index + 1}"

    def to_csv(self) -> str:
        return self.dataframe.to_csv(index=False)


@dataclass
class ExtractedImage:
    """An image/chart extracted from a PDF page."""
    page_number: int
    image_index: int
    image_bytes: bytes
    width: int
    height: int
    ext: str = "png"
    analysis: Optional[str] = None
    extracted_data: Optional[pd.DataFrame] = None

    @property
    def label(self) -> str:
        return f"Page {self.page_number} - Image {self.image_index + 1}"

    @property
    def base64(self) -> str:
        return base64.b64encode(self.image_bytes).decode("utf-8")

    @property
    def data_uri(self) -> str:
        return f"data:image/{self.ext};base64,{self.base64}"


@dataclass
class ExtractionResult:
    """Complete extraction result for a PDF."""
    file_name: str
    tables: List[ExtractedTable] = field(default_factory=list)
    images: List[ExtractedImage] = field(default_factory=list)
    page_count: int = 0


# ---------------------------------------------------------------------------
# Table Extraction (pdfplumber)
# ---------------------------------------------------------------------------
def extract_tables_from_pdf(pdf_path: str) -> List[ExtractedTable]:
    """
    Extract all tables from a PDF using pdfplumber.
    Returns a list of ExtractedTable objects with clean DataFrames.
    """
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "lines_strict",
                        "horizontal_strategy": "lines_strict",
                    }
                )
                # If strict mode finds nothing, try with default (text-based)
                if not page_tables:
                    page_tables = page.extract_tables()

                for tbl_idx, raw_table in enumerate(page_tables):
                    if not raw_table or len(raw_table) < 2:
                        continue

                    df = _clean_table(raw_table)
                    if df is not None and not df.empty:
                        tables.append(
                            ExtractedTable(
                                page_number=page_num,
                                table_index=tbl_idx,
                                dataframe=df,
                                raw_data=raw_table,
                            )
                        )
    except Exception as e:
        print(f"Table extraction error: {e}")

    return tables


def _clean_table(raw_table: List[List]) -> Optional[pd.DataFrame]:
    """Clean a raw table: use first row as header, strip whitespace, drop empties."""
    try:
        # Clean cells
        cleaned = []
        for row in raw_table:
            cleaned.append([
                str(cell).strip().replace("\n", " ") if cell else ""
                for cell in row
            ])

        # First row as header
        header = cleaned[0]
        data = cleaned[1:]

        # If header is all empty, generate column names
        if all(h == "" for h in header):
            header = [f"Col_{i+1}" for i in range(len(header))]

        # Deduplicate header names
        seen = {}
        unique_header = []
        for h in header:
            name = h if h else "Unnamed"
            if name in seen:
                seen[name] += 1
                unique_header.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 0
                unique_header.append(name)

        df = pd.DataFrame(data, columns=unique_header)

        # Drop rows/cols that are completely empty
        df = df.replace("", pd.NA).dropna(how="all").dropna(axis=1, how="all")
        df = df.fillna("")

        # Try to convert numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col].str.replace(",", "").str.strip())
            except (ValueError, AttributeError):
                pass

        return df if not df.empty else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Image/Chart Extraction (PyMuPDF)
# ---------------------------------------------------------------------------
MIN_IMAGE_SIZE = 100  # Minimum width/height in pixels to consider as a chart


def extract_images_from_pdf(pdf_path: str) -> List[ExtractedImage]:
    """
    Extract images/charts from a PDF using PyMuPDF.
    Filters out tiny images (icons, logos) to focus on charts/figures.
    """
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            img_idx = 0
            for img_info in image_list:
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    if not base_image:
                        continue

                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    # Skip small images (icons, bullets, etc.)
                    if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                        continue

                    image_bytes = base_image["image"]
                    ext = base_image.get("ext", "png")

                    images.append(
                        ExtractedImage(
                            page_number=page_num + 1,
                            image_index=img_idx,
                            image_bytes=image_bytes,
                            width=width,
                            height=height,
                            ext=ext,
                        )
                    )
                    img_idx += 1
                except Exception:
                    continue

        doc.close()
    except Exception as e:
        print(f"Image extraction error: {e}")

    return images


# ---------------------------------------------------------------------------
# Chart Analysis via GPT-4o Vision
# ---------------------------------------------------------------------------
def analyze_chart_with_vision(image: ExtractedImage) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Send a chart image to GPT-4o Vision to extract numerical data.
    Returns (analysis_text, extracted_dataframe_or_None).
    """
    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data extraction specialist. When given a chart or graph image:\n"
                        "1. Describe what the chart shows (type, axes, title).\n"
                        "2. Extract ALL numerical data points visible in the chart.\n"
                        "3. Present the data as a markdown table with clear headers.\n"
                        "4. If exact values aren't readable, provide your best estimate.\n"
                        "5. Note any trends or key observations.\n\n"
                        "IMPORTANT: Always include a markdown table with the extracted data."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract all numerical data from this chart/graph. "
                                "Present the data as a structured markdown table."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image.data_uri, "detail": "high"},
                        },
                    ],
                },
            ],
            max_tokens=2000,
            temperature=0.1,
        )

        analysis = response.choices[0].message.content
        df = _parse_markdown_table(analysis)
        return analysis, df

    except Exception as e:
        return f"Vision analysis error: {e}", None


def _parse_markdown_table(text: str) -> Optional[pd.DataFrame]:
    """Try to parse the first markdown table found in text into a DataFrame."""
    try:
        lines = text.strip().split("\n")
        table_lines = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            if "|" in stripped:
                # Skip separator rows like |---|---|
                if set(stripped.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
                    in_table = True
                    continue
                if stripped.startswith("|"):
                    in_table = True
                    table_lines.append(stripped)
                elif in_table:
                    break
            elif in_table:
                break

        if len(table_lines) < 2:
            return None

        # Parse header
        header = [
            cell.strip() for cell in table_lines[0].split("|")
            if cell.strip()
        ]

        # Parse data rows
        data = []
        for row_line in table_lines[1:]:
            cells = [cell.strip() for cell in row_line.split("|") if cell.strip()]
            if cells:
                data.append(cells)

        if not data:
            return None

        df = pd.DataFrame(data, columns=header[:len(data[0])])

        # Try numeric conversion
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(
                    df[col].str.replace(",", "").str.replace("%", "").str.strip()
                )
            except (ValueError, AttributeError):
                pass

        return df
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Full Extraction Pipeline
# ---------------------------------------------------------------------------
def extract_all(pdf_path: str) -> ExtractionResult:
    """Run the full extraction pipeline on a PDF."""
    file_name = Path(pdf_path).name
    result = ExtractionResult(file_name=file_name)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            result.page_count = len(pdf.pages)
    except Exception:
        pass

    result.tables = extract_tables_from_pdf(pdf_path)
    result.images = extract_images_from_pdf(pdf_path)

    return result
