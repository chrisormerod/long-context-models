#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 19:23:56 2026

@author: cormerod
"""
from __future__ import annotations

import re
from typing import Optional
import pandas as pd
from config import DATA_DIR

_XML_ESCAPE_TABLE = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&apos;",
}


def xml_escape(text: str) -> str:
    """Escape text for use in XML element content."""
    # Order matters: escape & first
    return (
        text.replace("&", _XML_ESCAPE_TABLE["&"])
            .replace("<", _XML_ESCAPE_TABLE["<"])
            .replace(">", _XML_ESCAPE_TABLE[">"])
            .replace('"', _XML_ESCAPE_TABLE['"'])
            .replace("'", _XML_ESCAPE_TABLE["'"])
    )


def to_valid_xml_tag(tag: str) -> str:
    """
    Convert an arbitrary label into a valid XML element name.
    Example: "Position Statement" -> "Position_Statement"
    """
    tag = (tag or "").strip()
    if not tag:
        return "Span"

    # Replace whitespace with underscores, drop other invalid chars
    tag = re.sub(r"\s+", "_", tag)
    tag = re.sub(r"[^A-Za-z0-9_.-]", "", tag)

    # XML names can't start with a digit, hyphen, or period
    if not re.match(r"^[A-Za-z_]", tag):
        tag = f"Tag_{tag}" if tag else "Span"

    return tag


def dataframe_to_annotated_xml(
    df: pd.DataFrame,
    *,
    full_text: Optional[str] = None,
    label_col: str = "discourse_type",
    start_col: str = "discourse_start",
    end_col: str = "discourse_end",
    segment_text_col: str = "discourse_text",
    unannotated_labels: tuple[str, ...] = ("Unannotated",),
) -> str:
    """
    Convert a span dataframe (like the uploaded form) into a single XML-annotated string.

    Two modes:
      1) If `full_text` is provided and start/end columns exist, we insert tags into the
         original text using character offsets.
      2) Otherwise, we concatenate row texts (segment_text_col), wrapping any non-unannotated
         segments.

    Assumes spans are non-overlapping within the text. If overlaps are detected (mode 1),
    raises ValueError.
    """
    if df.empty:
        return ""

    # Mode 1: reconstruct from full_text + offsets
    if full_text is not None and start_col in df.columns and end_col in df.columns:
        spans = df[[start_col, end_col, label_col]].copy()
        spans[start_col] = spans[start_col].astype(int)
        spans[end_col] = spans[end_col].astype(int)
        spans = spans.sort_values([start_col, end_col], kind="stable")

        out_parts: list[str] = []
        cursor = 0

        for _, row in spans.iterrows():
            start = int(row[start_col])
            end = int(row[end_col])
            label = str(row[label_col])

            if start < cursor:
                raise ValueError(f"Overlapping spans detected (start={start} < cursor={cursor}).")
            if start < 0 or end < 0 or end < start or end > len(full_text):
                raise ValueError(f"Invalid span [{start}, {end}) for text length {len(full_text)}.")

            # Unannotated gap before the span
            if start > cursor:
                out_parts.append(xml_escape(full_text[cursor:start]))

            span_text = full_text[start:end]
            if label in unannotated_labels:
                out_parts.append(xml_escape(span_text))
            else:
                tag = to_valid_xml_tag(label)
                # out_parts.append(f"{xml_escape(span_text)}")
                out_parts.append(f"<{tag}>{xml_escape(span_text)}</{tag}>")

            cursor = end

        # Trailing text
        if cursor < len(full_text):
            out_parts.append(xml_escape(full_text[cursor:]))

        return "".join(out_parts)

    # Mode 2: concatenate provided segment texts (no need for full_text)
    if segment_text_col not in df.columns:
        raise ValueError(
            f"Need either full_text+({start_col},{end_col}) or a '{segment_text_col}' column."
        )

    # Try to preserve original order; if start exists, sort by it.
    dfx = df.copy()
    if start_col in dfx.columns:
        dfx = dfx.sort_values([start_col], kind="stable")

    parts: list[str] = []
    for _, row in dfx.iterrows():
        label = str(row.get(label_col, ""))
        seg_text = "" if pd.isna(row[segment_text_col]) else str(row[segment_text_col])

        if label in unannotated_labels or not label:
            parts.append(xml_escape(seg_text))
        else:
            tag = to_valid_xml_tag(label)
            parts.append(f"<{tag}>{xml_escape(seg_text)}</{tag}>")

    return "".join(parts)


def get_persuade():
    train = pd.read_csv(f"{DATA_DIR}/persuade/persuade_train_srctexts.csv")
    test = pd.read_csv(f"{DATA_DIR}/persuade/persuade_corpus_2.0_test.csv")

    import pdb
    pdb.set_trace()
    
