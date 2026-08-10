"""
Format-specific parsers that turn a source policy document (Markdown, HTML,
or plain text) into a common (doc_id, title, source_format, sections)
representation, ready for heading-aware chunking (see chunking.py).

Supporting multiple source formats is a course requirement and also a
realistic simulation of a real HR knowledge base, where policy documents
accumulate over time in whatever format the authoring tool of the day
produced (Word exports to HTML, plain-text legacy memos, Markdown wiki
pages, etc). Each parser's only job is to identify a title, a stable doc_id,
and a heading-ordered list of (heading, text) sections; chunking and
embedding are format-agnostic from that point on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.rag.chunking import Section

DOC_ID_RE = re.compile(r"doc\s*id\s*:\s*`?([a-z0-9\-]+)`?", re.IGNORECASE)


@dataclass
class ParsedDocument:
    doc_id: str
    title: str
    source_format: str
    sections: list[Section]


def _fallback_doc_id(path: Path) -> str:
    return path.stem.replace("_", "-").lower()


def parse_markdown(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    title = path.stem
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    doc_id_match = DOC_ID_RE.search(text)
    doc_id = doc_id_match.group(1) if doc_id_match else _fallback_doc_id(path)

    # Split on H2 ("## ") headings. Content before the first H2 becomes an
    # "Overview" section (title + doc metadata + any preamble).
    sections: list[Section] = []
    heading = "Overview"
    buf: list[str] = []
    order = 0

    def flush():
        nonlocal buf, heading, order
        content = "\n".join(buf).strip()
        if content:
            sections.append(Section(heading=heading, text=content, order=order))
            order += 1
        buf = []

    for line in lines:
        if line.startswith("## "):
            flush()
            heading = line[3:].strip()
            # Strip a leading "1. " style numeral for a cleaner citation label.
            heading = re.sub(r"^\d+\.\s*", "", heading)
            continue
        buf.append(line)
    flush()

    return ParsedDocument(doc_id=doc_id, title=title, source_format="markdown", sections=sections)


def parse_html(path: Path) -> ParsedDocument:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else path.stem

    doc_id = None
    meta_tag = soup.find("meta", attrs={"name": "doc-id"})
    if meta_tag and meta_tag.get("content"):
        doc_id = meta_tag["content"].strip()
    if not doc_id:
        match = DOC_ID_RE.search(soup.get_text())
        doc_id = match.group(1) if match else _fallback_doc_id(path)

    body = soup.find("body") or soup

    sections: list[Section] = []
    heading = "Overview"
    buf: list[str] = []
    order = 0

    def flush():
        nonlocal buf, heading, order
        content = "\n".join(buf).strip()
        if content:
            sections.append(Section(heading=heading, text=content, order=order))
            order += 1
        buf = []

    for el in body.find_all(["h1", "h2", "p", "li", "table"], recursive=True):
        if el.name == "h1":
            continue
        if el.name == "h2":
            flush()
            heading = re.sub(r"^\d+\.\s*", "", el.get_text(strip=True))
            continue
        text = el.get_text(" ", strip=True)
        if text:
            buf.append(text)
    flush()

    return ParsedDocument(doc_id=doc_id, title=title, source_format="html", sections=sections)


SECTION_HEADER_RE = re.compile(r"^SECTION\s+\d+\s*:\s*(.+)$", re.IGNORECASE)


def parse_txt(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    title = lines[0].strip() if lines else path.stem
    doc_id_match = DOC_ID_RE.search(text)
    doc_id = doc_id_match.group(1) if doc_id_match else _fallback_doc_id(path)

    sections: list[Section] = []
    heading = "Overview"
    buf: list[str] = []
    order = 0

    def flush():
        nonlocal buf, heading, order
        content = "\n".join(buf).strip()
        if content:
            sections.append(Section(heading=heading, text=content, order=order))
            order += 1
        buf = []

    for line in lines:
        m = SECTION_HEADER_RE.match(line.strip())
        if m:
            flush()
            heading = m.group(1).strip()
            continue
        buf.append(line)
    flush()

    return ParsedDocument(doc_id=doc_id, title=title, source_format="text", sections=sections)


_PARSERS = {
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".html": parse_html,
    ".htm": parse_html,
    ".txt": parse_txt,
}


def parse_document(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    parser = _PARSERS.get(suffix)
    if parser is None:
        raise ValueError(f"Unsupported corpus file format: {path.suffix} ({path})")
    return parser(path)


def iter_corpus_files(corpus_dir: Path):
    for suffix in _PARSERS:
        yield from sorted(corpus_dir.glob(f"*{suffix}"))
