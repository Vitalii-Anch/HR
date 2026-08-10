"""
Heading-aware chunking with overlap.

Why this strategy (rationale, also summarized in design-and-evaluation.md):

1. Heading-aware first: policy documents in this corpus are written with
   clear Markdown/HTML headings (##, <h2>) or, for the plain-text document,
   numbered "SECTION N:" headers. Splitting along these boundaries first
   (rather than naive fixed-size windows over the raw file) keeps each chunk
   topically coherent -- e.g. the whole "Blackout Periods" section of the PTO
   policy stays together rather than being arbitrarily split mid-sentence
   across two chunks that land in different vector-search results. This
   directly improves retrieval precision and makes citations meaningful
   (a chunk maps to "PTO Policy > Section 5: Blackout Periods", not to an
   arbitrary byte offset).

2. Overlapping sub-chunks within long sections: some sections (e.g. the
   Remote Work Policy's "Tax and Legal Considerations") are long enough that
   embedding the whole section as one vector would dilute its semantic
   signal and exceed a comfortable chunk size for a small embedding model.
   We further split any section longer than CHUNK_SIZE_CHARS into
   overlapping windows of CHUNK_SIZE_CHARS characters with CHUNK_OVERLAP_CHARS
   of overlap. The overlap ensures a sentence or clause that spans a chunk
   boundary still appears in full in at least one chunk, which reduces the
   chance that a relevant fact is only half-present (and therefore
   under-retrieved) in any single chunk.

3. Character-based (not token-based) windows: this keeps the implementation
   dependency-free (no tokenizer needed at chunk time) and is a reasonable
   proxy given all corpus documents are English prose of similar register.

Chunk size and overlap are intentionally modest (target ~800 chars / ~150
overlap) because the policy sections themselves are already short (most are
well under 800 characters), so in practice most chunks *are* whole sections;
the overlapping-window logic mainly protects the handful of longer sections.
"""
from __future__ import annotations

from dataclasses import dataclass

CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 40  # drop trailing slivers shorter than this after splitting


@dataclass
class Section:
    """A heading-delimited section of a source document, before sub-chunking."""
    heading: str
    text: str
    order: int


@dataclass
class Chunk:
    """A single retrievable chunk with its section heading for citation."""
    heading: str
    text: str
    order: int
    chunk_index: int  # index of this chunk within its section (0 if section wasn't split)


def split_section_into_chunks(
    section: Section,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[Chunk]:
    """Split one Section's text into one or more overlapping Chunks.

    If the section text already fits within `chunk_size`, this returns a
    single chunk equal to the whole section (the common case for this
    corpus, per the module docstring).
    """
    text = section.text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [Chunk(heading=section.heading, text=text, order=section.order, chunk_index=0)]

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if len(piece) >= MIN_CHUNK_CHARS or not chunks:
            chunks.append(Chunk(heading=section.heading, text=piece, order=section.order, chunk_index=idx))
            idx += 1
        if end == len(text):
            break
        start += step
    return chunks


def chunk_sections(sections: list[Section]) -> list[Chunk]:
    """Chunk every section in a document and return a flat, ordered chunk list."""
    all_chunks: list[Chunk] = []
    for section in sections:
        all_chunks.extend(split_section_into_chunks(section))
    return all_chunks
