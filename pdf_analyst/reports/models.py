"""Report data model (produced by validation, consumed by the UI and exporters)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Evidence:
    page: int
    quote: str


@dataclass
class Section:
    key: str
    heading: str
    basis: str                     # "document" | "interpretation" | "mixed"
    body_markdown: str
    pages: list[int] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)   # verbatim quotes verified against the PDF
    supported: bool = True         # False: claims document facts but has no verifiable page reference


@dataclass
class Metric:
    label: str
    value: str
    pages: list[int] = field(default_factory=list)
    verified: bool = True


@dataclass
class Overview:
    title: str
    title_source: str              # "document" | "pdf_metadata" | "filename"
    authors: list[str]
    date: str | None
    document_type: str
    page_count: int
    filename: str
    size_bytes: int


@dataclass
class SourceRef:
    page: int
    quotes: list[str] = field(default_factory=list)
    used_in: list[str] = field(default_factory=list)   # section headings


@dataclass
class Report:
    overview: Overview
    sections: list[Section]
    metrics: list[Metric]
    references: list[SourceRef]
    not_stated: list[str]
    validation_notes: list[str]      # what the validator corrected/removed
    warnings: list[str]              # extraction / coverage warnings for the user
    meta: dict                       # model, timing, chunk count, llm calls, sha256, ...

    def to_dict(self) -> dict:
        return asdict(self)
