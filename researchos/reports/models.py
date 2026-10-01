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

    @classmethod
    def from_dict(cls, d: dict) -> "Report":
        """Rebuild a report sent back by the browser (exports are generated without server state).

        Raises ValueError/TypeError/KeyError on malformed input.
        """
        def s(v, limit=20000) -> str:
            if not isinstance(v, str):
                raise TypeError("expected string")
            return v[:limit]

        def ints(v) -> list[int]:
            return [int(x) for x in (v or [])][:500]

        def strs(v, limit=2000) -> list[str]:
            return [s(x, limit) for x in (v or [])][:500]

        o = d["overview"]
        return cls(
            overview=Overview(
                title=s(o["title"], 500), title_source=s(o["title_source"], 20), authors=strs(o["authors"], 200),
                date=s(o["date"], 100) if o.get("date") else None, document_type=s(o["document_type"], 100),
                page_count=int(o["page_count"]), filename=s(o["filename"], 255), size_bytes=int(o["size_bytes"]),
            ),
            sections=[
                Section(
                    key=s(x["key"], 40), heading=s(x["heading"], 200),
                    basis=x["basis"] if x.get("basis") in ("document", "interpretation", "mixed") else "mixed",
                    body_markdown=s(x["body_markdown"]), pages=ints(x.get("pages")),
                    evidence=[Evidence(int(e["page"]), s(e["quote"], 500)) for e in x.get("evidence", [])][:50],
                    supported=bool(x.get("supported", True)),
                )
                for x in d["sections"]
            ][:40],
            metrics=[Metric(s(m["label"], 300), s(m["value"], 200), ints(m.get("pages")), bool(m.get("verified")))
                     for m in d.get("metrics", [])][:40],
            references=[SourceRef(int(r["page"]), strs(r.get("quotes"), 500), strs(r.get("used_in"), 200))
                        for r in d.get("references", [])][:500],
            not_stated=strs(d.get("not_stated"), 200),
            validation_notes=strs(d.get("validation_notes"), 500),
            warnings=strs(d.get("warnings"), 500),
            meta={k: v for k, v in (d.get("meta") or {}).items() if isinstance(v, (str, int, float)) and len(str(v)) < 500},
        )
