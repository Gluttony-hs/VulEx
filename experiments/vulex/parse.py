import re

from experiments.vulex.schemas import ParsedTriple

BULLET_RE = re.compile(
    r"^\s*-\s*location:\s*(?P<location>[^;]+);\s*type:\s*(?P<cwe>[^;]+);"
    r"\s*code_evidence:\s*(?P<code_evidence>.+?)\s*$",
    re.IGNORECASE,
)
FIELD_MARKUP_RE = re.compile(r"\*\*(location|type|code_evidence):\*\*", re.IGNORECASE)
FIELD_LINE_RE = re.compile(
    r"^\s*-\s*(?P<field>location|type|code_evidence):\s*(?P<value>.+?)\s*$",
    re.IGNORECASE,
)
CWE_RE = re.compile(r"(?:CWE[-\s]*)?(\d+)", re.IGNORECASE)


def normalize_cwe_text(text: str) -> str:
    """Normalize CWE spellings to `CWE-<number>`."""
    match = CWE_RE.search(text or "")
    if not match:
        raise ValueError(f"cannot normalize CWE value {text!r}")
    return f"CWE-{int(match.group(1))}"


def normalize_space(text: str) -> str:
    """Collapse whitespace for stable comparisons."""
    return " ".join(str(text).strip().split())


def build_triple(location: str, cwe: str, code_evidence: str) -> ParsedTriple | None:
    """Normalize a complete field group into a finding; discard it when CWE parsing fails."""
    cwe_match = CWE_RE.search(cwe or "")
    if not cwe_match:
        return None
    return ParsedTriple(
        location=normalize_space(location),
        cwe=f"CWE-{int(cwe_match.group(1))}",
        code_evidence=normalize_space(code_evidence),
    )


def parse_response(text: str) -> list[ParsedTriple]:
    """Parse the one-line or three-line code_evidence bullet schema."""
    triples: list[ParsedTriple] = []
    pending: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = FIELD_MARKUP_RE.sub(lambda match: f"{match.group(1)}:", raw_line)
        match = BULLET_RE.match(line)
        if match:
            triple = build_triple(
                match.group("location"),
                match.group("cwe"),
                match.group("code_evidence"),
            )
            if triple is not None:
                triples.append(triple)
            pending = {}
            continue

        field_match = FIELD_LINE_RE.match(line)
        if not field_match:
            if line.strip():
                pending = {}
            continue

        field = field_match.group("field").lower()
        value = field_match.group("value")
        if field == "location":
            pending = {"location": value}
        elif field == "type" and set(pending) == {"location"}:
            pending["cwe"] = value
        elif field == "code_evidence" and set(pending) == {"location", "cwe"}:
            triple = build_triple(
                pending["location"],
                pending["cwe"],
                value,
            )
            if triple is not None:
                triples.append(triple)
            pending = {}
        else:
            pending = {}
    return triples
