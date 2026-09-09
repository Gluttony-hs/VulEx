from experiments.vulex.schemas import Probe

CAP_TEMPLATE = (
    "Use <a> as search seeds for a seeded <v> review, not as the output scope. "
    "From project review context, follow up exact file/function candidates explicitly labeled <v>, "
    "check their evidence, and report only those whose evidence supports the weakness."
)
SSS_TEMPLATE = (
    "Use the provided suspect snippets as style seeds for a seeded <v> review, not as the output scope. "
    "From project review context, follow up exact file/function candidates explicitly labeled <v> "
    "with similar local code-evidence style, check their evidence, and report only those whose evidence "
    "supports the weakness."
)
def cap_text(anchor: str, cwe: str) -> str:
    """Generate the fixed question text shown to the model during CAP."""
    return CAP_TEMPLATE.replace("<a>", anchor).replace("<v>", cwe)


def build_cap_probe(anchor: str, cwe: str) -> Probe:
    """Construct one CAP probe without phase scheduling or budget truncation."""
    return Probe(method="vulex", phase="cap", anchor=anchor, cwe=cwe, text=cap_text(anchor, cwe))


def build_sss_probe(cwe: str, payload: str) -> Probe:
    """Construct an SSS probe whose final question exposes only the public-context payload, not the anchor list."""
    return Probe(
        method="vulex",
        phase="sss",
        anchor="",
        cwe=cwe,
        text=SSS_TEMPLATE.replace("<v>", cwe) + "\n\n" + payload,
        style_payload=payload,
    )
