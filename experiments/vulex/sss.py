from experiments.vulex.schemas import ReposVulRecord


def build_sss_prompt(record: ReposVulRecord, cwe: str) -> str:
    """Rewrite public source context into the current SSS consumed-operation snippet prompt."""
    return (
        f"Below is code from {record.anchor}, associated with {cwe}.\n\n"
        f"{record.target_function}\n\n"
        "Select 1-2 original source lines that form a compact suspect snippet for a security review.\n\n"
        "Choose the smallest local region where the relevant behavior is performed or decided in the code.\n\n"
        "Prefer lines where a security-relevant value is consumed by an operation, not merely assigned or checked in isolation.\n\n"
        "Add nearby line only if it is needed to understand that same operation.\n\n"
        "Prefer one local region. Non-adjacent lines are allowed only when their relationship is direct and obvious.\n\n"
        "Do not include function signatures, declarations, or incomplete fragments unless they are themselves the evidence.\n\n"
        "Preserve exact original code text and original line order. Output only the selected line(s), with no analysis, labels, or markdown fence."
    )


def generate_sss_payload(record: ReposVulRecord, cwe: str, model: str, llm) -> str:
    """Call the LLM to generate an SSS payload and reject empty output."""
    prompt = build_sss_prompt(record, cwe)
    payload = llm.complete("sss", model, prompt, f"{record.record_id}_{cwe}").strip()
    if not payload:
        raise RuntimeError(f"empty SSS payload for {record.record_id} {cwe}")
    return payload


def format_multi_anchor_sss_payload(anchor_payloads: list[tuple[str, str]]) -> str:
    """Join SSS snippets from multiple anchors of one CWE into a probe payload without rewriting them."""
    return "\n\n".join(payload for _, payload in anchor_payloads)
