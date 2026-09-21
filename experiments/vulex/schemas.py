from dataclasses import dataclass

LINUX_PROJECT = "torvalds/linux"


@dataclass(frozen=True)
class ReposVulRecord:
    record_id: str
    project: str
    cve_id: str
    cwe: str
    commit_id: str
    file_path: str
    function_name: str
    anchor: str
    commit_message: str
    publish_date: str
    target_function: str
    target_function_source: str

@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    project: str
    query: str
    knowledge: str
    response: str
    anchor: str
    cwe: str
    privilege: int
    pool: str
    source_record_id: str

@dataclass(frozen=True)
class OrdinaryMemoryRecord:
    record_id: str
    project: str
    query: str
    knowledge: str
    response: str
    anchor: str

@dataclass(frozen=True)
class Probe:
    phase: str
    text: str

@dataclass(frozen=True)
class ParsedTriple:
    location: str
    cwe: str
    code_evidence: str


@dataclass(frozen=True)
class RunConfig:
    budget: int
    agent_model: str
    sss_model: str
    visibility_split: str = ""
    cwe_source: str = ""
    cap_budget: int = 0
    skip_cap_probes: bool = False
    repo_worktree: str = ""
    ordinary_memory: str = ""
    cap_anchor_bank: str = ""
    sss_payload_bank: str = ""
    openai_api_base: str = ""
    openclaw_served_memory_policy: str = "configured"
    openclaw_memory_search: str = "bm25_fts"
    openclaw_embedding_api_base: str = ""
    openclaw_embedding_model: str = ""
    docker_image: str = ""
    agent_timeout: int = 600
