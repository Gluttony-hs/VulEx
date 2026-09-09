from dataclasses import InitVar, dataclass

LINUX_PROJECT = "torvalds/linux"
SUPPORTED_METHOD = "vulex"
SUPPORTED_METHODS = {"vulex"}
OPENCLAW_NATIVE_BACKEND = "openclaw_native_memory"
OPENCLAW_NATIVE_RETRIEVER = "openclaw_native"
OPENCLAW_SERVED_MEMORY_POLICIES = {"configured", "empty"}


def require_linux_project(project: str) -> None:
    """Validate that the input project is the only supported project, `torvalds/linux`."""
    if project != LINUX_PROJECT:
        raise ValueError(f"only {LINUX_PROJECT} is supported in this experiment, got {project!r}")


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

    def __post_init__(self) -> None:
        """Validate ReposVul project constraints during construction."""
        require_linux_project(self.project)
        if self.target_function_source not in {"function_before", "code_before_patch"}:
            raise ValueError(f"unsupported target_function_source {self.target_function_source!r}")
        if not self.target_function.strip():
            raise ValueError("target_function must be non-empty")


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

    def __post_init__(self) -> None:
        """Validate memory project, privilege, and pool fields during construction."""
        require_linux_project(self.project)
        if not self.query.strip():
            raise ValueError("query must be non-empty")
        if not self.knowledge.strip():
            raise ValueError("knowledge must be non-empty")
        if not self.response.strip():
            raise ValueError("response must be non-empty")
        if self.privilege not in (0, 1):
            raise ValueError(f"privilege must be 0 or 1, got {self.privilege!r}")
        if self.pool not in {"public", "embargoed", "silent"}:
            raise ValueError(f"unsupported pool {self.pool!r}")


@dataclass(frozen=True)
class OrdinaryMemoryRecord:
    record_id: str
    project: str
    query: str
    knowledge: str
    response: str
    anchor: str
    source_file: str
    source_function: str
    source_hash: str
    sampling_seed: int

    def __post_init__(self) -> None:
        """Validate ordinary-code memory project, anchor, and required text fields."""
        require_linux_project(self.project)
        if "::" not in self.anchor:
            raise ValueError("ordinary memory anchor must be file::function")
        source_file, source_function = self.anchor.split("::", 1)
        if not source_file or not source_function:
            raise ValueError("ordinary memory anchor must be file::function")
        if source_file != self.source_file or source_function != self.source_function:
            raise ValueError("ordinary memory anchor must match source_file and source_function")
        if not self.query.strip():
            raise ValueError("query must be non-empty")
        if not self.knowledge.strip():
            raise ValueError("knowledge must be non-empty")
        if not self.response.strip():
            raise ValueError("response must be non-empty")

@dataclass(frozen=True)
class Probe:
    method: str
    phase: str
    anchor: str
    cwe: str
    text: str
    style_payload: str = ""

    def __post_init__(self) -> None:
        """Restrict probes to the VulEx CAP and SSS phases."""
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(f"unsupported method {self.method!r}")
        if self.method == "vulex" and self.phase not in {"cap", "sss"}:
            raise ValueError(f"phase must be 'cap' or 'sss' for vulex, got {self.phase!r}")


@dataclass(frozen=True)
class ParsedTriple:
    location: str
    cwe: str
    code_evidence: str


@dataclass(frozen=True)
class RunConfig:
    project: str
    method: str
    memory_size: int
    budget: int
    seed: int
    agent_model: str
    sss_model: str
    summarization_model: str
    visibility_split: str = ""
    memory_split: str = ""
    cwe_source: str = ""
    cap_budget: int = 0
    skip_cap_probes: bool = False
    sss_payload_anchor_limit: int = 1
    repo_worktree: str = ""
    ordinary_memory: str = ""
    cap_anchor_bank: str = ""
    cap_anchor_bank_sha256: str = ""
    sss_anchor_selection: str = "codebert_repo_mean"
    sss_codebert_model: str = ""
    sss_codebert_cache_dir: str = ""
    sss_codebert_repo_worktree: str = ""
    sss_codebert_prior_weight: float = 0.50
    sss_codebert_repo_demand_count: int = 512
    sss_codebert_content_window: int = 512
    sss_codebert_chunk_step: int = 512
    sss_codebert_pooling: str = "cls"
    sss_payload_bank: str = ""
    sss_payload_bank_sha256: str = ""
    openai_api_base: str = ""
    openclaw_served_memory_policy: str = "configured"
    openclaw_memory_search: str = "bm25_fts"
    openclaw_embedding_api_base: str = ""
    openclaw_embedding_model: str = ""
    docker_image: str = ""
    agent_timeout: int = 600
    allow_inline_sss_payload_build: InitVar[bool] = False

    def __post_init__(self, allow_inline_sss_payload_build: bool) -> None:
        """Validate the formal run configuration; an empty repo_worktree selects an isolated empty workspace."""
        require_linux_project(self.project)
        if self.method != SUPPORTED_METHOD:
            raise ValueError(f"unsupported method {self.method!r}; this package contains only vulex")
        if self.memory_size <= 0:
            raise ValueError("memory_size must be positive")
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if self.cap_budget < 0:
            raise ValueError("cap_budget must be non-negative")
        if self.cap_budget > self.budget:
            raise ValueError("cap_budget cannot exceed budget")
        if self.sss_payload_anchor_limit <= 0:
            raise ValueError("sss_payload_anchor_limit must be positive")
        if self.agent_timeout <= 0:
            raise ValueError("agent_timeout must be positive")
        if self.sss_anchor_selection != "codebert_repo_mean":
            raise ValueError(f"unsupported sss_anchor_selection {self.sss_anchor_selection!r}")
        if self.sss_payload_bank_sha256 and not self.sss_payload_bank:
            raise ValueError("sss_payload_bank_sha256 requires sss_payload_bank")
        if self.cap_anchor_bank_sha256 and not self.cap_anchor_bank:
            raise ValueError("cap_anchor_bank_sha256 requires cap_anchor_bank")
        for field_name, value in (
            ("cap_anchor_bank_sha256", self.cap_anchor_bank_sha256),
            ("sss_payload_bank_sha256", self.sss_payload_bank_sha256),
        ):
            if not value:
                continue
            digest = value.lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError(f"{field_name} must be a 64-character lowercase hex SHA256 digest")
        if self.sss_codebert_pooling not in {"mean", "cls"}:
            raise ValueError("sss_codebert_pooling must be one of: mean, cls")
        if not 0.0 <= float(self.sss_codebert_prior_weight) <= 1.0:
            raise ValueError("sss_codebert_prior_weight must be in [0, 1]")
        if self.sss_codebert_repo_demand_count <= 0:
            raise ValueError("sss_codebert_repo_demand_count must be positive")
        if self.sss_codebert_content_window <= 0 or self.sss_codebert_chunk_step <= 0:
            raise ValueError("sss_codebert_content_window and sss_codebert_chunk_step must be positive")
        if self.openclaw_served_memory_policy not in OPENCLAW_SERVED_MEMORY_POLICIES:
            raise ValueError("openclaw_served_memory_policy must be one of: configured, empty")
        if self.openclaw_memory_search not in {"bm25_fts", "vector", "hybrid"}:
            raise ValueError("openclaw_memory_search must be one of: bm25_fts, vector, hybrid")
        cap_limit = 0 if self.skip_cap_probes else self.cap_budget or self.budget
        if self.budget > cap_limit and not self.sss_payload_bank and not allow_inline_sss_payload_build:
            raise ValueError("sss_payload_bank is required when the configuration enters the SSS phase")
