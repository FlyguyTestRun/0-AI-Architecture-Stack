"""Central configuration for every layer of the stack.

Configuration is read from environment variables (optionally via a .env file) and
validated once at import time by :func:`get_settings`. Each layer reads only its own
section, which keeps the layers swappable without a shared global.

Every selector below accepts ``auto``. ``auto`` means "use the best backend that is
actually available on this machine, otherwise fall back to the offline one". That is
what allows the reference demo to run on a clean clone with no Docker and no Ollama.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

# Every section reads the same dotenv file. The nested sections are built by
# default_factory, which constructs them independently of the outer Settings, so
# without this they would read only the process environment. The documented
# "copy .env.example to .env" flow would then appear to do nothing: a deployment
# could set a provider explicitly and still silently get the offline fallback.
_DOTENV = ".env"

LLMProvider = Literal["auto", "ollama", "echo"]
VectorBackend = Literal["auto", "qdrant", "chroma", "memory"]
EmbeddingBackend = Literal["auto", "sentence-transformers", "hashing"]
OrchestratorKind = Literal["auto", "langgraph", "crewai", "simple"]


class LLMSettings(BaseSettings):
    """Layer 4: the LLM layer."""

    model_config = SettingsConfigDict(
        env_prefix="ZEROSTACK_LLM_", env_file=_DOTENV, env_file_encoding="utf-8", extra="ignore"
    )

    provider: LLMProvider = "auto"
    model: str = "gemma3:4b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_seconds: float = 120.0


class RAGSettings(BaseSettings):
    """Layer 3: the RAG pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="ZEROSTACK_RAG_", env_file=_DOTENV, env_file_encoding="utf-8", extra="ignore"
    )

    backend: VectorBackend = "auto"
    embedding_backend: EmbeddingBackend = "auto"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    qdrant_url: str = "http://localhost:6333"
    # Where the embedded Qdrant keeps its files when no server is reachable. This is
    # what makes an index survive between CLI invocations without Docker.
    qdrant_path: Path = DEFAULT_DATA_DIR / "qdrant"
    collection: str = "zerostack"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    # auto and hybrid both fuse keyword and vector results. Keyword catches exact
    # terms a vector index never saw, such as part numbers, error codes and policy
    # names; vector catches meaning. Neither alone is sufficient on real documents.
    retrieval_mode: Literal["auto", "hybrid", "vector", "keyword"] = "auto"
    keyword_weight: float = 1.0
    vector_weight: float = 1.0
    # Candidates pulled from each retriever before fusion. Wider than top_k so
    # fusion has room to promote something the other retriever ranked highly.
    fusion_candidates: int = 20
    # The graph tier answers questions whose answer is spread across documents
    # that never mention each other, which passage retrieval cannot reach.
    graph_enabled: bool = True
    graph_hops: int = 2
    graph_max_entities: int = 25
    graph_max_relations: int = 12
    score_threshold: float = 0.0
    # Drop any hit scoring below this fraction of the best hit. An absolute threshold
    # is not portable across embedding backends, a relative one is.
    relevance_ratio: float = 0.5
    corpus_dir: Path = DEFAULT_DATA_DIR / "corpus"
    # Directories the application will ingest from when the caller is untrusted.
    # Empty means "the corpus directory only". The API enforces this; the CLI does
    # not, because an operator with shell access can already read these files and
    # restricting them would add friction without adding safety.
    allowed_ingest_roots: list[Path] = []


class OrchestratorSettings(BaseSettings):
    """Layer 2: the agent orchestrator."""

    model_config = SettingsConfigDict(
        env_prefix="ZEROSTACK_ORCHESTRATOR_",
        env_file=_DOTENV,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    kind: OrchestratorKind = "auto"
    max_steps: int = 6
    enable_tools: bool = True


class DataSettings(BaseSettings):
    """Layer 7: the data layer."""

    model_config = SettingsConfigDict(
        env_prefix="ZEROSTACK_DATA_", env_file=_DOTENV, env_file_encoding="utf-8", extra="ignore"
    )

    sqlite_path: Path = DEFAULT_DATA_DIR / "zerostack.db"
    duckdb_path: Path = DEFAULT_DATA_DIR / "analytics.duckdb"
    supabase_url: str = ""
    supabase_key: str = ""


class ObservabilitySettings(BaseSettings):
    """The cross cutting observability layer."""

    model_config = SettingsConfigDict(
        env_prefix="ZEROSTACK_OBS_", env_file=_DOTENV, env_file_encoding="utf-8", extra="ignore"
    )

    enabled: bool = True
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"
    export_traces: bool = False
    log_level: str = "INFO"
    metrics_enabled: bool = True
    trace_log_path: Path = DEFAULT_DATA_DIR / "traces.jsonl"
    # Caps the in memory span buffer. The JSONL log keeps the full history.
    max_retained_spans: int = 1000


class CostSettings(BaseSettings):
    """Token accounting and spend ceilings."""

    model_config = SettingsConfigDict(
        env_prefix="ZEROSTACK_COST_", env_file=_DOTENV, env_file_encoding="utf-8", extra="ignore"
    )

    enabled: bool = True
    # Zero means no ceiling. A ceiling is checked before a call, not after, so
    # that it is a limit rather than a report of the overspend.
    daily_token_budget: int = 0
    daily_cost_budget_usd: float = 0.0
    # Per million tokens as "model:prompt:completion", comma separated. Prices
    # change often, so they are configuration rather than constants.
    price_table: str = ""


class CacheSettings(BaseSettings):
    """Semantic answer cache."""

    model_config = SettingsConfigDict(
        env_prefix="ZEROSTACK_CACHE_", env_file=_DOTENV, env_file_encoding="utf-8", extra="ignore"
    )

    enabled: bool = True
    # High by default. A loose threshold serves the answer to a question that
    # merely resembles the one asked, which is a correctness bug wearing the
    # costume of a performance win.
    threshold: float = 0.95
    max_entries: int = 500
    ttl_seconds: float = 3600.0


class Settings(BaseSettings):
    """Top level settings object composed of one section per layer."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "zerostack"
    environment: str = Field(default="local", description="local, staging or production")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    cost: CostSettings = Field(default_factory=CostSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)

    def ensure_directories(self) -> None:
        """Create the directories the local backends write into."""
        for path in (
            self.data.sqlite_path.parent,
            self.data.duckdb_path.parent,
            self.observability.trace_log_path.parent,
            self.rag.corpus_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process wide settings singleton."""
    settings = Settings()
    settings.ensure_directories()
    return settings


def reset_settings_cache() -> None:
    """Clear the settings cache. Used by tests that patch the environment."""
    get_settings.cache_clear()
