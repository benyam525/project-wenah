"""
Configuration management for the Wenah compliance framework.

Uses pydantic-settings for environment variable loading and validation.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================================================================
    # Anthropic API Configuration
    # ==========================================================================
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key for Claude access",
    )
    claude_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Claude model to use for analysis",
    )

    # ==========================================================================
    # Vector Database Configuration
    # ==========================================================================
    chroma_persist_directory: str = Field(
        default="./data/vector_db",
        description="Directory for ChromaDB persistence",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )
    chroma_collection_name: str = Field(
        default="civil_rights_laws",
        description="ChromaDB collection name",
    )

    # ==========================================================================
    # API Configuration
    # ==========================================================================
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host",
    )
    api_port: int = Field(
        default=8000,
        description="API server port",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    # ==========================================================================
    # Scoring Configuration
    # ==========================================================================
    rule_engine_base_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Base weight for rule engine scores",
    )
    llm_analysis_base_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Base weight for LLM analysis scores",
    )
    high_confidence_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Threshold for high confidence classification",
    )
    low_confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for low confidence classification",
    )

    # ==========================================================================
    # Guardrails Configuration
    # ==========================================================================
    enable_hallucination_guardrails: bool = Field(
        default=True,
        description="Enable hallucination detection guardrails",
    )
    min_grounding_ratio: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum ratio of claims that must be grounded",
    )
    max_hedging_penalty: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Maximum confidence penalty for hedging language",
    )

    # ==========================================================================
    # Data Paths
    # ==========================================================================
    laws_directory: str = Field(
        default="./data/laws",
        description="Directory containing law YAML files",
    )
    rules_directory: str = Field(
        default="./data/rules",
        description="Directory containing rule YAML files",
    )
    precedents_directory: str = Field(
        default="./data/precedents",
        description="Directory containing precedent YAML files",
    )

    @property
    def laws_path(self) -> Path:
        """Get laws directory as Path object."""
        return Path(self.laws_directory)

    @property
    def rules_path(self) -> Path:
        """Get rules directory as Path object."""
        return Path(self.rules_directory)

    @property
    def precedents_path(self) -> Path:
        """Get precedents directory as Path object."""
        return Path(self.precedents_directory)

    @property
    def vector_db_path(self) -> Path:
        """Get vector database directory as Path object."""
        return Path(self.chroma_persist_directory)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Convenience function for accessing settings
settings = get_settings()
