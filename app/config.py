"""Application configuration.

All configuration comes from environment variables (optionally a ``.env`` file in
development). Secrets are typed as :class:`pydantic.SecretStr` so they are never
rendered by ``repr()``/``str()`` and therefore never leak into logs or tracebacks.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

if TYPE_CHECKING:
    from app.db.connection import DatabaseConnection


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


def _split_csv(value: object) -> object:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


CsvList = Annotated[list[str], NoDecode]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # `VAR=` (empty) means "not set" rather than an empty string.
        env_parse_none_str="",
    )

    # --- Application -----------------------------------------------------------------
    app_env: Environment = Environment.DEVELOPMENT
    app_name: str = "binance-pay-verifier"
    log_level: str = "INFO"
    log_json: bool = True
    enable_docs: bool | None = None  # default: enabled outside production

    # --- Database --------------------------------------------------------------------
    database_url: SecretStr = SecretStr(
        "postgresql+psycopg://postgres:postgres@localhost:5432/binance_pay"
    )
    # Max DB connections per process (API or worker). With Supabase's transaction pooler
    # a small number is enough: connections are multiplexed by Supavisor.
    db_pool_max: int = Field(default=5, ge=2, le=100)
    database_pool_timeout_seconds: float = Field(default=10, gt=0)
    database_statement_timeout_ms: int = Field(default=15_000, ge=1_000)
    # Supabase: TLS is required automatically; pooler mode is auto-detected (port 6543 =
    # transaction pooler). Override only if needed.
    database_ssl_mode: str | None = None
    database_pooler_mode: Literal["session", "transaction"] | None = None

    # --- API security ----------------------------------------------------------------
    # Owner master key(s) for /v1/admin/* (manage clients and their access tokens).
    # Clients authenticate with per-client tokens stored (hashed) in the database.
    admin_api_keys: Annotated[list[SecretStr], NoDecode] = Field(default_factory=list)
    token_last_used_update_seconds: int = Field(default=60, ge=0)
    cors_allowed_origins: CsvList = Field(default_factory=list)
    allowed_hosts: CsvList = Field(default_factory=list)
    max_request_body_bytes: int = Field(default=16 * 1024, ge=1024)
    rate_limit_verify_per_minute: int = Field(default=60, ge=1)
    rate_limit_admin_per_minute: int = Field(default=30, ge=1)
    rate_limit_credentials_per_minute: int = Field(default=5, ge=1)
    trust_forwarded_headers: bool = False

    # --- Encryption ------------------------------------------------------------------
    credentials_encryption_key: SecretStr | None = None
    credentials_encryption_previous_keys: Annotated[list[SecretStr], NoDecode] = Field(
        default_factory=list
    )

    # --- Sync worker -------------------------------------------------------------------
    sync_lock_wait_seconds: float = Field(default=10, ge=0, le=60)
    # Run the periodic sync loop inside the API process (otherwise run the worker process).
    run_sync_worker_in_api: bool = False

    # --- Binance account API (Pay trade history) ---------------------------------------
    # Each client registers its own read-only Binance API key through
    # PUT /v1/me/binance-credentials; keys are stored encrypted, never in the environment.
    binance_api_base_url: str = "https://api.binance.com"
    binance_api_timeout_seconds: float = Field(default=10, gt=0, le=60)
    binance_api_recv_window_ms: int = Field(default=10_000, ge=1_000, le=60_000)
    binance_api_max_retries: int = Field(default=2, ge=0, le=5)
    # The endpoint weighs 3000 (UID): keep syncs spaced out across ALL replicas.
    binance_api_sync_interval_seconds: int = Field(default=15, ge=5)
    binance_api_min_interval_seconds: float = Field(default=5, ge=1)
    binance_api_initial_lookback_hours: int = Field(default=48, ge=1, le=24 * 89)
    binance_api_overlap_seconds: int = Field(default=300, ge=0, le=3600)
    binance_pay_order_types: CsvList = Field(default_factory=lambda: ["C2C"])

    # --- Payment verification --------------------------------------------------------
    default_asset: str = "USDT"
    default_payment_max_age_minutes: int = Field(default=60, ge=1)
    max_payment_age_minutes: int = Field(default=1440, ge=1)
    payment_clock_skew_seconds: int = Field(default=300, ge=0, le=3600)
    payment_code_case_insensitive: bool = False
    # Explicit, opt-in absolute tolerance. 0 means exact match (the default).
    payment_amount_tolerance: str = "0"
    # true: a payment GREATER than expected is also accepted (underpayment is still rejected).
    payment_accept_overpayment: bool = False

    # ---------------------------------------------------------------------------------
    @field_validator(
        "cors_allowed_origins",
        "allowed_hosts",
        "binance_pay_order_types",
        mode="before",
    )
    @classmethod
    def _csv(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("admin_api_keys", "credentials_encryption_previous_keys", mode="before")
    @classmethod
    def _csv_secrets(cls, value: object) -> object:
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        return _split_csv(value)

    @field_validator("default_asset")
    @classmethod
    def _upper_asset(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def _validate(self) -> Self:
        for key in self.admin_api_keys:
            if len(key.get_secret_value()) < 32:
                raise ValueError("ADMIN_API_KEYS must be at least 32 characters long")

        if self.max_payment_age_minutes < self.default_payment_max_age_minutes:
            raise ValueError("MAX_PAYMENT_AGE_MINUTES must be >= DEFAULT_PAYMENT_MAX_AGE_MINUTES")

        try:
            tolerance = Decimal(self.payment_amount_tolerance)
        except InvalidOperation as exc:
            raise ValueError("PAYMENT_AMOUNT_TOLERANCE must be a decimal string") from exc
        if not tolerance.is_finite() or tolerance < 0:
            raise ValueError("PAYMENT_AMOUNT_TOLERANCE must be >= 0")
        return self

    # --- Derived values --------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.app_env is Environment.PRODUCTION

    @property
    def docs_enabled(self) -> bool:
        return self.enable_docs if self.enable_docs is not None else not self.is_production

    @property
    def encryption_configured(self) -> bool:
        return self.credentials_encryption_key is not None

    @property
    def database(self) -> DatabaseConnection:
        from app.db.connection import resolve_database  # noqa: PLC0415 - avoid import cycle

        return resolve_database(
            self.database_url.get_secret_value(),
            ssl_mode=self.database_ssl_mode,
            pooler_mode=self.database_pooler_mode,
            statement_timeout_ms=self.database_statement_timeout_ms,
            application_name=self.app_name,
        )

    @property
    def amount_tolerance(self) -> Decimal:
        return Decimal(self.payment_amount_tolerance)

    def secret_values(self) -> list[str]:
        """Every configured secret, used by the log redaction filter."""
        candidates: list[SecretStr | None] = [
            self.database_url,
            self.credentials_encryption_key,
            *self.admin_api_keys,
            *self.credentials_encryption_previous_keys,
        ]
        values = [c.get_secret_value() for c in candidates if c is not None]
        # Also redact the DB password alone (raw and URL-encoded forms), parsed with the
        # same tolerant parser used to connect (passwords may contain "@", "#", "/"...).
        from urllib.parse import quote  # noqa: PLC0415

        from app.db.connection import normalize_url  # noqa: PLC0415 - avoid import cycle

        try:
            password = normalize_url(self.database_url.get_secret_value()).password
        except Exception:
            password = None
        if password:
            values += [password, quote(password, safe="")]
        return [v for v in values if len(v) >= 6]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
