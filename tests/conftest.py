from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.config import Settings
from app.core.encryption import generate_key
from app.integrations.binance.account_api import (
    BinanceAuthError,
    KeyPermissions,
    PayTransaction,
    parse_key_permissions,
)

ADMIN_KEY = "test-admin-key-0123456789-abcdefghijklmn"
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/binance_pay_test"
)
ENCRYPTION_KEY = generate_key()

# Synthetic Binance credentials (same shape as real ones).
READ_ONLY_KEY = "rokey" + "A" * 59
READ_ONLY_SECRET = "rosecret" + "B" * 56
OTHER_KEY = "otherkey" + "C" * 56
OTHER_SECRET = "othersecret" + "D" * 53


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "app_env": "test",
        "database_url": SecretStr(TEST_DATABASE_URL),
        "admin_api_keys": [SecretStr(ADMIN_KEY)],
        "credentials_encryption_key": SecretStr(ENCRYPTION_KEY),
        "sync_lock_wait_seconds": 2,
        "binance_api_min_interval_seconds": 1,
        "log_json": False,
        "rate_limit_verify_per_minute": 1000,
        "rate_limit_admin_per_minute": 1000,
        "rate_limit_credentials_per_minute": 1000,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture
def settings() -> Settings:
    return make_settings()


# --- Fake Binance (several accounts, each with its own API key) ---------------------------


READ_ONLY = {"enableReading": True, "ipRestrict": False}


@dataclass
class FakeAccount:
    api_secret: str
    permissions: dict[str, Any] = field(default_factory=lambda: dict(READ_ONLY))
    transactions: list[PayTransaction] = field(default_factory=list)
    calls: list[tuple[datetime, datetime]] = field(default_factory=list)
    fail: Exception | None = None
    delay: float = 0.0

    def add(
        self,
        transaction_id: str,
        amount: str,
        currency: str = "USDT",
        *,
        when: datetime | None = None,
        order_type: str = "C2C",
        payer: str | None = "User-0000aaaa",
        order_id: str | None = None,
    ) -> None:
        self.transactions.append(
            PayTransaction(
                order_type=order_type,
                transaction_id=transaction_id,
                transaction_time=when or datetime.now(UTC).replace(microsecond=0),
                amount=Decimal(amount),
                currency=currency,
                payer_name=payer,
                payer_binance_id=None,
                order_id=order_id,
            )
        )


class FakeBinanceClient:
    def __init__(self, binance: FakeBinance, api_key: str, api_secret: str) -> None:
        self._binance = binance
        self._key = api_key
        self._secret = api_secret

    def _account(self) -> FakeAccount:
        account = self._binance.accounts.get(self._key)
        if account is None or account.api_secret != self._secret:
            raise BinanceAuthError("Invalid API-key", code=-2015, retryable=False)
        if account.fail is not None:
            raise account.fail
        return account

    async def key_permissions(self) -> KeyPermissions:
        return parse_key_permissions(self._account().permissions)

    async def fetch_page(self, start: datetime, end: datetime) -> list[PayTransaction]:
        account = self._account()
        return [t for t in account.transactions if start <= t.transaction_time <= end]

    async def fetch_transactions(
        self, start: datetime, end: datetime, *, max_pages: int = 20
    ) -> list[PayTransaction]:
        account = self._account()
        account.calls.append((start, end))
        if account.delay:
            await asyncio.sleep(account.delay)
        return [t for t in account.transactions if start <= t.transaction_time <= end]


class FakeBinance:
    def __init__(self) -> None:
        self.accounts: dict[str, FakeAccount] = {
            READ_ONLY_KEY: FakeAccount(api_secret=READ_ONLY_SECRET),
            OTHER_KEY: FakeAccount(api_secret=OTHER_SECRET),
        }
        self.clients_built = 0

    def factory(self, api_key: SecretStr, api_secret: SecretStr) -> FakeBinanceClient:
        self.clients_built += 1
        return FakeBinanceClient(self, api_key.get_secret_value(), api_secret.get_secret_value())

    @property
    def main(self) -> FakeAccount:
        return self.accounts[READ_ONLY_KEY]

    @property
    def other(self) -> FakeAccount:
        return self.accounts[OTHER_KEY]


@pytest.fixture
def binance() -> FakeBinance:
    return FakeBinance()


# --- Database ---------------------------------------------------------------------------


def _run_migrations(url: str) -> None:
    from alembic.config import Config

    from alembic import command

    cfg = Config(str(Path(__file__).parents[1] / "alembic.ini"))
    cfg.set_main_option("script_location", str(Path(__file__).parents[1] / "alembic"))
    cfg.cmd_opts = type("Opts", (), {"x": [f"url={url}"]})()  # type: ignore[assignment]
    command.upgrade(cfg, "head")


@pytest.fixture(scope="session")
async def engine() -> AsyncIterator[AsyncEngine]:
    from app.db.session import create_engine

    settings = make_settings()
    eng = create_engine(settings)
    try:
        async with eng.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - environment dependent
        await eng.dispose()
        pytest.skip(f"PostgreSQL not available for integration tests: {type(exc).__name__}")
    async with eng.begin() as conn:
        # Fresh schema every session (migrations are not reversible past 0004).
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
        # Simulate Supabase's public API roles so the RLS/REVOKE migrations are exercised.
        await conn.execute(
            text(
                "DO $$ BEGIN "
                "IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') "
                "THEN CREATE ROLE anon NOLOGIN; END IF; "
                "IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') "
                "THEN CREATE ROLE authenticated NOLOGIN; END IF; END $$"
            )
        )
        await conn.execute(text("GRANT USAGE ON SCHEMA public TO anon, authenticated"))
    await asyncio.to_thread(_run_migrations, TEST_DATABASE_URL)
    yield eng
    await eng.dispose()


@pytest.fixture
async def sessionmaker(engine: AsyncEngine) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "TRUNCATE payment_claims, payments, evidence_sync_state, binance_credentials, "
                "api_tokens, tenants, rate_limit_counters RESTART IDENTITY CASCADE"
            )
        )
    yield async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest.fixture
def build(
    engine: AsyncEngine,
    sessionmaker: async_sessionmaker[AsyncSession],
    binance: FakeBinance,
) -> Callable[..., Any]:
    """Build a fully wired container against the test DB and the fake Binance."""
    from app.container import build_container

    def _build(**overrides: Any) -> Any:
        return build_container(
            make_settings(**overrides), engine, sessionmaker, client_factory=binance.factory
        )

    return _build


@dataclass
class ClientHandle:
    tenant_id: int
    token: str
    token_id: int

    @property
    def auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


@pytest.fixture
def new_client(build: Callable[..., Any]) -> Callable[..., Any]:
    """Create a client (+ token) and, by default, register its read-only Binance key."""

    async def _new(
        name: str = "olivier",
        *,
        api_key: str | None = READ_ONLY_KEY,
        api_secret: str = READ_ONLY_SECRET,
    ) -> ClientHandle:
        container = build()
        tenant, issued = await container.tenants.create_tenant(name)
        if api_key is not None:
            await container.credentials.save(tenant.id, SecretStr(api_key), SecretStr(api_secret))
        return ClientHandle(tenant.id, issued.token, issued.id)

    return _new
