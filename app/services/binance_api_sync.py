"""Binance Pay trade history -> PostgreSQL, per client, and the ``PaymentEvidenceProvider``.

* Each client's incoming transfers (positive amount, allowed ``orderType``, default
  ``C2C``) are stored in ``payments`` with ``tenant_id``, ``source=BINANCE_PAY_HISTORY`` and
  ``payment_code=transactionId``, using that client's own (encrypted) read-only API key.
* Incremental: a per-client cursor in ``evidence_sync_state`` plus an overlap window;
  duplicates are impossible thanks to ``UNIQUE(tenant_id, source, external_id)``.
* Multi-replica safe and weight-friendly (3000 per call, per Binance account): a per-client
  advisory lock serializes syncs and ``BINANCE_API_MIN_INTERVAL_SECONDS`` is enforced
  through PostgreSQL, so N replicas never multiply the calls to Binance.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.config import Settings
from app.core.exceptions import (
    AppError,
    EvidenceNotConfiguredError,
    EvidenceProviderUnavailableError,
    EvidenceSyncPendingError,
)
from app.db.locks import BINANCE_API_SYNC_LOCK_NAMESPACE, advisory_lock
from app.db.models import BinanceCredential, EvidenceSyncState, Payment, PaymentSource, Tenant
from app.db.repositories.payments import NewPayment, PaymentRepository, StoreOutcome
from app.integrations.binance.account_api import BinanceApiError, filter_incoming
from app.integrations.binance.evidence import PaymentEvidence, PaymentStatus
from app.services.binance_credentials import BinanceCredentialService

logger = logging.getLogger(__name__)

SOURCE = PaymentSource.BINANCE_PAY_HISTORY
_MAX_WINDOW = timedelta(days=89)


class SyncMode(StrEnum):
    PERIODIC = "periodic"
    ON_DEMAND = "on_demand"
    MANUAL = "manual"


@dataclass
class ApiSyncSummary:
    fetched: int = 0
    incoming: int = 0
    payments_imported: int = 0
    duplicates: int = 0
    synced: bool = False
    failed: bool = False
    not_configured: bool = False
    busy: bool = False
    skipped_recent: bool = False
    errors: list[str] = field(default_factory=list)


def utcnow() -> datetime:
    return datetime.now(UTC)


def payment_to_evidence(payment: Payment) -> PaymentEvidence:
    return PaymentEvidence(
        evidence_id=payment.id,
        external_id=payment.external_id,
        payment_code=payment.payment_code,
        amount=payment.amount,
        asset=payment.asset,
        status=PaymentStatus(payment.payment_status),
        timestamp=payment.received_at,
        source=PaymentSource(payment.source),
        trusted=payment.trusted,
        ambiguous=payment.ambiguous,
    )


class BinanceApiSyncService:
    def __init__(
        self,
        *,
        settings: Settings,
        engine: AsyncEngine,
        sessionmaker: async_sessionmaker[AsyncSession],
        credentials: BinanceCredentialService,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._settings = settings
        self._engine = engine
        self._sessionmaker = sessionmaker
        self._credentials = credentials
        self._clock = clock

    def _normalize_code(self, code: str) -> str:
        code = code.strip()
        return code.upper() if self._settings.payment_code_case_insensitive else code

    async def configured_tenant_ids(self) -> list[int]:
        async with self._sessionmaker() as session:
            result = await session.scalars(
                select(Tenant.id)
                .join(BinanceCredential, BinanceCredential.tenant_id == Tenant.id)
                .where(Tenant.enabled)
                .order_by(Tenant.id)
            )
            return list(result)

    async def _state(self, session: AsyncSession, tenant_id: int) -> EvidenceSyncState | None:
        return await session.scalar(
            select(EvidenceSyncState).where(
                EvidenceSyncState.tenant_id == tenant_id, EvidenceSyncState.source == SOURCE
            )
        )

    async def _save_state(
        self, tenant_id: int, *, cursor: datetime | None, synced_at: datetime, error: str | None
    ) -> None:
        values: dict[str, object] = {"last_synced_at": synced_at, "last_error": error}
        if cursor is not None:
            values["cursor_time"] = cursor
        async with self._sessionmaker() as session, session.begin():
            stmt = insert(EvidenceSyncState).values(tenant_id=tenant_id, source=SOURCE, **values)
            await session.execute(
                stmt.on_conflict_do_update(
                    index_elements=[EvidenceSyncState.tenant_id, EvidenceSyncState.source],
                    set_=values,
                )
            )

    async def sync_all(self, mode: SyncMode) -> dict[int, ApiSyncSummary]:
        return {t: await self.sync(t, mode) for t in await self.configured_tenant_ids()}

    async def sync(self, tenant_id: int, mode: SyncMode) -> ApiSyncSummary:
        summary = ApiSyncSummary()
        wait = 0.0 if mode is SyncMode.PERIODIC else self._settings.sync_lock_wait_seconds
        async with advisory_lock(
            self._engine,
            BINANCE_API_SYNC_LOCK_NAMESPACE,
            tenant_id,
            wait_seconds=wait,
            transaction_pooler=self._settings.database.transaction_pooler,
        ) as acquired:
            if not acquired:
                summary.busy = True
                return summary
            try:
                client = await self._credentials.client_for(tenant_id)
            except AppError as exc:
                summary.failed = True
                summary.errors.append(exc.public_message)
                logger.error("binance_credentials_unreadable", extra={"tenant_id": tenant_id})
                return summary
            if client is None:
                summary.not_configured = True
                return summary

            async with self._sessionmaker() as session:
                state = await self._state(session, tenant_id)
            now = self._clock()
            if (
                state is not None
                and state.last_synced_at is not None
                and (now - state.last_synced_at).total_seconds()
                < self._settings.binance_api_min_interval_seconds
            ):
                # Synced moments ago (maybe by the replica we waited for): protects the
                # API weight limit no matter how many verify requests arrive.
                summary.skipped_recent = True
                return summary

            if state is not None and state.cursor_time is not None:
                start = state.cursor_time - timedelta(
                    seconds=self._settings.binance_api_overlap_seconds
                )
            else:
                start = now - timedelta(hours=self._settings.binance_api_initial_lookback_hours)
            start = max(start, now - _MAX_WINDOW)

            try:
                transactions = await client.fetch_transactions(start, now)
            except BinanceApiError as exc:
                summary.failed = True
                summary.errors.append(exc.public_message)
                logger.warning(
                    "binance_api_sync_failed",
                    extra={
                        "tenant_id": tenant_id,
                        "error_type": type(exc).__name__,
                        "binance_code": exc.code,
                    },
                )
                await self._save_state(
                    tenant_id, cursor=None, synced_at=now, error=exc.public_message
                )
                return summary

            summary.fetched = len(transactions)
            incoming = filter_incoming(transactions, self._settings.binance_pay_order_types)
            summary.incoming = len(incoming)
            async with self._sessionmaker() as session, session.begin():
                repo = PaymentRepository(session)
                for tx in incoming:
                    outcome = await repo.store(
                        NewPayment(
                            tenant_id=tenant_id,
                            source=SOURCE,
                            external_id=tx.transaction_id,
                            payment_code=self._normalize_code(tx.payment_code),
                            amount=tx.amount,
                            asset=tx.currency,
                            # Pay trade history only lists settled transfers.
                            payment_status=PaymentStatus.PAID,
                            received_at=tx.transaction_time,
                            trusted=True,  # signed API response from Binance itself
                            payer_name=tx.payer_name,
                            payer_binance_id=tx.payer_binance_id,
                        )
                    )
                    if outcome is StoreOutcome.INSERTED:
                        summary.payments_imported += 1
                    else:
                        summary.duplicates += 1
            await self._save_state(tenant_id, cursor=now, synced_at=now, error=None)
            summary.synced = True
            logger.info(
                "binance_api_sync_done",
                extra={
                    "tenant_id": tenant_id,
                    "mode": mode.value,
                    "fetched": summary.fetched,
                    "incoming": summary.incoming,
                    "imported": summary.payments_imported,
                },
            )
            return summary


class BinancePayHistoryProvider:
    """``PaymentEvidenceProvider`` backed by each client's Binance Pay trade history."""

    source = SOURCE

    def __init__(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        sync_service: BinanceApiSyncService,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._sync = sync_service

    async def _lookup(self, tenant_id: int, payment_code: str) -> PaymentEvidence | None:
        async with self._sessionmaker() as session:
            payment = await PaymentRepository(session).find_best_by_code(
                tenant_id, payment_code, source=SOURCE
            )
            return payment_to_evidence(payment) if payment is not None else None

    async def find_payment(self, tenant_id: int, payment_code: str) -> PaymentEvidence | None:
        evidence = await self._lookup(tenant_id, payment_code)
        if evidence is not None:
            return evidence
        # Not stored yet (e.g. paid seconds ago): query Binance now and look again.
        summary = await self._sync.sync(tenant_id, SyncMode.ON_DEMAND)
        if summary.not_configured:
            raise EvidenceNotConfiguredError("No Binance API key for this client")
        refreshed = await self._lookup(tenant_id, payment_code)
        if refreshed is not None:
            return refreshed
        if summary.failed:
            raise EvidenceProviderUnavailableError("Binance API sync failed")
        if summary.busy:
            raise EvidenceSyncPendingError("Binance API sync in progress")
        return None
