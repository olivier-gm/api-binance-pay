"""Composition root: wires settings, DB, Binance integration and services together.

One container per process. It holds no business state: everything that matters lives in
PostgreSQL, so any number of replicas can run side by side.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.config import Settings
from app.core.encryption import CredentialCipher, build_cipher
from app.core.exceptions import ConfigurationError
from app.integrations.binance.account_api import BinancePayHistoryClient
from app.integrations.binance.evidence import PaymentEvidenceProvider
from app.services.binance_api_sync import BinanceApiSyncService, BinancePayHistoryProvider
from app.services.binance_credentials import (
    BinanceAccountClient,
    BinanceCredentialService,
    ClientFactory,
)
from app.services.payment_claim import PaymentClaimService
from app.services.payment_verifier import PaymentVerifier
from app.services.rate_limiter import RateLimiter
from app.services.tenants import TenantService

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]
    cipher: CredentialCipher | None
    tenants: TenantService
    credentials: BinanceCredentialService
    api_sync_service: BinanceApiSyncService
    evidence_provider: PaymentEvidenceProvider
    claim_service: PaymentClaimService
    verifier: PaymentVerifier
    rate_limiter: RateLimiter


def default_client_factory(settings: Settings) -> ClientFactory:
    def factory(api_key: SecretStr, api_secret: SecretStr) -> BinanceAccountClient:
        return BinancePayHistoryClient(
            api_key=api_key,
            api_secret=api_secret,
            base_url=settings.binance_api_base_url,
            timeout_seconds=settings.binance_api_timeout_seconds,
            recv_window_ms=settings.binance_api_recv_window_ms,
            max_retries=settings.binance_api_max_retries,
        )

    return factory


def build_container(
    settings: Settings,
    engine: AsyncEngine,
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    client_factory: ClientFactory | None = None,
) -> Container:
    cipher: CredentialCipher | None = None
    if settings.credentials_encryption_key is not None:
        cipher = build_cipher(
            settings.credentials_encryption_key.get_secret_value(),
            [k.get_secret_value() for k in settings.credentials_encryption_previous_keys],
        )
    elif settings.is_production:
        raise ConfigurationError("CREDENTIALS_ENCRYPTION_KEY is required in production")
    else:
        logger.warning("credentials_encryption_key_missing_binance_credentials_disabled")

    credentials = BinanceCredentialService(
        sessionmaker=sessionmaker,
        cipher=cipher,
        client_factory=client_factory or default_client_factory(settings),
    )
    api_sync_service = BinanceApiSyncService(
        settings=settings, engine=engine, sessionmaker=sessionmaker, credentials=credentials
    )
    evidence_provider = BinancePayHistoryProvider(
        sessionmaker=sessionmaker, sync_service=api_sync_service
    )
    claim_service = PaymentClaimService(sessionmaker)
    verifier = PaymentVerifier(
        evidence_provider=evidence_provider,
        claim_service=claim_service,
        case_insensitive_code=settings.payment_code_case_insensitive,
        amount_tolerance=settings.amount_tolerance,
        accept_overpayment=settings.payment_accept_overpayment,
        clock_skew=timedelta(seconds=settings.payment_clock_skew_seconds),
    )
    return Container(
        settings=settings,
        engine=engine,
        sessionmaker=sessionmaker,
        cipher=cipher,
        tenants=TenantService(
            sessionmaker, last_used_update_seconds=settings.token_last_used_update_seconds
        ),
        credentials=credentials,
        api_sync_service=api_sync_service,
        evidence_provider=evidence_provider,
        claim_service=claim_service,
        verifier=verifier,
        rate_limiter=RateLimiter(sessionmaker),
    )
