"""Payment verification use case.

Depends only on :class:`PaymentEvidenceProvider` (Binance Pay trade history today, the
Merchant API later) and on :class:`PaymentClaimService`. Every verification is scoped to
one client (tenant). Any doubt results in a *non-verified* status.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from app.core.exceptions import (
    EvidenceNotConfiguredError,
    EvidenceProviderUnavailableError,
    EvidenceSyncPendingError,
)
from app.integrations.binance.evidence import PaymentEvidenceProvider, PaymentStatus
from app.services.payment_claim import ClaimStatus, PaymentClaimService

logger = logging.getLogger(__name__)

PAYMENT_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{3,63}$")


class VerificationStatus(StrEnum):
    VERIFIED = "VERIFIED"
    NOT_FOUND = "NOT_FOUND"
    PENDING_SYNC = "PENDING_SYNC"
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    ASSET_MISMATCH = "ASSET_MISMATCH"
    UNTRUSTED_EVIDENCE = "UNTRUSTED_EVIDENCE"
    EXPIRED_PAYMENT = "EXPIRED_PAYMENT"
    ALREADY_CLAIMED = "ALREADY_CLAIMED"
    INVALID_PAYMENT_CODE = "INVALID_PAYMENT_CODE"
    BINANCE_API_UNAVAILABLE = "BINANCE_API_UNAVAILABLE"
    BINANCE_NOT_CONFIGURED = "BINANCE_NOT_CONFIGURED"
    # Extensions (documented in README): evidence exists but cannot be accepted.
    PAYMENT_NOT_COMPLETED = "PAYMENT_NOT_COMPLETED"
    AMBIGUOUS_PAYMENT = "AMBIGUOUS_PAYMENT"


RETRYABLE = frozenset(
    {
        VerificationStatus.NOT_FOUND,
        VerificationStatus.PENDING_SYNC,
        VerificationStatus.BINANCE_API_UNAVAILABLE,
        VerificationStatus.PAYMENT_NOT_COMPLETED,
    }
)


@dataclass(frozen=True)
class VerificationRequest:
    tenant_id: int
    payment_code: str
    expected_amount: Decimal
    asset: str
    order_reference: str | None
    max_age_minutes: int
    client_id: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    payment_code: str
    expected_amount: Decimal
    asset: str
    order_reference: str | None
    received_amount: Decimal | None = None
    received_asset: str | None = None
    received_at: datetime | None = None
    idempotent: bool = False
    detail: str | None = None

    @property
    def verified(self) -> bool:
        return self.status is VerificationStatus.VERIFIED

    @property
    def retryable(self) -> bool:
        return self.status in RETRYABLE


def utcnow() -> datetime:
    return datetime.now(UTC)


class PaymentVerifier:
    def __init__(
        self,
        *,
        evidence_provider: PaymentEvidenceProvider,
        claim_service: PaymentClaimService,
        case_insensitive_code: bool = False,
        amount_tolerance: Decimal = Decimal(0),
        accept_overpayment: bool = False,
        clock_skew: timedelta = timedelta(minutes=5),
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._provider = evidence_provider
        self._claims = claim_service
        self._case_insensitive = case_insensitive_code
        self._tolerance = amount_tolerance
        self._accept_overpayment = accept_overpayment
        self._skew = clock_skew
        self._clock = clock

    def normalize_code(self, code: str) -> str:
        code = code.strip()
        return code.upper() if self._case_insensitive else code

    async def verify(self, request: VerificationRequest) -> VerificationResult:
        code = self.normalize_code(request.payment_code)
        asset = request.asset.strip().upper()
        base = VerificationResult(
            status=VerificationStatus.NOT_FOUND,
            payment_code=code,
            expected_amount=request.expected_amount,
            asset=asset,
            order_reference=request.order_reference,
        )

        def result(status: VerificationStatus, **kw: object) -> VerificationResult:
            res = replace(base, status=status, **kw)  # type: ignore[arg-type]
            logger.info(
                "payment_verification",
                extra={
                    "tenant_id": request.tenant_id,
                    "payment_code": code,
                    "status": status.value,
                    "order_reference": request.order_reference,
                    "idempotent": res.idempotent,
                },
            )
            return res

        # 1. Request-level checks.
        if not PAYMENT_CODE_RE.fullmatch(code):
            return result(VerificationStatus.INVALID_PAYMENT_CODE, detail="Invalid code format")

        # 2-4. Find evidence (DB first, then an on-demand Binance query inside the provider).
        try:
            evidence = await self._provider.find_payment(request.tenant_id, code)
        except EvidenceSyncPendingError:
            return result(VerificationStatus.PENDING_SYNC, detail="Binance sync in progress")
        except EvidenceNotConfiguredError:
            return result(
                VerificationStatus.BINANCE_NOT_CONFIGURED,
                detail="Register your read-only Binance API key: PUT /v1/me/binance-credentials",
            )
        except EvidenceProviderUnavailableError:
            return result(
                VerificationStatus.BINANCE_API_UNAVAILABLE, detail="Binance API unavailable"
            )
        if evidence is None:
            return result(VerificationStatus.NOT_FOUND)

        # 6. Exact code match (defense in depth; the provider already queried with "=").
        if evidence.payment_code != code:
            return result(VerificationStatus.NOT_FOUND)

        found = {
            "received_amount": evidence.amount,
            "received_asset": evidence.asset,
            "received_at": evidence.timestamp,
        }
        # 5. Authenticity of the evidence.
        if not evidence.trusted:
            return result(VerificationStatus.UNTRUSTED_EVIDENCE, detail="Evidence is not trusted")
        if evidence.ambiguous:
            return result(
                VerificationStatus.AMBIGUOUS_PAYMENT,
                detail="Contradicting evidence for this payment code",
            )
        # 8. Asset (checked before amount: amounts of different assets are not comparable).
        if evidence.asset != asset:
            return result(VerificationStatus.ASSET_MISMATCH, **found)
        # 7. Amount: exact Decimal comparison unless an explicit tolerance is configured.
        #    With accept_overpayment, any amount >= expected (minus tolerance) is accepted.
        difference = evidence.amount - request.expected_amount
        if self._accept_overpayment:
            amount_ok = difference >= -self._tolerance
        else:
            amount_ok = abs(difference) <= self._tolerance
        if not amount_ok:
            return result(VerificationStatus.AMOUNT_MISMATCH, **found)
        if evidence.status is not PaymentStatus.PAID:
            return result(
                VerificationStatus.PAYMENT_NOT_COMPLETED,
                detail=f"Payment status is {evidence.status.value}",
                **found,
            )

        # Idempotency: the same order asking again gets its confirmed result back, even if
        # the payment is now older than max_age.
        existing = await self._claims.get_claim(evidence.evidence_id)
        if existing is not None and request.order_reference is not None:
            if existing.order_reference == request.order_reference:
                return result(VerificationStatus.VERIFIED, idempotent=True, **found)
            return result(VerificationStatus.ALREADY_CLAIMED, **found)

        # 9. Time window (UTC).
        now = self._clock()
        if evidence.timestamp > now + self._skew:
            return result(
                VerificationStatus.EXPIRED_PAYMENT, detail="Timestamp is in the future", **found
            )
        if now - evidence.timestamp > timedelta(minutes=request.max_age_minutes):
            return result(VerificationStatus.EXPIRED_PAYMENT, **found)

        # 10-11. Atomic claim.
        claim = await self._claims.claim(
            payment_id=evidence.evidence_id,
            order_reference=request.order_reference,
            expected_amount=request.expected_amount,
            asset=asset,
            client_id=request.client_id,
        )
        match claim.status:
            case ClaimStatus.CLAIMED:
                return result(VerificationStatus.VERIFIED, **found)
            case ClaimStatus.IDEMPOTENT:
                return result(VerificationStatus.VERIFIED, idempotent=True, **found)
            case ClaimStatus.ALREADY_CLAIMED:
                return result(VerificationStatus.ALREADY_CLAIMED, **found)
            case _:
                return result(VerificationStatus.NOT_FOUND)
