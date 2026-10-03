"""Binance account API client: Binance Pay trade history (``GET /sapi/v1/pay/transactions``).

This uses a normal Binance account API key (no merchant account), created with ONLY the
"Enable Reading" permission. Each Binance Pay transfer is returned with its
``transactionId`` (e.g. ``P_A99TESTPAYX71116``), which is what the payer can quote and
what the Binance notification email does NOT contain.

* Requests are signed with HMAC-SHA256 over the query string (``X-MBX-APIKEY`` header).
* Amounts are parsed as :class:`~decimal.Decimal` (never float).
* Limited retries for network errors / 5xx / rate limits; clock drift (-1021) is fixed
  once by syncing with the server time.
* The endpoint weighs 3000 (UID): callers must space syncs out (see the sync service).
* Never logs or raises with the API key, secret, signature or response bodies.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlencode

import httpx
from pydantic import SecretStr

from app.core.exceptions import AppError

logger = logging.getLogger(__name__)

PAY_TRANSACTIONS_PATH = "/sapi/v1/pay/transactions"
API_RESTRICTIONS_PATH = "/sapi/v1/account/apiRestrictions"
SERVER_TIME_PATH = "/api/v3/time"
PAGE_LIMIT = 100  # API maximum
MAX_WINDOW_MS = 90 * 24 * 3600 * 1000  # API maximum interval between startTime and endTime
_TIMESTAMP_ERROR = -1021


class BinanceApiError(AppError):
    """Binance API unreachable, rate limited or rejected the request."""

    public_message = "Binance API unavailable"

    def __init__(self, message: str, *, code: int | None = None, retryable: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class BinanceAuthError(BinanceApiError):
    public_message = "Binance API key rejected (check key, permissions and IP whitelist)"


@dataclass(frozen=True)
class KeyPermissions:
    """Result of ``GET /sapi/v1/account/apiRestrictions`` (weight 1)."""

    enable_reading: bool
    ip_restricted: bool
    # Every permission that could move or trade funds. Must ALL be false.
    dangerous: dict[str, bool]

    @property
    def read_only(self) -> bool:
        return self.enable_reading and not any(self.dangerous.values())

    @property
    def problems(self) -> list[str]:
        issues = [] if self.enable_reading else ["enableReading must be ON"]
        issues += [f"{name} must be OFF" for name, on in self.dangerous.items() if on]
        return issues

    def as_dict(self) -> dict[str, bool]:
        return {
            "enableReading": self.enable_reading,
            "ipRestrict": self.ip_restricted,
            **self.dangerous,
        }


# Permissions that allow trading, withdrawing or moving funds. A key with any of them is
# rejected: we only ever need to READ the Pay history.
DANGEROUS_PERMISSIONS = (
    "enableWithdrawals",
    "enableInternalTransfer",
    "permitsUniversalTransfer",
    "enableSpotAndMarginTrading",
    "enableMargin",
    "enableFutures",
    "enableVanillaOptions",
    "enablePortfolioMarginTrading",
)


def parse_key_permissions(body: dict[str, Any]) -> KeyPermissions:
    return KeyPermissions(
        enable_reading=bool(body.get("enableReading", False)),
        ip_restricted=bool(body.get("ipRestrict", False)),
        # Unknown/missing flags are treated as OFF; present ones are taken literally.
        dangerous={name: bool(body.get(name, False)) for name in DANGEROUS_PERMISSIONS},
    )


@dataclass(frozen=True)
class PayTransaction:
    order_type: str
    transaction_id: str
    transaction_time: datetime
    amount: Decimal  # positive = income, negative = expenditure
    currency: str
    payer_name: str | None
    payer_binance_id: str | None
    order_id: str | None = None  # "Order ID" shown to the payer in the Binance app

    @property
    def is_incoming(self) -> bool:
        return self.amount > 0

    @property
    def payment_code(self) -> str:
        """What the payer quotes: the Order ID, falling back to the transactionId."""
        return self.order_id or self.transaction_id


def parse_transaction(item: dict[str, Any]) -> PayTransaction:
    """Strictly parse one item; raises ValueError on anything unexpected."""
    try:
        transaction_id = str(item["transactionId"]).strip()
        amount = Decimal(str(item["amount"]))
        millis = int(item["transactionTime"])
        currency = str(item["currency"]).strip().upper()
        order_type = str(item["orderType"]).strip().upper()
    except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
        raise ValueError("Malformed Pay transaction") from exc
    if not transaction_id or not currency or not amount.is_finite():
        raise ValueError("Malformed Pay transaction")
    payer = item.get("payerInfo") or {}
    name = payer.get("name") if isinstance(payer, dict) else None
    binance_id = payer.get("binanceId") if isinstance(payer, dict) else None
    order_id = str(item.get("orderId") or "").strip() or None
    return PayTransaction(
        order_type=order_type,
        transaction_id=transaction_id,
        transaction_time=datetime.fromtimestamp(millis / 1000, tz=UTC),
        amount=amount,
        currency=currency,
        payer_name=str(name)[:128] if name else None,
        payer_binance_id=str(binance_id)[:64] if binance_id else None,
        order_id=order_id[:64] if order_id else None,
    )


class BinancePayHistoryClient:
    def __init__(
        self,
        *,
        api_key: SecretStr,
        api_secret: SecretStr,
        base_url: str = "https://api.binance.com",
        timeout_seconds: float = 10.0,
        recv_window_ms: int = 10_000,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._key = api_key
        self._secret = api_secret
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._recv_window = recv_window_ms
        self._max_retries = max_retries
        self._backoff = retry_backoff_seconds
        self._transport = transport
        self._clock = clock
        self._time_offset_ms = 0  # server time - local time, learned on -1021

    def sign(self, query: str) -> str:
        return hmac.new(
            self._secret.get_secret_value().encode(), query.encode(), hashlib.sha256
        ).hexdigest()

    def _http(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout, transport=self._transport
        )

    def _now_ms(self) -> int:
        return int(self._clock() * 1000) + self._time_offset_ms

    async def _sync_server_time(self) -> None:
        try:
            async with self._http() as client:
                response = await client.get(SERVER_TIME_PATH)
            server_ms = int(response.json()["serverTime"])
        except httpx.HTTPError, KeyError, ValueError, TypeError:
            return
        self._time_offset_ms = server_ms - int(self._clock() * 1000)
        logger.warning("binance_clock_offset_adjusted", extra={"offset_ms": self._time_offset_ms})

    async def _signed_get(self, path: str, params: dict[str, Any]) -> Any:
        attempt = 0
        resynced = False
        while True:
            query = urlencode(
                {**params, "recvWindow": self._recv_window, "timestamp": self._now_ms()}
            )
            url = f"{path}?{query}&signature={self.sign(query)}"
            try:
                async with self._http() as client:
                    response = await client.get(
                        url, headers={"X-MBX-APIKEY": self._key.get_secret_value()}
                    )
                return self._handle(response)
            except BinanceApiError as exc:
                if exc.code == _TIMESTAMP_ERROR and not resynced:
                    resynced = True
                    await self._sync_server_time()
                    continue
                if not exc.retryable or attempt >= self._max_retries:
                    raise
                error: Exception = exc
            except httpx.HTTPError as exc:
                if attempt >= self._max_retries:
                    raise BinanceApiError(
                        f"Binance API unreachable ({type(exc).__name__})"
                    ) from None
                error = exc
            attempt += 1
            delay = self._backoff * (2 ** (attempt - 1))
            logger.warning(
                "binance_api_retry",
                extra={"attempt": attempt, "delay_s": delay, "error_type": type(error).__name__},
            )
            await asyncio.sleep(delay)

    @staticmethod
    def _handle(response: httpx.Response) -> Any:
        try:
            body = json.loads(response.text, parse_float=Decimal)
        except ValueError:
            body = None
        code = body.get("code") if isinstance(body, dict) else None
        try:
            numeric_code = int(code) if code is not None else None
        except TypeError, ValueError:
            numeric_code = None
        status = response.status_code
        if status == 200:
            if isinstance(body, dict) and body.get("success") is False:
                raise BinanceApiError("Binance Pay history returned success=false", retryable=True)
            return body
        if status in (418, 429):
            raise BinanceApiError(f"Binance API rate limit (HTTP {status})", code=numeric_code)
        if status >= 500:
            raise BinanceApiError(f"Binance API error (HTTP {status})", code=numeric_code)
        if numeric_code == _TIMESTAMP_ERROR:
            raise BinanceApiError("Binance API timestamp out of recvWindow", code=numeric_code)
        if status in (401, 403) or numeric_code in (-2014, -2015, -1022):
            raise BinanceAuthError(
                f"Binance API key rejected (HTTP {status}, code {numeric_code})",
                code=numeric_code,
                retryable=False,
            )
        raise BinanceApiError(
            f"Binance API request rejected (HTTP {status}, code {numeric_code})",
            code=numeric_code,
            retryable=False,
        )

    async def key_permissions(self) -> KeyPermissions:
        body = await self._signed_get(API_RESTRICTIONS_PATH, {})
        if not isinstance(body, dict):
            raise BinanceApiError("Unexpected apiRestrictions response", retryable=False)
        return parse_key_permissions(body)

    async def fetch_page(self, start: datetime, end: datetime) -> list[PayTransaction]:
        body = await self._signed_get(
            PAY_TRANSACTIONS_PATH,
            {
                "startTime": int(start.timestamp() * 1000),
                "endTime": int(end.timestamp() * 1000),
                "limit": PAGE_LIMIT,
            },
        )
        items = body.get("data") if isinstance(body, dict) else None
        if not isinstance(items, list):
            raise BinanceApiError("Unexpected Binance Pay history response", retryable=False)
        parsed: list[PayTransaction] = []
        for item in items:
            try:
                parsed.append(parse_transaction(item))
            except ValueError:
                logger.error("binance_pay_transaction_malformed")
        return parsed

    async def fetch_transactions(
        self, start: datetime, end: datetime, *, max_pages: int = 20
    ) -> list[PayTransaction]:
        """All transactions in [start, end], walking backwards when a page is full."""
        if (end - start).total_seconds() * 1000 > MAX_WINDOW_MS:
            raise ValueError("Window larger than 90 days")
        seen: dict[str, PayTransaction] = {}
        page_end = end
        for _ in range(max_pages):
            page = await self.fetch_page(start, page_end)
            for tx in page:
                seen.setdefault(tx.transaction_id, tx)
            if len(page) < PAGE_LIMIT:
                return sorted(seen.values(), key=lambda t: t.transaction_time)
            oldest = min(tx.transaction_time for tx in page)
            if oldest >= page_end or oldest <= start:
                break
            page_end = oldest  # inclusive: duplicates are removed by transaction id
        else:
            raise BinanceApiError("Too many Pay history pages in one sync", retryable=False)
        return sorted(seen.values(), key=lambda t: t.transaction_time)


def filter_incoming(
    transactions: Sequence[PayTransaction], order_types: Sequence[str]
) -> list[PayTransaction]:
    allowed = {t.strip().upper() for t in order_types if t.strip()}
    return [t for t in transactions if t.is_incoming and t.order_type in allowed]
