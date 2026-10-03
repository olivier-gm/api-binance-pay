"""BinancePayHistoryClient against a mocked HTTP transport (no network)."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import httpx
import pytest
from pydantic import SecretStr

from app.integrations.binance.account_api import (
    PAGE_LIMIT,
    BinanceApiError,
    BinanceAuthError,
    BinancePayHistoryClient,
    filter_incoming,
    parse_transaction,
)

KEY = "test-binance-api-key-0123456789"
SECRET = "test-binance-api-secret-abcdefghij"
NOW = datetime(2026, 9, 19, 2, 0, tzinfo=UTC)


def tx(
    transaction_id: str = "P_A99TESTPAYX71116",
    amount: str = "98.814",
    *,
    millis: int = 1789780602000,
    currency: str = "USDT",
    order_type: str = "C2C",
    payer: str | None = "User-0000aaaa",
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "orderType": order_type,
        "transactionId": transaction_id,
        "transactionTime": millis,
        "amount": amount,
        "currency": currency,
        "walletType": 1,
        "walletTypes": [1],
        "fundsDetail": [{"currency": currency, "amount": amount.lstrip("-")}],
        "payerInfo": {"name": payer, "type": "USER", "binanceId": "123", "accountId": "9"}
        if payer
        else {},
        "receiverInfo": {},
    }
    return item


def ok(items: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(
        200, json={"code": "000000", "message": "success", "data": items, "success": True}
    )


def make_client(
    handler: Callable[[httpx.Request], httpx.Response], **kw: Any
) -> BinancePayHistoryClient:
    return BinancePayHistoryClient(
        api_key=SecretStr(KEY),
        api_secret=SecretStr(SECRET),
        transport=httpx.MockTransport(handler),
        retry_backoff_seconds=0,
        **kw,
    )


async def test_request_is_signed_and_parsed_with_decimal() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok([tx(), tx("P_A99TESTPAYY71115", "99.79136539", payer="User-1111bbbb")])

    page = await make_client(handler).fetch_page(NOW - timedelta(hours=1), NOW)
    request = seen[0]
    assert request.url.path == "/sapi/v1/pay/transactions"
    assert request.headers["X-MBX-APIKEY"] == KEY
    query = urlsplit(str(request.url)).query
    unsigned, _, signature = query.rpartition("&signature=")
    assert signature == hmac.new(SECRET.encode(), unsigned.encode(), hashlib.sha256).hexdigest()
    params = dict(parse_qsl(unsigned))
    assert params["limit"] == "100" and params["recvWindow"] == "10000"
    assert int(params["endTime"]) == int(NOW.timestamp() * 1000)

    first, second = page
    assert first.transaction_id == "P_A99TESTPAYX71116"
    assert first.amount == Decimal("98.814") and isinstance(first.amount, Decimal)
    assert first.transaction_time == datetime(2026, 9, 19, 1, 16, 42, tzinfo=UTC)
    assert first.payer_name == "User-0000aaaa"
    assert second.amount == Decimal("99.79136539")  # 8 decimals, no float rounding


def test_parse_rejects_malformed_items() -> None:
    with pytest.raises(ValueError):
        parse_transaction({"transactionId": "X", "amount": "abc"})
    with pytest.raises(ValueError):
        parse_transaction({**tx(), "amount": "NaN"})


def test_filter_incoming_ignores_outgoing_and_other_types() -> None:
    items = [
        parse_transaction(tx("P_IN1", "98.814")),
        parse_transaction(tx("P_OUT1", "-100", payer=None)),
        parse_transaction(tx("P_BOX1", "5", order_type="CRYPTO_BOX")),
        parse_transaction(tx("P_ZERO", "0")),
    ]
    assert [t.transaction_id for t in filter_incoming(items, ["C2C"])] == ["P_IN1"]
    assert {t.transaction_id for t in filter_incoming(items, ["C2C", "CRYPTO_BOX"])} == {
        "P_IN1",
        "P_BOX1",
    }


async def test_invalid_key_is_not_retried_and_hides_secrets() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            401, json={"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."}
        )

    with pytest.raises(BinanceAuthError) as info:
        await make_client(handler).fetch_page(NOW - timedelta(hours=1), NOW)
    assert calls == 1
    for secret in (KEY, SECRET):
        assert secret not in str(info.value)


@pytest.mark.parametrize("status", [500, 503, 429, 418])
async def test_server_errors_and_rate_limits_are_retried_limited(status: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"code": -1003, "msg": "busy"})

    with pytest.raises(BinanceApiError):
        await make_client(handler, max_retries=2).fetch_page(NOW - timedelta(hours=1), NOW)
    assert calls == 3


async def test_transient_error_then_success() -> None:
    responses = iter([httpx.Response(502), ok([tx()])])
    page = await make_client(lambda r: next(responses)).fetch_page(NOW - timedelta(hours=1), NOW)
    assert len(page) == 1


async def test_timeout_maps_to_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    with pytest.raises(BinanceApiError, match="unreachable"):
        await make_client(handler, max_retries=1).fetch_page(NOW - timedelta(hours=1), NOW)


async def test_clock_drift_is_corrected_with_server_time() -> None:
    local = 1_000_000.0  # local clock far behind the server
    server_ms = int((local + 3600) * 1000)
    timestamps: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/time":
            return httpx.Response(200, json={"serverTime": server_ms})
        ts = int(dict(parse_qsl(request.url.query.decode()))["timestamp"])
        timestamps.append(ts)
        if abs(ts - server_ms) > 10_000:
            return httpx.Response(400, json={"code": -1021, "msg": "Timestamp outside recvWindow"})
        return ok([tx()])

    client = make_client(handler, clock=lambda: local)
    assert len(await client.fetch_page(NOW - timedelta(hours=1), NOW)) == 1
    assert len(timestamps) == 2 and timestamps[1] == server_ms


async def test_success_false_and_unexpected_payloads() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": "000001", "success": False, "data": None})

    with pytest.raises(BinanceApiError):
        await make_client(handler, max_retries=0).fetch_page(NOW - timedelta(hours=1), NOW)

    def bad(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    with pytest.raises(BinanceApiError):
        await make_client(bad, max_retries=0).fetch_page(NOW - timedelta(hours=1), NOW)


async def test_malformed_items_are_skipped_not_fatal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return ok([tx(), {"transactionId": "BROKEN"}])

    page = await make_client(handler).fetch_page(NOW - timedelta(hours=1), NOW)
    assert [t.transaction_id for t in page] == ["P_A99TESTPAYX71116"]


async def test_full_pages_walk_backwards_and_deduplicate() -> None:
    base = int(NOW.timestamp() * 1000)
    first = [tx(f"P_NEW{i:04d}", "1", millis=base - i * 1000) for i in range(PAGE_LIMIT)]
    oldest = base - (PAGE_LIMIT - 1) * 1000
    second = [
        first[-1],
        *(tx(f"P_OLD{i:04d}", "1", millis=oldest - (i + 1) * 1000) for i in range(5)),
    ]
    end_times: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(parse_qsl(request.url.query.decode()))
        end_times.append(int(params["endTime"]))
        return ok(first if len(end_times) == 1 else second)

    result = await make_client(handler).fetch_transactions(NOW - timedelta(days=1), NOW)
    assert len(result) == PAGE_LIMIT + 5
    assert end_times[1] == oldest
    assert result == sorted(result, key=lambda t: t.transaction_time)


async def test_window_larger_than_90_days_rejected() -> None:
    client = make_client(lambda r: ok([]))
    with pytest.raises(ValueError):
        await client.fetch_transactions(NOW - timedelta(days=91), NOW)


def test_json_numbers_never_become_floats() -> None:
    body = json.loads('{"amount": 98.814}', parse_float=Decimal)
    assert isinstance(body["amount"], Decimal)


def test_key_permissions_read_only_detection() -> None:
    from app.integrations.binance.account_api import parse_key_permissions

    read_only = parse_key_permissions({"enableReading": True, "ipRestrict": True})
    assert read_only.read_only and read_only.ip_restricted and read_only.problems == []

    trading = parse_key_permissions(
        {"enableReading": True, "enableSpotAndMarginTrading": True, "enableWithdrawals": True}
    )
    assert not trading.read_only
    assert trading.problems == [
        "enableWithdrawals must be OFF",
        "enableSpotAndMarginTrading must be OFF",
    ]
    assert not parse_key_permissions({"enableReading": False}).read_only


async def test_key_permissions_endpoint_is_signed() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"enableReading": True, "ipRestrict": False})

    perms = await make_client(handler).key_permissions()
    assert perms.read_only
    assert seen[0].url.path == "/sapi/v1/account/apiRestrictions"
    assert "signature=" in str(seen[0].url) and seen[0].headers["X-MBX-APIKEY"] == KEY

 d e f   t e s t _ o r d e r _ i d _ i s _ t h e _ p a y m e n t _ c o d e ( )   - >   N o n e : 
         i t e m   =   t x ( )   |   { " o r d e r I d " :   " 4 5 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 " } 
         p a r s e d   =   p a r s e _ t r a n s a c t i o n ( i t e m ) 
         a s s e r t   p a r s e d . o r d e r _ i d   = =   " 4 5 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 " 
         a s s e r t   p a r s e d . p a y m e n t _ c o d e   = =   " 4 5 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 " 
         a s s e r t   p a r s e d . t r a n s a c t i o n _ i d   = =   " P _ A 9 9 T E S T P A Y X 7 1 1 1 6 " 
 
 
 d e f   t e s t _ p a y m e n t _ c o d e _ f a l l s _ b a c k _ t o _ t r a n s a c t i o n _ i d ( )   - >   N o n e : 
         p a r s e d   =   p a r s e _ t r a n s a c t i o n ( t x ( ) ) 
         a s s e r t   p a r s e d . o r d e r _ i d   i s   N o n e 
         a s s e r t   p a r s e d . p a y m e n t _ c o d e   = =   " P _ A 9 9 T E S T P A Y X 7 1 1 1 6 " 
  
 