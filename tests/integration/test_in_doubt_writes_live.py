"""Live integration: a write whose outcome is in doubt raises AmbiguousCommitError.

The faults come from dqlitetestlib's NetworkFaults (python-dqlite-dev): a one-way
partition that makes the leader step down, and a COMMIT whose reply is dropped.
In both the write is applied; the error must not read as a clean failure.
Restores the original leader on exit.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import pytest

from dqlitedbapi.aio import AsyncConnection
from dqlitedbapi.exceptions import AmbiguousCommitError

if TYPE_CHECKING:
    from dqlitetestlib import NetworkFaults, TestClusterControl  # type: ignore[import-not-found]


def _port(address: str) -> int:
    return int(address.rsplit(":", 1)[1])


async def _rows(cluster_address: str, database: str) -> list[Any]:
    last: Exception | None = None
    for _ in range(60):
        observer = AsyncConnection(cluster_address, database=database, timeout=3)
        try:
            cursor = await observer.execute("SELECT id FROM t")
            return list(await cursor.fetchall())
        except Exception as exc:  # noqa: BLE001 - a new leader is still being elected
            last = exc
            await asyncio.sleep(0.5)
        finally:
            await observer.close()
    raise AssertionError(f"cluster did not serve reads again: {last!r}")


@contextlib.asynccontextmanager
async def _original_leader_restored(cluster_control: TestClusterControl) -> AsyncIterator[None]:
    original = await cluster_control.current_leader_node()
    try:
        yield
    finally:
        for _ in range(60):
            try:
                if await cluster_control.find_leader() == original.address:
                    break
                await cluster_control.transfer_leadership_to(original.node_id)
            except Exception:  # noqa: BLE001 - mid-election
                pass
            await asyncio.sleep(0.5)


@pytest.mark.integration
async def test_autocommit_write_when_leadership_is_lost(
    cluster_address: str,
    cluster_control: TestClusterControl,
    network_faults: NetworkFaults,
) -> None:
    database = "in_doubt_" + uuid.uuid4().hex
    async with _original_leader_restored(cluster_control):
        conn = AsyncConnection(cluster_address, database=database, timeout=15)
        try:
            await conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            assert conn._client is not None
            with (
                network_faults.isolate_from_followers(_port(conn._client.address)),
                pytest.raises(AmbiguousCommitError),
            ):
                await conn.execute("INSERT INTO t VALUES (42)")
        finally:
            await conn.close()
        assert await _rows(cluster_address, database) == [(42,)]


@pytest.mark.integration
async def test_commit_whose_reply_is_lost(
    cluster_address: str,
    cluster_control: TestClusterControl,
    network_faults: NetworkFaults,
) -> None:
    database = "in_doubt_" + uuid.uuid4().hex
    async with _original_leader_restored(cluster_control):
        conn = AsyncConnection(cluster_address, database=database, timeout=3)
        try:
            await conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            await conn.execute("BEGIN")
            await conn.execute("INSERT INTO t VALUES (7)")
            client = conn._client
            assert client is not None and client._protocol is not None
            client_port = client._protocol._writer.get_extra_info("sockname")[1]
            with (
                network_faults.drop_replies(_port(client.address), client_port),
                pytest.raises(AmbiguousCommitError),
            ):
                await conn.commit()
        finally:
            await conn.close()
        assert await _rows(cluster_address, database) == [(7,)]
