from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from neurons.validator.api.api import API
from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


@pytest.fixture
async def test_db_operations(db_client: DatabaseClient):
    logger = MagicMock(spec=InfiniteGamesLogger)

    db_operations = DatabaseOperations(db_client=db_client, logger=logger)

    return db_operations


@pytest.fixture
async def test_api_client(test_db_operations: DatabaseOperations):
    api_access_keys = "test-key"

    api_instance = API(
        host="127.0.0.1",
        port=8000,
        db_operations=test_db_operations,
        env="test",
        api_access_keys=api_access_keys,
    )

    app = api_instance.create_api()
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport, base_url="http://testserver", headers={"X-API-Key": api_access_keys}
    ) as client:
        yield client


@pytest.fixture
async def test_api_client_no_auth(test_db_operations: DatabaseOperations):
    api_access_keys = "test-key"

    api_instance = API(
        host="127.0.0.1",
        port=8000,
        db_operations=test_db_operations,
        env="test",
        api_access_keys=api_access_keys,
    )

    app = api_instance.create_api()
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={
            # No API key header
        },
    ) as client:
        yield client
