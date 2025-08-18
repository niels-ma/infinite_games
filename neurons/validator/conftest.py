import tempfile
from unittest.mock import MagicMock

import pytest

from neurons.validator.db.client import DatabaseClient
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


@pytest.fixture(scope="function")
async def mocked_if_logger():
    logger = MagicMock(spec=InfiniteGamesLogger)

    return logger


@pytest.fixture(scope="function")
async def db_client(mocked_if_logger: InfiniteGamesLogger):
    temp_db = tempfile.NamedTemporaryFile(delete=False)
    db_path = temp_db.name
    temp_db.close()

    db_client = DatabaseClient(db_path=db_path, logger=mocked_if_logger)

    await db_client.migrate()

    yield db_client
