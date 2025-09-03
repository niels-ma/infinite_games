import sys
import time
from typing import Any, Awaitable, Callable, Iterable, Optional

import aiosqlite

from neurons.validator.alembic.migrate import run_migrations
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class DatabaseClient:
    __db_path: str
    __logger: InfiniteGamesLogger

    def __init__(self, db_path: str, logger: InfiniteGamesLogger) -> None:
        # Validate db_path
        if not isinstance(db_path, str):
            raise TypeError("db_path must be an instance of str.")

        # Validate logger
        if not isinstance(logger, InfiniteGamesLogger):
            raise TypeError("logger must be an instance of InfiniteGamesLogger.")

        self.__db_path = db_path
        self.__logger = logger

    def __get_caller_name(self) -> str:
        try:
            return sys._getframe(3).f_code.co_name
        except Exception:
            return "unknown"

    async def __wrap_execution(
        self, operation: Callable[[aiosqlite.Connection], Awaitable[any]]
    ) -> Awaitable[Any]:
        start_time = time.time()

        caller = self.__get_caller_name()

        connection = None

        try:
            connection = await aiosqlite.connect(self.__db_path, timeout=180)

            # Needed for each connection
            await connection.execute("PRAGMA foreign_keys = ON")

            query_start_time = time.time()

            response = await operation(connection)

            query_ms = round((time.time() - query_start_time) * 1000)
            elapsed_time_ms = round((time.time() - start_time) * 1000)

            extra = {
                "caller": caller,
                "query_ms": query_ms,
                "elapsed_time_ms": elapsed_time_ms,
            }

            if elapsed_time_ms > 500:
                log_method = self.__logger.warning
            else:
                log_method = self.__logger.debug

            log_method("SQL executed", extra=extra)

            return response
        except Exception as e:
            elapsed_time_ms = round((time.time() - start_time) * 1000)

            extra = {"elapsed_time_ms": elapsed_time_ms, "caller": caller}

            self.__logger.exception("SQL errored", extra=extra)

            raise e
        finally:
            if connection is not None:
                await connection.close()

    async def insert(
        self, sql: str, parameters: Optional[Iterable[Any]] = None
    ) -> Iterable[aiosqlite.Row]:
        async def execute(connection: aiosqlite.Connection):
            cursor = await connection.execute(sql=sql, parameters=parameters)
            inserted = await cursor.fetchall()

            await cursor.close()
            await connection.commit()

            return inserted

        return await self.__wrap_execution(execute)

    async def script(
        self,
        sql_script: str,
    ) -> None:
        async def execute(connection: aiosqlite.Connection):
            cursor = await connection.executescript(sql_script=sql_script)

            await cursor.close()
            await connection.commit()

            return None

        return await self.__wrap_execution(execute)

    async def insert_many(self, sql: str, parameters: Iterable[Iterable[Any]]) -> None:
        async def execute(connection: aiosqlite.Connection):
            cursor = await connection.executemany(sql=sql, parameters=parameters)

            await cursor.close()
            await connection.commit()

            # Execute many does not support returning
            # https://github.com/python/cpython/issues/100021
            return None

        return await self.__wrap_execution(execute)

    async def delete(
        self, sql: str, parameters: Optional[Iterable[Any]] = None
    ) -> Iterable[aiosqlite.Row]:
        async def execute(connection: aiosqlite.Connection):
            cursor = await connection.execute(sql=sql, parameters=parameters)
            deleted = await cursor.fetchall()

            await cursor.close()
            await connection.commit()

            return deleted

        return await self.__wrap_execution(execute)

    async def update(
        self, sql: str, parameters: Optional[Iterable[Any]] = None
    ) -> Iterable[aiosqlite.Row]:
        async def execute(connection: aiosqlite.Connection):
            cursor = await connection.execute(sql=sql, parameters=parameters)
            updated = await cursor.fetchall()

            await cursor.close()
            await connection.commit()

            return updated

        return await self.__wrap_execution(execute)

    async def one(
        self,
        sql: str,
        parameters: Optional[Iterable[Any]] = None,
        use_row_factory: bool = False,
    ) -> Optional[aiosqlite.Row]:
        async def execute(connection: aiosqlite.Connection):
            if use_row_factory:
                connection.row_factory = aiosqlite.Row

            cursor = await connection.execute(sql=sql, parameters=parameters)
            row = await cursor.fetchone()

            await cursor.close()

            return row

        return await self.__wrap_execution(execute)

    async def many(
        self,
        sql: str,
        parameters: Optional[Iterable[Any]] = None,
        use_row_factory: bool = False,
    ) -> Iterable[aiosqlite.Row]:
        async def execute(connection: aiosqlite.Connection):
            if use_row_factory:
                connection.row_factory = aiosqlite.Row
            cursor = await connection.execute(sql=sql, parameters=parameters)

            # Note query should be paginated
            rows = await cursor.fetchall()

            await cursor.close()

            if len(rows) > 500:
                self.__logger.warning("Query returning many rows", extra={"rows": len(rows)})

            return rows

        return await self.__wrap_execution(execute)

    async def migrate(self):
        start_time = time.time()

        self.__logger.info("Running migrations")

        run_migrations(db_file_name=self.__db_path)

        elapsed_time_ms = round((time.time() - start_time) * 1000)

        self.__logger.info("Migrations complete", extra={"elapsed_time_ms": elapsed_time_ms})
