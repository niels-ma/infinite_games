import logging

import pytest

from neurons.validator.utils.logger.context import start_session, start_trace
from neurons.validator.utils.logger.formatters import JSONFormatter
from neurons.validator.utils.logger.logger import (
    api_logger,
    create_logger,
    logger,
    override_loggers_level,
)


class TestLogger:
    @pytest.fixture
    def logger(self):
        """Fixture to create a logger instance for testing."""
        return create_logger(name="test_logger")

    def test_logger_name_and_level(self, logger):
        """Test that the logger is configured with the correct name and level."""
        assert logger.name == "test_logger"

    def test_logger_handlers(self, logger):
        """Test that the logger has the correct handlers attached."""
        handlers = logger.handlers

        assert len(handlers) == 1  # Expecting 1 handler ( JSON)

        # Validate handler
        assert isinstance(handlers[0].formatter, JSONFormatter)

    def test_logger_context_methods(self, logger):
        """Test that the logger has the correct context methods."""

        # Validate the type of each context method
        assert logger.start_session == start_session
        assert logger.start_trace == start_trace

    def test_override_loggers_level(self):
        from neurons.validator.utils.logger.logger import loggers_level as loggers_level1

        assert loggers_level1 is None
        assert logger.level != logging.DEBUG
        assert api_logger.level != logging.DEBUG

        override_loggers_level(logging.DEBUG)

        from neurons.validator.utils.logger.logger import loggers_level as loggers_level2

        assert loggers_level2 == logging.DEBUG
        assert logger.level == logging.DEBUG
        assert api_logger.level == logging.DEBUG
