import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from bittensor.core.config import Config

from neurons.validator.utils.config import VALID_NETWORK_CONFIGS, get_config

DEFAULT_DB_DIRECTORY = str(Path.cwd())


class TestConfig:
    @pytest.fixture
    def mock_subtensor(self):
        with patch("neurons.validator.utils.config.AsyncSubtensor") as mock:
            # Mock the add_args method
            mock.add_args = MagicMock()
            yield mock

    @pytest.fixture
    def mock_logging_machine(self):
        with patch("neurons.validator.utils.config.LoggingMachine") as mock:
            # Mock the add_args method
            mock.add_args = MagicMock()
            yield mock

    @pytest.fixture
    def mock_wallet(self):
        with patch("neurons.validator.utils.config.Wallet") as mock:
            # Mock the add_args method
            mock.add_args = MagicMock()
            yield mock

    @pytest.fixture
    def mock_config(self):
        with patch("neurons.validator.utils.config.Config") as mock:
            yield mock

    def create_mock_args(
        self,
        netuid=None,
        network=None,
        ifgames_env=None,
        db_directory=None,
        logging_trace=None,
        logging_debug=None,
        logging_info=None,
    ):
        """Helper method to create a mock args object with the required attributes"""
        mock_args = MagicMock()

        mock_args.__getattribute__ = MagicMock()
        mock_args.__getattribute__.side_effect = lambda x: {
            "netuid": netuid,
            "subtensor.network": network,
            "ifgames.env": ifgames_env,
            "db.directory": db_directory,
            "logging.trace": logging_trace,
            "logging.debug": logging_debug,
            "logging.info": logging_info,
        }.get(x)

        return mock_args

    def test_arg_additions(self, mock_subtensor, mock_logging_machine, mock_wallet):
        """Test that all component add_args methods are called"""
        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            mock_parse_args.return_value = self.create_mock_args(netuid=6, network="finney")

            config, env, db_path, logger_level = get_config()

            mock_subtensor.add_args.assert_called_once()
            mock_logging_machine.add_args.assert_called_once()
            mock_wallet.add_args.assert_called_once()

            assert isinstance(config, Config)
            assert env == "prod"
            assert db_path == str(Path(DEFAULT_DB_DIRECTORY) / "validator.db")
            assert logger_level == logging.WARNING

    def test_required_args_missing(self):
        """Test behavior when required arguments are missing"""
        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            mock_parse_args.return_value = self.create_mock_args(
                netuid=None, network=None, ifgames_env=None, db_directory=None
            )

            with pytest.raises(ValueError):
                get_config()

    def test_db_directory_validation(self):
        valid_dir = "/parent/test_db_dir"

        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            mock_parse_args.return_value = self.create_mock_args(
                netuid=6, network="finney", ifgames_env=None, db_directory=valid_dir
            )
            _, env, db_path, _ = get_config()

            assert env == "prod"
            assert db_path == str(Path(valid_dir) / "validator.db")

        invalid_dir = "./"

        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            mock_parse_args.return_value = self.create_mock_args(
                netuid=6, network="finney", ifgames_env=None, db_directory=invalid_dir
            )
            with pytest.raises(
                ValueError, match=f"Invalid db.directory '{invalid_dir}' must be an absolute path."
            ):
                get_config()

    @pytest.mark.parametrize("test_config", VALID_NETWORK_CONFIGS)
    def test_valid_network_configs_constant(self, test_config: dict[str, any]):
        """Test that all configurations in VALID_NETWORK_CONFIGS are properly configured"""

        assert isinstance(test_config, dict)
        assert "subtensor.network" in test_config
        assert "netuid" in test_config
        assert "ifgames.env" in test_config
        assert isinstance(test_config["netuid"], int)
        assert test_config["subtensor.network"] in ["finney", "test", "local", None]
        assert test_config["ifgames.env"] in ["prod", "test", None]

    @pytest.mark.parametrize(
        "logging_trace,logging_debug,logging_info,expected_logger_level",
        [
            (True, True, True, logging.DEBUG),
            (True, None, None, logging.DEBUG),
            (None, True, None, logging.DEBUG),
            (None, None, True, logging.INFO),
            (None, None, None, logging.WARNING),
        ],
    )
    def test_logger_args(self, logging_trace, logging_debug, logging_info, expected_logger_level):
        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            mock_parse_args.return_value = self.create_mock_args(
                netuid=6,
                network="finney",
                logging_trace=logging_trace,
                logging_debug=logging_debug,
                logging_info=logging_info,
            )

            _, _, _, logger_level = get_config()

            assert logger_level == expected_logger_level

    @pytest.mark.parametrize(
        "test_args,expected_valid,expected_env,expected_db_path",
        [
            ((6, "finney", None, "/p1/p2"), True, "prod", "/p1/p2/validator.db"),
            ((155, "test", None, "/p1/"), True, "test", "/p1/validator_test.db"),
            (
                (6, "local", "prod", ""),
                True,
                "prod",
                str(Path(DEFAULT_DB_DIRECTORY) / "validator.db"),
            ),
            ((155, "local", "test", "/"), True, "test", "/validator_test.db"),
            (
                (6, None, "prod", None),
                True,
                "prod",
                str(Path(DEFAULT_DB_DIRECTORY) / "validator.db"),
            ),
            (
                (155, None, "test", None),
                True,
                "test",
                str(Path(DEFAULT_DB_DIRECTORY) / "validator_test.db"),
            ),
            (
                (6, "ws://fake_ip:fake_port", "prod", None),
                True,
                "prod",
                str(Path(DEFAULT_DB_DIRECTORY) / "validator.db"),
            ),
            (
                (155, "ws://fake_ip:fake_port", "test", None),
                True,
                "test",
                str(Path(DEFAULT_DB_DIRECTORY) / "validator_test.db"),
            ),
            ((7, "finney", None, None), False, None, None),
            ((6, "invalid", None, None), False, None, None),
            ((6, "local", None, None), False, None, None),
            ((155, "local", "invalid", None), False, None, None),
            ((155, None, "prod", None), False, None, None),
            ((6, None, "test", None), False, None, None),
            ((6, "ws://fake_ip:fake_port", "test", None), False, None, None),
            ((155, "ws://fake_ip:fake_port", "prod", None), False, None, None),
        ],
    )
    def test_configurations(
        self,
        test_args: tuple,
        expected_valid: bool,
        expected_env: str | None,
        expected_db_path: str | None,
        mock_config,
    ):
        """Test various network configurations for validity"""
        netuid, network, ifgames_env, db_directory = test_args
        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            mock_parse_args.return_value = self.create_mock_args(
                netuid=netuid, network=network, ifgames_env=ifgames_env, db_directory=db_directory
            )

            if expected_valid:
                _, env, db_path, _ = get_config()

                mock_config.assert_called_once()

                # Assert env
                assert env == expected_env

                # Assert db_path
                assert db_path == expected_db_path
            else:
                with pytest.raises(ValueError) as exc_info:
                    get_config()
                assert "Invalid netuid" in str(exc_info.value)
