import asyncio
import sqlite3
import sys

from bittensor import Dendrite, Subtensor
from bittensor_wallet import Wallet

from neurons.validator.api.api import API
from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.if_games.client import IfGamesClient
from neurons.validator.scheduler.tasks_scheduler import TasksScheduler
from neurons.validator.tasks.db_cleaner import DbCleaner
from neurons.validator.tasks.db_vacuum import DbVacuum
from neurons.validator.tasks.delete_events import DeleteEvents
from neurons.validator.tasks.export_predictions import ExportPredictions
from neurons.validator.tasks.export_scores import ExportScores
from neurons.validator.tasks.metagraph_scoring import MetagraphScoring
from neurons.validator.tasks.peer_scoring import PeerScoring
from neurons.validator.tasks.pull_events import PullEvents
from neurons.validator.tasks.query_miners import QueryMiners
from neurons.validator.tasks.resolve_events import ResolveEvents
from neurons.validator.tasks.set_weights import SetWeights
from neurons.validator.tasks.train_cp_model import TrainCommunityPredictionModel
from neurons.validator.utils.common.event_loop import measure_event_loop_lag
from neurons.validator.utils.config import get_config
from neurons.validator.utils.env import ENVIRONMENT_VARIABLES, assert_requirements
from neurons.validator.utils.logger.logger import (
    logger,
    override_loggers_level,
    set_bittensor_logger,
)


async def main():
    # Assert system requirements
    assert_requirements()

    # Start session id
    logger.start_session()

    # Set dependencies
    config, ifgames_env, db_path, logging_level = get_config()

    # Loggers
    override_loggers_level(logging_level)
    set_bittensor_logger()

    # Bittensor stuff
    bt_netuid = config.get("netuid")
    bt_network = config.get("subtensor").get("network")
    bt_wallet = Wallet(config=config)
    bt_dendrite = Dendrite(wallet=bt_wallet)
    bt_subtensor = Subtensor(config=config)
    bt_metagraph = bt_subtensor.metagraph(netuid=bt_netuid, lite=True)

    validator_hotkey = bt_wallet.hotkey.ss58_address
    validator_uid = bt_metagraph.hotkeys.index(validator_hotkey)

    # Components
    db_client = DatabaseClient(db_path=db_path, logger=logger)
    db_operations = DatabaseOperations(db_client=db_client, logger=logger)
    ifgames_api_client = IfGamesClient(env=ifgames_env, logger=logger, bt_wallet=bt_wallet)

    validator_api = API(
        host="0.0.0.0",
        port=8000,
        db_operations=db_operations,
        env=ifgames_env,
        api_access_keys=ENVIRONMENT_VARIABLES.API_ACCESS_KEYS,
    )

    # Migrate db
    await db_client.migrate()

    # Tasks
    pull_events_task = PullEvents(
        interval_seconds=50.0,
        page_size=50,
        db_operations=db_operations,
        api_client=ifgames_api_client,
    )

    resolve_events_task = ResolveEvents(
        interval_seconds=1800.0,
        db_operations=db_operations,
        api_client=ifgames_api_client,
        page_size=100,
        logger=logger,
    )

    delete_events_task = DeleteEvents(
        interval_seconds=1800.0,
        db_operations=db_operations,
        api_client=ifgames_api_client,
        page_size=100,
        logger=logger,
    )

    query_miners_task = QueryMiners(
        interval_seconds=180.0,
        db_operations=db_operations,
        dendrite=bt_dendrite,
        metagraph=bt_metagraph,
        env=ifgames_env,
        logger=logger,
    )

    export_predictions_task = ExportPredictions(
        interval_seconds=180.0,
        db_operations=db_operations,
        api_client=ifgames_api_client,
        batch_size=300,
        validator_uid=validator_uid,
        validator_hotkey=validator_hotkey,
        logger=logger,
    )

    peer_scoring_task = PeerScoring(
        interval_seconds=307.0,
        db_operations=db_operations,
        metagraph=bt_metagraph,
        logger=logger,
        page_size=100,
    )

    metagraph_scoring_task = MetagraphScoring(
        interval_seconds=347.0,
        db_operations=db_operations,
        page_size=1000,
        logger=logger,
    )

    export_scores_task = ExportScores(
        interval_seconds=373.0,
        page_size=500,
        db_operations=db_operations,
        api_client=ifgames_api_client,
        logger=logger,
        validator_uid=validator_uid,
        validator_hotkey=validator_hotkey,
    )

    set_weights_task = SetWeights(
        interval_seconds=379.0,
        db_operations=db_operations,
        logger=logger,
        metagraph=bt_metagraph,
        netuid=bt_netuid,
        subtensor=bt_subtensor,
        wallet=bt_wallet,
    )

    train_cp_model_task = TrainCommunityPredictionModel(
        interval_seconds=14400.0,
        db_operations=db_operations,
        logger=logger,
        env=ifgames_env,
    )

    db_cleaner_task = DbCleaner(
        interval_seconds=53.0, db_operations=db_operations, batch_size=4000, logger=logger
    )

    vacuum_task = DbVacuum(
        interval_seconds=300.0,
        db_operations=db_operations,
        logger=logger,
        pages=500,
    )

    # Add tasks to scheduler
    scheduler = TasksScheduler(logger=logger)

    scheduler.add(task=pull_events_task)
    scheduler.add(task=resolve_events_task)
    scheduler.add(task=delete_events_task)
    scheduler.add(task=query_miners_task)
    scheduler.add(task=export_predictions_task)
    scheduler.add(task=peer_scoring_task)
    scheduler.add(task=metagraph_scoring_task)
    scheduler.add(task=export_scores_task)
    scheduler.add(task=set_weights_task)
    scheduler.add(task=train_cp_model_task)
    scheduler.add(task=db_cleaner_task)
    scheduler.add(task=vacuum_task)

    # Start Validator API
    validator_api_task = asyncio.create_task(validator_api.start())

    # Start scheduler
    scheduler_task = asyncio.create_task(scheduler.start())

    # Measure event loop lag
    loop_lag_task = asyncio.create_task(
        measure_event_loop_lag(
            measuring_frequency_seconds=0.5, lag_threshold_seconds=1.0, logger=logger
        )
    )

    logger.info(
        "Validator started",
        extra={
            "validator_uid": validator_uid,
            "validator_hotkey": validator_hotkey,
            "bt_network": bt_network,
            "bt_netuid": bt_netuid,
            "ifgames_env": ifgames_env,
            "db_path": db_path,
            "python": sys.version,
            "sqlite": sqlite3.sqlite_version,
        },
    )

    await asyncio.gather(scheduler_task, validator_api_task, loop_lag_task)
