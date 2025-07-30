from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.if_games.client import IfGamesClient
from neurons.validator.models.if_games_client import MinerPrediction, PostPredictionsRequestBody
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.common.interval import (
    get_interval_iso_datetime,
    get_interval_start_minutes,
)
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class ExportPredictions(AbstractTask):
    interval: float
    api_client: IfGamesClient
    db_operations: DatabaseOperations
    batch_size: int
    validator_uid: int
    validator_hotkey: str
    logger: InfiniteGamesLogger

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        api_client: IfGamesClient,
        batch_size: int,
        validator_uid: int,
        validator_hotkey: str,
        logger: InfiniteGamesLogger,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        # Validate api_client
        if not isinstance(api_client, IfGamesClient):
            raise TypeError("api_client must be an instance of IfGamesClient.")

        # Validate batch_size
        if not isinstance(batch_size, int) or batch_size <= 0 or batch_size > 500:
            raise ValueError("batch_size must be a positive integer.")

        # Validate validator_uid
        if not isinstance(validator_uid, int) or validator_uid < 0 or validator_uid > 256:
            raise ValueError("validator_uid must be a positive integer.")

        # Validate validator_hotkey
        if not isinstance(validator_hotkey, str):
            raise TypeError("validator_hotkey must be a string.")

        # Validate logger
        if not isinstance(logger, InfiniteGamesLogger):
            raise TypeError("logger must be an instance of InfiniteGamesLogger.")

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.api_client = api_client
        self.batch_size = batch_size
        self.validator_uid = validator_uid
        self.validator_hotkey = validator_hotkey
        self.logger = logger

    @property
    def name(self):
        return "export-predictions"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        while True:
            current_interval_minutes = get_interval_start_minutes()

            # Get predictions to export
            predictions = await self.db_operations.get_predictions_to_export(
                current_interval_minutes=current_interval_minutes, batch_size=self.batch_size
            )

            if len(predictions) == 0:
                break

            # Export predictions
            parsed_predictions = self.parse_predictions_for_exporting(
                predictions_db_data=predictions
            )

            await self.api_client.post_predictions(body=parsed_predictions)

            # Mark predictions as exported
            ids = [prediction[0] for prediction in predictions]
            await self.db_operations.mark_predictions_as_exported(ids=ids)

            if len(predictions) < self.batch_size:
                break

    def parse_predictions_for_exporting(self, predictions_db_data: list[tuple[any]]):
        submissions = []

        for prediction_data in predictions_db_data:
            unique_event_id = prediction_data[1]
            miner_uid = prediction_data[2]
            miner_hotkey = prediction_data[3]
            event_type = prediction_data[4]
            predicted_outcome = prediction_data[5]
            interval_start_minutes = prediction_data[6]
            interval_agg_prediction = prediction_data[7]
            interval_count = prediction_data[8]
            submitted_at = prediction_data[9]

            prediction = MinerPrediction(
                unique_event_id=unique_event_id,
                provider_type=event_type,
                prediction=predicted_outcome,
                interval_start_minutes=interval_start_minutes,
                interval_agg_prediction=interval_agg_prediction,
                interval_agg_count=interval_count,
                interval_datetime=get_interval_iso_datetime(interval_start_minutes),
                miner_hotkey=miner_hotkey,
                miner_uid=miner_uid,
                validator_hotkey=self.validator_hotkey,
                validator_uid=self.validator_uid,
                submitted_at=submitted_at,
                title=None,
                outcome=None,
            )

            submissions.append(prediction)

        return PostPredictionsRequestBody(submissions=submissions)
