from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.if_games.client import IfGamesClient
from neurons.validator.models.event import EventsModel
from neurons.validator.models.if_games_client import MinerScore, PostScoresRequestBody
from neurons.validator.models.score import ScoresModel
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class ExportScores(AbstractTask):
    interval: float
    page_size: int
    db_operations: DatabaseOperations
    api_client: IfGamesClient
    logger: InfiniteGamesLogger
    validator_uid: int
    validator_hotkey: str

    def __init__(
        self,
        interval_seconds: float,
        page_size: int,
        db_operations: DatabaseOperations,
        api_client: IfGamesClient,
        logger: InfiniteGamesLogger,
        validator_uid: int,
        validator_hotkey: str,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        self.interval = interval_seconds
        self.page_size = page_size
        self.db_operations = db_operations
        self.api_client = api_client
        self.validator_uid = validator_uid
        self.validator_hotkey = validator_hotkey

        self.errors_count = 0
        self.logger = logger

    @property
    def name(self):
        return "export-scores"

    @property
    def interval_seconds(self):
        return self.interval

    def prepare_scores_payload(self, event: EventsModel, db_scores: list[ScoresModel]):
        scores = []

        for db_score in db_scores:
            # override the spec version to be at least 1039 - peer scoring start
            # also backend expects a string
            backend_spec_version = str(max(db_score.spec_version, 1039))

            score = MinerScore(
                event_id=db_score.event_id,  # awful: backend reconstructs unique_event_id
                prediction=db_score.prediction,
                answer=float(event.outcome),
                miner_hotkey=db_score.miner_hotkey,
                miner_uid=db_score.miner_uid,
                miner_score=db_score.event_score,
                miner_effective_score=db_score.alternative_metagraph_score,
                validator_hotkey=self.validator_hotkey,
                validator_uid=self.validator_uid,
                spec_version=backend_spec_version,
                registered_date=event.registered_date,
                scored_at=db_score.created_at,
            )

            scores.append(score)

        return PostScoresRequestBody(results=scores)

    async def export_scores_to_backend(self, payload: PostScoresRequestBody):
        await self.api_client.post_scores(body=payload)

        self.logger.debug(
            "Exported scores.",
            extra={
                "event_id": payload.results[0].event_id,
                "n_scores": len(payload.results),
            },
        )

    async def run(self):
        scored_events = await self.db_operations.get_peer_scored_events_for_export_alternative(
            max_events=self.page_size
        )
        if not scored_events:
            self.logger.debug("No peer scored events to export scores.")
        else:
            self.logger.debug(
                "Found peer scored events to export scores.",
                extra={"n_events": len(scored_events)},
            )

            for event in scored_events:
                scores = await self.db_operations.get_peer_scores_for_export_alternative(
                    event_id=event.event_id
                )
                if not len(scores) > 0:
                    self.errors_count += 1
                    self.logger.warning(
                        "No peer scores found for event.",
                        extra={"event_id": event.event_id},
                    )
                    continue

                payload = self.prepare_scores_payload(event=event, db_scores=scores)

                try:
                    await self.export_scores_to_backend(payload)
                except Exception:
                    self.errors_count += 1
                    self.logger.exception(
                        "Failed to export scores.",
                        extra={"event_id": event.event_id},
                    )
                    continue

                await self.db_operations.mark_peer_scores_as_exported(event_id=event.event_id)

                await self.db_operations.mark_event_as_exported(
                    unique_event_id=event.unique_event_id
                )

        self.logger.debug(
            "Export scores task completed.",
            extra={"errors_count": self.errors_count},
        )

        self.errors_count = 0
