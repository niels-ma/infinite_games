import asyncio
import base64
import json
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.config import IfgamesEnvType
from neurons.validator.utils.env import ENVIRONMENT_VARIABLES
from neurons.validator.utils.logger.logger import InfiniteGamesLogger

MODEL_NAME = "community_prediction_lr"
TOP_N_RANKS = {
    "test": 20,
    "prod": 100,
}


class TrainCommunityPredictionModel(AbstractTask):
    interval: float
    db_operations: DatabaseOperations
    logger: InfiniteGamesLogger
    env: IfgamesEnvType

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        logger: InfiniteGamesLogger,
        env: IfgamesEnvType,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        self.interval = interval_seconds
        self.db_operations = db_operations

        self.errors_count = 0
        self.logger = logger
        self.env = env

    @property
    def name(self):
        return "train-cp-model"

    @property
    def interval_seconds(self):
        return self.interval

    async def get_data(self) -> pd.DataFrame:
        raw_data = await self.db_operations.get_community_train_dataset()
        if not raw_data:
            self.logger.error("No data to train the community prediction model.")
            return None

        df = pd.DataFrame.from_records(raw_data, columns=raw_data[0].keys())
        df = df.astype(
            {
                "miner_uid": "int32",
                "miner_hotkey": "str",
                "miner_rank": "int32",
                "prev_batch_miner_rank": "int32",
                "prev_metagraph_score": "float32",
                "event_id": "str",
                "event_rank": "int32",
                "scoring_batch": "datetime64[ns]",
                "batch_rank": "int32",
                "event_batch_rank": "int32",
                "outcome": "int32",
                "agg_prediction": "float32",
                "metagraph_score": "float32",
            }
        )

        self.logger.debug(
            "Data fetched successfully from the database.",
            extra={
                "num_rows": len(df),
                "distinct_events": df["event_id"].nunique(),
                "distinct_batches": df["scoring_batch"].nunique(),
                "distinct_miner_uids": df["miner_uid"].nunique(),
                "distinct_miner_ranks": df["miner_rank"].nunique(),
            },
        )

        return df

    @staticmethod
    def pivot_top_n(df: pd.DataFrame, n: int = 10, train: bool = True) -> pd.DataFrame:
        """
        Pivot the DataFrame to create a wide format for the top N miners which act as features.
        """
        top_n = df[df["prev_batch_miner_rank"] <= n].copy()

        if top_n["prev_batch_miner_rank"].nunique() != n:
            raise ValueError(
                f"""Expected {n} unique prev_batch_miner_rank values, """
                + f"""but got {top_n['prev_batch_miner_rank'].nunique()}"""
            )

        if set(top_n["prev_batch_miner_rank"].unique().tolist()) != set(range(1, n + 1)):
            raise ValueError(
                f"""Expected prev_batch_miner_rank values to be in the range 1 to {n}, """
                + f"""but got {top_n['prev_batch_miner_rank'].unique().tolist()}"""
            )

        relevant_columns = ["event_id", "batch_rank", "prev_batch_miner_rank", "agg_prediction"]
        index_cols = ["event_id", "batch_rank"]
        if train:
            relevant_columns.append("outcome")
            index_cols.append("outcome")

        features = top_n[relevant_columns].copy()
        wide = features.pivot_table(
            index=index_cols,
            columns="prev_batch_miner_rank",
            values="agg_prediction",
            observed=True,
        )
        pred_columns = [f"prediction_rank_{int(r)}" for r in wide.columns]
        wide.columns = pred_columns
        wide = wide.reset_index()

        # fill NaNs with row means
        row_means = wide[pred_columns].mean(axis=1)
        # Using a loop here instead of the vectorized fillna() method because we are applying
        # row-wise means to each column individually, which ensures that each column's NaNs
        # are filled with the corresponding row's mean value.
        for c in pred_columns:
            wide[c] = wide[c].fillna(row_means)
        return wide

    @staticmethod
    def train_model(wide_df: pd.DataFrame) -> LogisticRegression:
        """
        Train the model using the wide df.
        """

        X = wide_df.drop(columns=["event_id", "batch_rank", "outcome"])
        y = wide_df["outcome"]

        lr_model = LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            max_iter=1000,
            l1_ratio=0.5,
            class_weight="balanced",
            C=1.0,
            n_jobs=-1,
        )

        lr_model.fit(X, y)

        return lr_model

    async def save_model(self, model: LogisticRegression, other_data: dict | None = None):
        """
        Save the model to the database.
        """
        other_data_json = json.dumps(other_data) if other_data else None
        pickled_model = pickle.dumps(model)
        b64_model_str = base64.b64encode(pickled_model).decode("ascii")

        await self.db_operations.save_community_predictions_model(
            name=MODEL_NAME,
            model_blob=b64_model_str,
            other_data_json=other_data_json,
        )
        self.logger.debug("Community Prediction model saved successfully to DB.")

    async def run(self):
        if not ENVIRONMENT_VARIABLES.API_ACCESS_KEYS:
            self.logger.debug("API access keys are not set - will not train the model.")
            return

        df = await self.get_data()
        if df is None:
            return

        wide_df = self.pivot_top_n(df, n=TOP_N_RANKS[self.env], train=True)

        lr_model = await asyncio.to_thread(self.train_model, wide_df)

        self.logger.debug(
            "Model trained successfully.",
            extra={
                "coefficients": lr_model.coef_.tolist(),
                "intercept": lr_model.intercept_.tolist(),
            },
        )

        await self.save_model(lr_model)

        self.logger.debug("Community Prediction model training completed successfully.")
