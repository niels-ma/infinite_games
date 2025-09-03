import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

COPY_MINER_PENALTY_POW = 2
PREDICTION_ROUND_DIGITS = 2
CREDIT_ROUND_DIGITS = 3
STD_THRESHOLD = 1e-4
MISCORRELATION_RO = 0.05


class ClusterSelector:
    """Pipeline that clusters miners, calibrates representative
    predictions, selects clusters, and allocates credit.

    Parameters
    ----------
    ranked_predictions : DataFrame
        Expected columns: ``['event_id', 'event_rank', 'outcome',
        'miner_uid', 'miner_hotkey', 'prediction']``.
    latest_metagraph_neurons : DataFrame
        Expected columns: ``['miner_uid', 'miner_hotkey']``.
    internal_forecasts : DataFrame
        Expected columns: ``['event_id', 'prediction']``.
    metagraph_block: current metagraph block
    """

    def __init__(
        self,
        ranked_predictions: pd.DataFrame,
        latest_metagraph_neurons: pd.DataFrame,
        internal_forecasts: pd.DataFrame,
        metagraph_block: int,
    ) -> None:
        # Core data
        self.ranked_predictions = ranked_predictions.copy()
        self.latest_metagraph_neurons = latest_metagraph_neurons.copy()
        self.internal_forecasts = internal_forecasts.copy()

        # Hyper-parameters / constants
        self.prediction_round_digits = PREDICTION_ROUND_DIGITS
        self.miscorrelation_ro = MISCORRELATION_RO
        self.std_threshold = STD_THRESHOLD
        self.random_seed = metagraph_block
        self.copy_miner_penalty_pow = COPY_MINER_PENALTY_POW
        self.credit_round_digits = CREDIT_ROUND_DIGITS
        self.n_bags = 10
        self.col_frac = 0.8
        self.hgb_params = dict(
            max_depth=3,
            learning_rate=0.05,
            max_iter=100,
            l2_regularization=5.0,
        )

        self.selected_clusters_credit: pd.DataFrame | None = None

    def _choose_medoid(self, matrix: pd.DataFrame) -> str:
        """Return the column label whose miner is the medoid (min sum of
        correlation distance) among *columns* of *matrix* (events × miners)."""
        dist = squareform(pdist(matrix.T, metric="correlation"))
        return matrix.columns[dist.sum(axis=0).argmin()]

    def _platt_calibrate(self, p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        p = np.clip(p, 1e-6, 1 - 1e-6)

        def nll(params):
            invT, bias = params
            p2 = expit(invT * logit(p) + bias)
            return -np.sum(y * np.log(p2) + (1 - y) * np.log(1 - p2))

        res = minimize(nll, x0=[1.0, 0.0], bounds=[(0.001, None), (None, None)])
        return float(res.x[0]), float(res.x[1])

    def _col_log_loss(self, col: pd.Series, y_true: pd.Series) -> pd.Series:
        p = np.clip(col.values, 1e-6, 1 - 1e-6)
        y = y_true.loc[col.index].values
        ll = log_loss(y, p, labels=[0, 1])
        return pd.Series({"scaled_log_loss": ll})

    def calculate_selected_clusters_credits(self) -> pd.DataFrame:
        """Run the full pipeline and return ``selected_clusters_credit``."""

        # -------------------------------------------------------------
        # 1. Prepare *recent* predictions with an internal forecaster
        # -------------------------------------------------------------
        rp = self.ranked_predictions
        im = self.internal_forecasts

        recent_events = rp[["event_id", "event_rank", "outcome"]].drop_duplicates()

        events_if = recent_events.merge(im, how="left", on="event_id")
        events_if["miner_uid"] = 1000
        events_if["miner_hotkey"] = "internal_forecaster"
        events_if.fillna(0.5, inplace=True)
        events_if = events_if[rp.columns]  # enforce same column order

        recent = pd.concat([rp, events_if], axis=0)

        # ensure cartesian completeness (every miner × event)
        miners = recent[["miner_uid", "miner_hotkey"]].drop_duplicates()
        recent = recent_events.merge(miners, how="cross").merge(
            recent,
            how="left",
            on=["event_id", "event_rank", "outcome", "miner_uid", "miner_hotkey"],
        )
        recent["prediction"] = recent["prediction"].fillna(0.5)

        # Add uniq key to both data sets
        recent["miner_key"] = recent["miner_hotkey"] + "__" + recent["miner_uid"].astype(str)
        recent["prediction"] = recent["prediction"].round(self.prediction_round_digits)
        recent["outcome_num"] = recent["outcome"].astype(int)
        recent["abs_error"] = (recent["prediction"] - recent["outcome_num"]).abs()

        # -------------------------------------------------------------
        # 2. Build metagraph incl. internal forecaster & cluster miners
        # -------------------------------------------------------------
        mg = self.latest_metagraph_neurons.copy()
        mg.loc[len(mg)] = [1000, "internal_forecaster"]
        mg["miner_key"] = mg["miner_hotkey"] + "__" + mg["miner_uid"].astype(str)

        # miners × events error matrix
        err_mat = recent.pivot_table(index="miner_key", columns="event_id", values="abs_error")

        # hierarchical clustering by correlation distance
        dist_vec = pdist(err_mat.values, metric="correlation")
        labels = fcluster(
            linkage(dist_vec, method="complete"), self.miscorrelation_ro, criterion="distance"
        )
        clusters_info = pd.DataFrame({"miner_key": err_mat.index, "cluster_id": labels})

        # Identify internal forecaster clusters
        ifc_cluster = clusters_info.loc[
            clusters_info["miner_key"] == "internal_forecaster__1000", "cluster_id"
        ].iat[0]

        # Keep only current miners in metagraph
        clusters_info = clusters_info.merge(mg, on="miner_key", how="inner")

        clusters_info = clusters_info.merge(
            clusters_info.groupby("cluster_id")["miner_key"].nunique().rename("miner_count"),
            on="cluster_id",
        )

        clusters_data = recent.merge(clusters_info, on="miner_key")

        # -------------------------------------------------------------
        # 3. Representative (medoid) & Platt calibration per cluster
        # -------------------------------------------------------------
        reps: dict[int, pd.Series] = {}
        cid_rep_map: dict[int, str] = {}

        for cid, g in clusters_data.groupby("cluster_id"):
            pivot = g.pivot_table(
                index="event_id", columns="miner_key", values="prediction"
            ).fillna(0.5)
            rep_key = pivot.columns[0] if pivot.shape[1] == 1 else self._choose_medoid(pivot)
            raw_p = pivot[rep_key]
            y_ser = (
                g.drop_duplicates("event_id")
                .set_index("event_id")
                .loc[raw_p.index, "outcome_num"]
                .astype(float)
            )
            invT, bias = self._platt_calibrate(raw_p.values, y_ser.values)
            reps[cid] = pd.Series(expit(invT * logit(raw_p) + bias), index=raw_p.index).round(
                self.prediction_round_digits
            )
            cid_rep_map[cid] = rep_key

        clusters_info["representative_key"] = clusters_info["cluster_id"].map(cid_rep_map)

        # -------------------------------------------------------------
        # 4. Build cluster-level feature matrix X & target y
        # -------------------------------------------------------------
        X = pd.concat(reps, axis=1).sort_index(axis=1)
        y = (
            clusters_data.drop_duplicates("event_id")
            .set_index("event_id")
            .loc[X.index, "outcome_num"]
        )

        # -------------------------------------------------------------
        # 5. Variance filter on clusters
        # -------------------------------------------------------------
        stds = X.std()
        clusters_info["scaled_std"] = clusters_info["cluster_id"].map(stds)
        var_cols = stds[stds >= self.std_threshold].index.to_numpy()
        var_cols = np.union1d(var_cols, ifc_cluster)  # always keep IF
        X_var = X[var_cols]

        # -------------------------------------------------------------
        # 6. Elastic-net logistic regression for sparse selection
        # -------------------------------------------------------------
        model = LogisticRegressionCV(
            penalty="elasticnet",
            solver="saga",
            l1_ratios=[0.8, 0.9, 1.0],
            Cs=np.logspace(-4, 1, 10),
            scoring="neg_log_loss",
            max_iter=20_000,
            cv=5,
            n_jobs=-1,
            refit=True,
        ).fit(X_var, y)

        coef_df = pd.DataFrame(
            {
                "cluster_id": X_var.columns,
                "coefficients": model.coef_.ravel().round(3),
            }
        )
        clusters_info = clusters_info.merge(coef_df, on="cluster_id", how="left")

        selected = coef_df[(coef_df["coefficients"] > 0) | (coef_df["cluster_id"] == ifc_cluster)]
        selected = selected.merge(
            pd.DataFrame(
                {"cluster_id": list(cid_rep_map), "representative_key": list(cid_rep_map.values())}
            ),
            on="cluster_id",
        )

        net_cols = np.union1d(selected["cluster_id"].values, ifc_cluster)
        X_net = X_var[net_cols]

        # -------------------------------------------------------------
        # 7. Log-loss filtering versus baseline / internal forecaster
        # -------------------------------------------------------------
        logloss_df = (
            X_net.apply(self._col_log_loss, axis=0, y_true=y)
            .T.reset_index()
            .rename(columns={"index": "cluster_id"})
        )
        clusters_info = clusters_info.merge(logloss_df, on="cluster_id", how="left")

        p_base = recent_events.outcome.astype(int).mean()
        p_base = 1 - p_base if p_base > 0.5 else p_base
        base_ll = -(p_base * np.log(p_base) + (1 - p_base) * np.log(1 - p_base))
        if_ll = logloss_df.loc[logloss_df["cluster_id"] == ifc_cluster, "scaled_log_loss"].iat[0]
        ref_ll = min(base_ll, if_ll)

        ll_cols = logloss_df.loc[logloss_df["scaled_log_loss"] <= ref_ll, "cluster_id"].values
        ll_cols = np.union1d(ll_cols, ifc_cluster)
        X_sel = X_net[ll_cols]

        # -------------------------------------------------------------
        # 8. Bagged leave-one-out importance (HistGB)
        # -------------------------------------------------------------
        rng_global = np.random.default_rng(self.random_seed)
        delta_sum = pd.Series(0.0, index=X.columns)
        appearances = pd.Series(0, index=X.columns)

        def cv_logloss(arr_X, arr_y, cv, rng):
            losses = []
            for tr_idx, va_idx in cv.split(arr_X, arr_y):
                mdl = HistGradientBoostingClassifier(
                    **self.hgb_params, random_state=rng.integers(0, 2**32 - 1)
                )
                mdl.fit(arr_X[tr_idx], arr_y[tr_idx])
                p = mdl.predict_proba(arr_X[va_idx])[:, 1]
                losses.append(log_loss(arr_y[va_idx], p, labels=[0, 1]))
            return float(np.mean(losses))

        bag_pool = set(X_sel.columns) - {ifc_cluster}
        bag_size = int(self.col_frac * len(bag_pool))

        if bag_size >= 1:
            for _ in range(self.n_bags):
                rng = np.random.default_rng(rng_global.integers(0, 2**32 - 1))
                cv_bag = StratifiedKFold(
                    n_splits=3, shuffle=True, random_state=rng.integers(0, 2**32 - 1)
                )
                cols = list(rng.choice(list(bag_pool), size=bag_size, replace=False)) + [
                    ifc_cluster
                ]
                X_bag = X_sel[cols]
                base_loss = cv_logloss(X_bag.values, y.values, cv_bag, rng)
                for col in cols:
                    X_minus = X_bag.drop(columns=col)
                    loss_minus = cv_logloss(X_minus.values, y.values, cv_bag, rng)
                    delta = loss_minus - base_loss  # >0 ==> helpful
                    delta_sum[col] += delta
                    appearances[col] += 1

        delta_mean = (delta_sum / appearances.clip(lower=1)).fillna(0.0).clip(lower=0)

        # -------------------------------------------------------------
        # 9. Credit allocation per cluster & per miner
        # -------------------------------------------------------------
        selected_clusters_credit = selected.merge(
            pd.DataFrame({"cluster_id": delta_mean.index, "cluster_credit": delta_mean.values}),
            on="cluster_id",
        )

        total_credit = selected_clusters_credit["cluster_credit"].sum()
        selected_clusters_credit["cluster_credit_adjusted"] = (
            0 if total_credit == 0 else selected_clusters_credit["cluster_credit"] / total_credit
        )

        selected_clusters_credit = selected_clusters_credit.merge(
            clusters_info,
            on=["cluster_id", "representative_key", "coefficients"],
            how="right",
        )

        selected_clusters_credit["raw_credit_per_miner"] = selected_clusters_credit[
            "cluster_credit_adjusted"
        ] / selected_clusters_credit["miner_count"].pow(self.copy_miner_penalty_pow)
        selected_clusters_credit["round_credit_per_miner"] = (
            np.floor(
                selected_clusters_credit["raw_credit_per_miner"] * 10**self.credit_round_digits
            )
            / 10**self.credit_round_digits
        )

        self.selected_clusters_credit = selected_clusters_credit.sort_values(
            "round_credit_per_miner", ascending=False
        )
        return self.selected_clusters_credit
