import pandas as pd
from pandas.testing import assert_frame_equal

from neurons.validator.utils.cluster_selector.cluster_selector import ClusterSelector


class TestClusterSelector:
    def test_prepare_events_predictions(self):
        # ranked predictions: 2 events, 2 miners, each predicts only one event
        ranked_predictions = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "event_rank": 1,
                    "outcome": 1,
                    "miner_uid": 1,
                    "miner_hotkey": "hotkey_1",
                    "prediction": 0.7,
                },
                {
                    "event_id": 2,
                    "event_rank": 2,
                    "outcome": 0,
                    "miner_uid": 2,
                    "miner_hotkey": "hotkey_2",
                    "prediction": 0.3,
                },
            ]
        )

        # internal forecasts only for event 1; event 2 should default to 0.5
        internal_forecasts = pd.DataFrame([{"event_id": 1, "prediction": 0.8}])

        unit = ClusterSelector(
            ranked_predictions=pd.DataFrame(),
            latest_metagraph_neurons=pd.DataFrame(),
            internal_forecasts=pd.DataFrame(),
            random_seed=0,
        )
        events_predictions, events = unit.prepare_events_predictions(
            ranked_predictions=ranked_predictions,
            internal_forecasts=internal_forecasts,
            prediction_round_digits=2,
        )

        # Assert events
        expected_events = pd.DataFrame(
            [
                {"event_id": 1, "event_rank": 1, "outcome": 1},
                {"event_id": 2, "event_rank": 2, "outcome": 0},
            ]
        )
        assert_frame_equal(events, expected_events)

        # Assert events predictions (cartesian 2 events × 3 miners = 6 rows)
        expected_events_predictions = pd.DataFrame(
            [
                # event 1
                {
                    "event_id": 1,
                    "event_rank": 1,
                    "outcome": 1,
                    "miner_uid": 1,
                    "miner_hotkey": "hotkey_1",
                    "prediction": 0.70,
                    "miner_key": "hotkey_1__1",
                    "outcome_num": 1,
                    "abs_error": 0.30,
                },
                #  Event 1 for miner 2 gets default
                {
                    "event_id": 1,
                    "event_rank": 1,
                    "outcome": 1,
                    "miner_uid": 2,
                    "miner_hotkey": "hotkey_2",
                    "prediction": 0.50,
                    "miner_key": "hotkey_2__2",
                    "outcome_num": 1,
                    "abs_error": 0.50,
                },
                {
                    "event_id": 1,
                    "event_rank": 1,
                    "outcome": 1,
                    "miner_uid": 1000,
                    "miner_hotkey": "internal_forecaster",
                    "prediction": 0.80,
                    "miner_key": "internal_forecaster__1000",
                    "outcome_num": 1,
                    "abs_error": 0.20,
                },
                # event 2
                {
                    #  Event 2 for miner 1 gets default
                    "event_id": 2,
                    "event_rank": 2,
                    "outcome": 0,
                    "miner_uid": 1,
                    "miner_hotkey": "hotkey_1",
                    "prediction": 0.50,
                    "miner_key": "hotkey_1__1",
                    "outcome_num": 0,
                    "abs_error": 0.50,
                },
                {
                    "event_id": 2,
                    "event_rank": 2,
                    "outcome": 0,
                    "miner_uid": 2,
                    "miner_hotkey": "hotkey_2",
                    "prediction": 0.30,
                    "miner_key": "hotkey_2__2",
                    "outcome_num": 0,
                    "abs_error": 0.30,
                },
                {
                    #  Event 2 for internal forecaster gets default
                    "event_id": 2,
                    "event_rank": 2,
                    "outcome": 0,
                    "miner_uid": 1000,
                    "miner_hotkey": "internal_forecaster",
                    "prediction": 0.50,
                    "miner_key": "internal_forecaster__1000",
                    "outcome_num": 0,
                    "abs_error": 0.50,
                },
            ]
        )

        assert_frame_equal(events_predictions, expected_events_predictions)

    def test_prepare_events_predictions_with_internal_rounding_and_abs_error(self):
        # Single event, single miner with a prediction to test rounding and abs error
        ranked_predictions = pd.DataFrame(
            [
                {
                    "event_id": 10,
                    "event_rank": 1,
                    "outcome": 1,
                    "miner_uid": 42,
                    "miner_hotkey": "hotkey_42",
                    "prediction": 0.126,
                },
            ]
        )

        internal_forecasts = pd.DataFrame(
            [], columns=["event_id", "prediction"]
        )  # no internal forecasts provided

        events_predictions, events = ClusterSelector.prepare_events_predictions(
            ranked_predictions=ranked_predictions,
            internal_forecasts=internal_forecasts,
            prediction_round_digits=2,
        )

        # Assert events
        expected_events = pd.DataFrame([{"event_id": 10, "event_rank": 1, "outcome": 1}])

        assert_frame_equal(events, expected_events)

        # Assert events predictions
        expected_events_predictions = pd.DataFrame(
            [
                {
                    "event_id": 10,
                    "event_rank": 1,
                    "outcome": 1,
                    "miner_uid": 42,
                    "miner_hotkey": "hotkey_42",
                    "prediction": 0.13,
                    "miner_key": "hotkey_42__42",
                    "outcome_num": 1,
                    "abs_error": 0.87,
                },
                {
                    "event_id": 10,
                    "event_rank": 1,
                    "outcome": 1,
                    "miner_uid": 1000,
                    "miner_hotkey": "internal_forecaster",
                    "prediction": 0.50,
                    "miner_key": "internal_forecaster__1000",
                    "outcome_num": 1,
                    "abs_error": 0.50,
                },
            ]
        )

        assert_frame_equal(events_predictions, expected_events_predictions)

    def test_cluster_miners(self):
        # Two events with opposite outcomes
        ranked_predictions = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "event_rank": 1,
                    "outcome": 1,
                    "miner_uid": 1,
                    "miner_hotkey": "hotkey_1",
                    "prediction": 0.60,
                },
                {
                    "event_id": 1,
                    "event_rank": 1,
                    "outcome": 1,
                    "miner_uid": 2,
                    "miner_hotkey": "hotkey_2",
                    "prediction": 0.60,
                },
                {
                    "event_id": 2,
                    "event_rank": 2,
                    "outcome": 0,
                    "miner_uid": 1,
                    "miner_hotkey": "hotkey_1",
                    "prediction": 0.60,
                },
                {
                    "event_id": 2,
                    "event_rank": 2,
                    "outcome": 0,
                    "miner_uid": 2,
                    "miner_hotkey": "hotkey_2",
                    "prediction": 0.60,
                },
            ]
        )

        internal_forecasts = pd.DataFrame(
            [
                {"event_id": 1, "prediction": 0.59},
                {"event_id": 2, "prediction": 0.59},
            ]
        )

        # Latest metagraph includes only the two external miners
        latest_metagraph_neurons = pd.DataFrame(
            [
                {"miner_uid": 1, "miner_hotkey": "hotkey_1"},
                {"miner_uid": 2, "miner_hotkey": "hotkey_2"},
            ]
        )

        # Prepare expanded predictions with internal forecaster joined in
        events_predictions, _ = ClusterSelector.prepare_events_predictions(
            ranked_predictions=ranked_predictions,
            internal_forecasts=internal_forecasts,
            prediction_round_digits=2,
        )

        selector = ClusterSelector(
            ranked_predictions=ranked_predictions,
            latest_metagraph_neurons=latest_metagraph_neurons,
            internal_forecasts=internal_forecasts,
            random_seed=123,
        )

        # Cluster miners
        clusters_info, clusters_data, ifc_cluster = selector.cluster_miners(
            events_predictions=events_predictions
        )
        # Expect a single cluster id (=1) for all three miners
        expected_clusters_info = pd.DataFrame(
            [
                {
                    "miner_key": "hotkey_1__1",
                    "cluster_id": 1,
                    "miner_uid": 1,
                    "miner_hotkey": "hotkey_1",
                    "miner_count": 3,
                },
                {
                    "miner_key": "hotkey_2__2",
                    "cluster_id": 1,
                    "miner_uid": 2,
                    "miner_hotkey": "hotkey_2",
                    "miner_count": 3,
                },
                {
                    "miner_key": "internal_forecaster__1000",
                    "cluster_id": 1,
                    "miner_uid": 1000,
                    "miner_hotkey": "internal_forecaster",
                    "miner_count": 3,
                },
            ]
        )

        assert_frame_equal(clusters_info, expected_clusters_info, check_dtype=False)

        # Expected clusters_data: events_predictions joined with clusters_info on miner_key
        expected_clusters_data = pd.DataFrame(
            [
                # event 1
                {
                    "event_id": 1,
                    "event_rank": 1,
                    "outcome": 1,
                    "miner_uid_x": 1,
                    "miner_hotkey_x": "hotkey_1",
                    "prediction": 0.60,
                    "miner_key": "hotkey_1__1",
                    "outcome_num": 1,
                    "abs_error": 0.40,
                    "cluster_id": 1,
                    "miner_uid_y": 1,
                    "miner_hotkey_y": "hotkey_1",
                    "miner_count": 3,
                },
                {
                    "event_id": 1,
                    "event_rank": 1,
                    "outcome": 1,
                    "miner_uid_x": 2,
                    "miner_hotkey_x": "hotkey_2",
                    "prediction": 0.60,
                    "miner_key": "hotkey_2__2",
                    "outcome_num": 1,
                    "abs_error": 0.40,
                    "cluster_id": 1,
                    "miner_uid_y": 2,
                    "miner_hotkey_y": "hotkey_2",
                    "miner_count": 3,
                },
                {
                    "event_id": 1,
                    "event_rank": 1,
                    "outcome": 1,
                    "miner_uid_x": 1000,
                    "miner_hotkey_x": "internal_forecaster",
                    "prediction": 0.59,
                    "miner_key": "internal_forecaster__1000",
                    "outcome_num": 1,
                    "abs_error": 0.41,
                    "cluster_id": 1,
                    "miner_uid_y": 1000,
                    "miner_hotkey_y": "internal_forecaster",
                    "miner_count": 3,
                },
                # event 2
                {
                    "event_id": 2,
                    "event_rank": 2,
                    "outcome": 0,
                    "miner_uid_x": 1,
                    "miner_hotkey_x": "hotkey_1",
                    "prediction": 0.60,
                    "miner_key": "hotkey_1__1",
                    "outcome_num": 0,
                    "abs_error": 0.60,
                    "cluster_id": 1,
                    "miner_uid_y": 1,
                    "miner_hotkey_y": "hotkey_1",
                    "miner_count": 3,
                },
                {
                    "event_id": 2,
                    "event_rank": 2,
                    "outcome": 0,
                    "miner_uid_x": 2,
                    "miner_hotkey_x": "hotkey_2",
                    "prediction": 0.60,
                    "miner_key": "hotkey_2__2",
                    "outcome_num": 0,
                    "abs_error": 0.60,
                    "cluster_id": 1,
                    "miner_uid_y": 2,
                    "miner_hotkey_y": "hotkey_2",
                    "miner_count": 3,
                },
                {
                    "event_id": 2,
                    "event_rank": 2,
                    "outcome": 0,
                    "miner_uid_x": 1000,
                    "miner_hotkey_x": "internal_forecaster",
                    "prediction": 0.59,
                    "miner_key": "internal_forecaster__1000",
                    "outcome_num": 0,
                    "abs_error": 0.59,
                    "cluster_id": 1,
                    "miner_uid_y": 1000,
                    "miner_hotkey_y": "internal_forecaster",
                    "miner_count": 3,
                },
            ]
        )

        assert_frame_equal(clusters_data, expected_clusters_data, check_dtype=False)

        # Assert the internal forecaster
        assert ifc_cluster == 1

        expected_ifc_info = pd.DataFrame(
            [
                {
                    "miner_key": "internal_forecaster__1000",
                    "cluster_id": ifc_cluster,
                    "miner_uid": 1000,
                    "miner_hotkey": "internal_forecaster",
                    "miner_count": 3,
                }
            ]
        )
        actual_ifc_info = clusters_info[
            clusters_info["miner_key"] == "internal_forecaster__1000"
        ].reset_index(drop=True)

        assert_frame_equal(actual_ifc_info, expected_ifc_info, check_dtype=False)
