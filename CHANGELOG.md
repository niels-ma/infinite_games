# Release Notes

## [1.6.7] - 2025-10-28
- **Bittensor Upgrade**: Upgraded to Bittensor version 9.12.1
- **Dependencies**: Pin async-substrate-interface to prevent reconnection loop
- **Weights**: Burn incentives

## [1.6.6] - 2025-10-22
- **Weights**: Await for inclusion and finalization when setting weights

## [1.6.5] - 2025-10-11
- **Scoring**: Ensure internal forecasts order
- **Miners Querying**: Ensure UTC timestamps in store miners

## [1.6.4] - 2025-10-01
- **Scoring**: Align events ranking across validators for Precog Index
- **Validator**: Increase the frequency of some tasks
- **Bittensor Upgrade**: Upgraded to Bittensor version 9.11.1

## [1.6.3] - 2025-09-25
- **Export Scores**: Fix export scores backwards compatibility
- **Scoring**: Improve consistency across validators for Precog Index

## [1.6.2] - 2025-09-20
- **Scoring**: Consistent random seed across validators for Precog Index

## [1.6.1] - 2025-09-16
- **Database**: Enforce SQLite version 3.40.1 as minimum
- **Bittensor Upgrade**: Upgraded to Bittensor version 9.10.1
- **Scoring**: Enable Precog Index scoring

## [1.6.0] - 2025-09-09
- **Database**: Enable WAL mode
- **Miners Querying**: Remove duplicate IP miners filter

## [1.5.9] - 2025-09-02
- **Validator**: Enforce min Python version 3.11
- **Validator**: Use async subtensor and metagraph

## [1.5.8] - 2025-08-19
- **Bittensor Upgrade**: Upgraded to Bittensor version 9.9.0

## [1.5.7] - 2025-07-29
- **API Client**: Added API client data validation
- **Bittensor Upgrade**: Upgraded to Bittensor version 9.8.3

## [1.5.6] - 2025-07-22
- **Logger**: Added logger level command options
- **Bittensor Upgrade**: Upgraded to Bittensor version 9.8.2

## [1.5.5] - 2025-07-15
- **Scoring**: Update interval scoring weights to parabolic decay
- **Bittensor Upgrade**: Upgraded to Bittensor version 9.8.0

## [1.5.4] - 2025-07-08
- **Database**: Added task to delete old soft deleted events
- **Miners Querying**: Filter out duplicate IP miners

## [1.5.3] - 2025-07-01
- **API**: Explicitly handle disconnected API request
- **Miners Querying**: Filter out duplicate cold key miners

## [1.5.2] - 2025-06-24
- **Data Exporting**: Clean up fields in export scores
- **Protocol**: Drop deprecated fields - starts, resolve_date, end_date
- **Validator Options**: Added --db.directory command option to allow to set db file directory

## [1.5.1] - 2025-06-17
- **Database**: Pre soft delete events prior to hard delete

## [1.5.0] - 2025-06-03
- **Bittensor Upgrade**: Upgraded to Bittensor version 9.7.0
- **Database**: Added task to delete old reasonings

## [1.4.9] - 2025-05-27
- **Bittensor Upgrade**: Upgraded to Bittensor version 9.6.1
- **Database**: Rebuilt the predictions table
- **Protocol**: Added metadata field in subnet synapse

## [1.4.8] - 2025-05-13
- **API**: Responses now include a `community_prediction_lr` field alongside `community_prediction`, exposing the probability from a trained community prediction LogisticRegression model.

## [1.4.7] - 2025-05-06
- **Protocol**: Added reasoning field in subnet synapse

## [1.4.6] - 2025-04-29
- **Scoring**: Improved metagraph scoring to correct for class imbalance by up-weighting less frequent YES outcomes
- **Database**: Added task to delete processed old scores
- **API**: Batched community predictions API

## [1.4.5] - 2025-04-22
- **Bittensor Upgrade**: Upgraded to Bittensor version 9.4.0

## [1.4.4] - 2025-04-07
- **Scoring**: Assign a weight of 1.0 to the first two intervals when computing the reverse exponential moving average for prediction aggregation.

## [1.4.3] - 2025-04-01
- **Bittensor Upgrade**: Upgraded to Bittensor version 9.2.0
- **Scoring**: Reduced penalty for unresponsive miners by imputing missing predictions as the value at 1/3 of the distance between the worst prediction and the mean prediction (closer to the worst) instead of using the completely wrong outcome.

## [1.4.2] - 2025-03-25
- **Scoring**: Increased moving average from 100 to 150 events score for setting the weights

## [1.4.1] - 2025-03-17
- **Bittensor Upgrade**: Upgraded to Bittensor version 9.1.0
- **Database**: Cleaned up old records in the events table
- **API**: Exposed the OpenAPI schema

## [1.4.0] - 2025-03-07
- **Bittensor Upgrade**: Upgraded to Bittensor version 9.0.4
- **Database**: Added periodic vacuum task
- **API**: Added community predictions route

## [1.3.9] - 2025-02-19
- **Bittensor Upgrade**: Upgraded to Bittensor version 9.0.1
- **Scoring**: Updated the scoring methodology by replacing the Brier scoring approach with Peer scoring.
- **Database**: Change database auto vacuum to incremental and add migration to vacuum & reclaim empty space

## [1.3.8] - 2025-02-11
- **Bittensor Upgrade**: Upgraded to Bittensor version 9.0.0
- **Scoring**: Peer Scores are computed and stored in a dedicated table while continuing to use Brier scoring.
- **Validator API**: Introduced an optional API for validators, which can be enabled by setting the appropriate environment variable.
- **Maintenance**: Improved the sanitization of scores exported to the database used for the Mainnet dashboard.
- **Miner Update**: Added a new LLM forecaster based on [forecasting-tools](https://github.com/Metaculus/forecasting-tools)

## [1.3.7] - 2025-02-03
- **Events Resolution**: Events resolution and deletion upgraded to batched requests to improve overhead and reduce resolution time.

## [1.3.6] - 2025-01-27
- **Bittensor Upgrade**: Upgraded to Bittensor version 8.5.2
- **Requirements Update**: Removed unused pip requirements and switched the torch requirement to the CPU version. This significantly reduces the size of the required Python environment.
- **Maintenance**:
    - Added a task to gradually and regularly delete old records from the local database, preventing uncontrolled database growth.
    - In a future update, the database will be vacuumed to reclaim storage from deleted records.
- **Database Enhancements**: Integrated Alembic for managing all changes to the local database schema.
- **Data Exporting**: Increased the payload batch size for data export to improve efficiency.

## [1.3.5] - 2025-01-20
- **Scoring**: After scoring data analysis and simulations, we reduced the pre-normalization exponential factor from 30 to 5. This will prevent outlier miners to get disproportionate gains for some events. Additionally, it improves the chances of new miners to catchup with the existing miners.
- **Maintenance**: Removed deprecated validator code, reorganize existing miner code, cleanup dead code.

## [1.3.4] - 2025-01-13
- **Validator Architecture**: Validator architecture re-implemented to handle scaling events.

## [1.3.3] - 2024-12-17
- **Bittensor Upgrade**: Upgraded to Bittensor version 8.5.1
- **Default `netuid` for Validators**: The `--netuid` argument now defaults to 6, ensuring validators register on the intended subnet when the argument is not explicitly provided.
- **Logging Improvements**: Enhanced logging and error reporting for better debugging and issue tracking
- **Resilient Score Exporting**: Re-exporting functionality added to ensure scores from failed or missed past intervals are included in the current interval.

## [1.3.2] - 2024-12-10
- **Bittensor Upgrade**: Upgraded to Bittensor version 7.4.0.
- **Bug Fix**: Patched a recent common `SSLEOFError` encountered during block number retrieval.
- **Query Optimization**: Miners are now queried at a maximum frequency of once every 5 minutes to reduce unnecessary overhead.
- **Database Enhancements**: Improved data insertion logic for the local database, ensuring better reliability and performance.
- **Logging Improvements**: Enhanced logging and error reporting for better debugging and issue tracking.
- **Code Appearance**: Applied formatting and organization improvements using `black` and `isort`.
