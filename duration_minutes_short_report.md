# Short report: predicting `duration_minutes` from existing truck-stop data

## What was modeled

The truckstop CSV contains **2,910 truck-stop records** and **17 raw columns**.  
For the vegetation-cutting use case, `duration_minutes` was treated as the regression target and filtered the data to the two **productive stop types**:

- `work`
- `recurring_site`

This leaves **2,472 stops** for modeling. `traffic` and `lunch` were excluded because those stops do not represent vegetation-cutting productivity.

## Important limitation

This CSV **does not yet contain** the main causal drivers described in the earlier project brief, such as:

- measured encroachment volume/length
- weather
- terrain steepness / roughness
- fencing / access constraints
- crew size / crew type
- tree species
- time since scan

Hence, the code below is **not yet the final vegetation-cutting planning model**. It is a strong **stop-duration prediction model from the currently available telemetry and metadata**.

## Algorithm choice

**CatBoostRegressor** is chosen as the main model.

Why:

1. It is one of the strongest practical algorithms for mixed tabular data with both numeric and categorical fields.
2. It handles categorical features natively, so columns like `vehicle`, `stop_reason`, and derived location bins do not need one-hot encoding.
3. It handles missing values and non-linear interactions well.
4. For real-world tabular regression problems, modern gradient-boosted tree models remain extremely competitive, often outperforming more complicated deep tabular models unless there is a very specific reason to use a foundation model or custom neural architecture.
5. CatBoost is production-friendly and easy to retrain as new operational and environmental features arrive.

## Feature decisions

### Discarded entirely

- `id`  
  Identifier only. No predictive meaning.

- `end_timestamp`  
  Must be dropped because `end_timestamp - start_timestamp = duration_minutes`. Keeping it would directly leak the target.

### Discarded from the deployable model because they are near-direct duration proxies

- `point_count`
- `low_speed_count`
- `stopped_signal_count`

These counts are extremely close to measuring how long the stop lasted. They produce unrealistically strong accuracy, but that accuracy is not deployable for forecasting future cutting effort.

### Used

- `vehicle` as a proxy for truck / crew / operating pattern
- `stop_reason` after filtering to productive reasons
- `lat`, `lon`, `max_radius_feet`
- `max_speed_mph`
- `low_speed_ratio`, `stopped_signal_ratio` in the **operational** model
- time features derived from `start_timestamp`
- route sequence features within each vehicle-day
- lag features from the prior stop
- recurrence features for repeated locations

### Used as sample weights, not predictors

- `confidence`
- `reason_confidence`

These are better treated as row-quality weights than as causal predictors.

## Engineered features

The code creates the following new variables:

- local time-of-day seasonality: `hour_sin`, `hour_cos`
- calendar features: `dayofweek`, `month`, `weekofyear`, `dayofyear`, `is_weekend`
- coarse spatial cell: `geo_cell_3dp`
- within-day sequence: `stop_index_vehicle_day`, `stop_index_mod4`, `stop_index_mod5`
- previous-stop context:
  - `gap_since_prev_stop_min`
  - `prev_stop_distance_miles`
  - `prev_duration_minutes`
  - `rolling_prev3_duration_mean`
- recurrence:
  - `prior_visits_same_geo_cell`
  - `prior_visits_same_geo_cell_vehicle`
- elapsed-study-time feature:
  - `days_since_dataset_start`

These variables help the model learn location effects, repeated-site behavior, route friction, and periodic operational patterns.

## Evaluation design

The code uses a **chronological split**:

- first 70% of stops for training
- next 15% for validation / early stopping
- last 15% for test

This is more realistic than a random split because it evaluates how well the model generalizes forward in time.

The target is trained on `log1p(duration_minutes)` to stabilize the long right tail, then transformed back to minutes for reporting.

## Results on the given truckstop CSV

### 1. Operational model (recommended with the current CSV)
This model excludes direct count leakage but still uses within-stop ratios.

- Test MAE: **7.48 minutes**
- Test RMSE: **12.84 minutes**
- Test R²: **0.682**

### 2. Strict planning-like model
This model avoids all within-stop behavior ratios and uses only time/location/sequence history.

- Test MAE: **13.31 minutes**
- Test RMSE: **22.86 minutes**
- Test R²: **-0.008**

Interpretation: the current CSV alone does **not** contain enough true pre-stop planning signal to estimate cutting duration well.

### 3. Diagnostic leaky model (not for deployment)
This model includes `point_count`, `low_speed_count`, and `stopped_signal_count`.

- Test MAE: **2.88 minutes**
- Test RMSE: **4.97 minutes**
- Test R²: **0.952**

Interpretation: this result is **too good** because those columns are effectively measuring stop length itself. They should not be used in a forecasting model.

## Recommended deployment stance

Use the **operational CatBoost model** as the best model that can be built from the current CSV.

Use the **strict model** only as a reality check for future planning. Its weak performance shows why the next version must add the missing business variables:

- encroachment quantity
- weather
- slope / roughness
- traffic/access
- fencing
- crew attributes
- time since scan
- vegetation/species descriptors

## Top conclusion

With the current truck-stop CSV, CatBoost can predict stop duration **reasonably well** only when it uses features that describe the realized stop behavior. For true forward planning of vegetation-cutting time, the model needs the operational and environmental features from the earlier project brief—especially measured encroachment.

## Files used/created:

- `train_duration_minutes_catboost.py` — training script
- `duration_model_outputs/metrics_operational.json`
- `duration_model_outputs/metrics_planning_strict.json`
- `duration_model_outputs/metrics_diagnostic_leaky.json`
- `duration_model_outputs/feature_importance_*.csv`
- `duration_model_outputs/test_predictions_*.csv`
