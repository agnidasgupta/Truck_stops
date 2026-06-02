# Revised Work Plan: JSON-First Truck-Stop Duration Modeling

## Why the plan changed
The earlier prototype was designed around a flat CSV export. The revised plan assumes the truck-stop actuals data originates as raw JSON from the `actuals/truckstops` endpoint and may arrive as:

- a paginated API response,
- a top-level JSON list,
- a JSON object with `results` / `data` / `items`, or
- a locally saved JSON export produced by a separate extraction script.

That changes the pipeline in three important ways:

1. **Schema normalization becomes a first-class step.** The model should not depend on one exact set of flat column names.
2. **The target should be computed from timestamps when possible.** `duration_minutes` or `duration_time` in the payload may be missing, renamed, or encoded differently.
3. **The training pipeline should separate planning-safe features from actuals-only operational summaries.**

## Revised modeling objective
Predict `duration_minutes` for each truck stop from normalized truck-stop actuals.

Primary target rule:
- If both `start_timestamp` and `end_timestamp` are present, compute `duration_minutes` from the timestamp difference.
- If timestamps are incomplete, fall back to the payload duration field after parsing it into minutes.

This makes the label resilient to schema drift and avoids coupling the model to one specific JSON export format.

## Revised data flow

### 1. Ingest raw JSON
Input can come from either:
- a saved JSON file,
- JSONL / NDJSON,
- or the live API URL.

The loader should support common paginated response patterns such as:
- `results`
- `data`
- `items`
- `records`
- `truckstops`

### 2. Flatten nested records
Use `pandas.json_normalize(..., sep='.')` so nested structures can still be mapped to a stable canonical schema.

### 3. Canonicalize source fields
Map the raw JSON into a stable modeling table with these canonical fields when available:
- `id`
- `vehicle`
- `start_timestamp`
- `end_timestamp`
- `duration_minutes`
- `lat`
- `lon`
- `max_radius_feet`
- `point_count`
- `low_speed_count`
- `low_speed_ratio`
- `stopped_signal_count`
- `stopped_signal_ratio`
- `max_speed_mph`
- `confidence`
- `stop_reason`
- `reason_confidence`

The canonicalization layer should tolerate common alias patterns such as camelCase, snake_case, nested metric paths, and coordinate arrays.

### 4. Engineer robust temporal and spatial features
Derived features should be created after canonicalization, not before:
- local calendar features from `start_timestamp`
- cyclic hour features
- rounded geographic cell features
- prior-stop gap and distance by vehicle
- prior duration rolling means
- within-day stop index features (including `% 4` / `% 5` logistics proxies)

### 5. Separate feature tiers
Use three explicit model modes.

#### A. `planning_strict`
Features that are closest to what is knowable before or at dispatch time:
- vehicle
- location
- calendar/time features
- prior-stop context
- route-progress proxies

This is the best proxy for a future planning model.

#### B. `operational`
Adds actuals-only summary signals that are present in the truck-stop JSON and useful for retrospective modeling:
- `stop_reason`
- `confidence`
- `reason_confidence`
- `low_speed_ratio`
- `stopped_signal_ratio`
- `max_speed_mph`

This is the recommended default when training directly on truck-stop actuals.

#### C. `diagnostic_leaky`
Adds duration-adjacent counts for QA and upper-bound benchmarking only:
- `point_count`
- `low_speed_count`
- `stopped_signal_count`

These features are often very predictive but are not appropriate for a deployable planning model because they partially encode the duration after the fact.

## Revised leakage policy
The revised plan is stricter about leakage because JSON payloads often include both raw counters and derived summaries.

### Always exclude from training
- `id`
- `end_timestamp`
- raw timestamp-derived identifiers

### Exclude from deployable planning models
- raw within-stop counts such as `point_count`, `low_speed_count`, `stopped_signal_count`

### Allow in retrospective operational mode
- `stop_reason`
- `confidence`
- `reason_confidence`
- ratio summaries such as `low_speed_ratio` and `stopped_signal_ratio`

## Revised algorithm choice
Use **CatBoostRegressor** as the main model because the revised JSON-first pipeline naturally mixes:
- numeric telemetry,
- categorical labels,
- missing values,
- and nonlinear interactions.

CatBoost is a good fit because it handles categorical features natively and is robust to heterogeneous tabular data with missingness.

## Revised validation strategy
Keep a chronological split so the model is evaluated on later stops rather than a random mix.

Recommended default split:
- 70% train
- 15% validation
- 15% test

This mirrors the real forecasting scenario more closely than a random split.

## Revised outputs
Each run should save:
- canonicalized flat truck-stop table
- raw JSON column inventory
- trained CatBoost model
- metrics JSON
- feature importance CSV
- test-set prediction CSV

## Revised results:
- Held out test:
-     "mae": 5.501713931221768,
-     "rmse": 10.708553054204147,
-     "r2": 0.7887436746599101
- Target (duration_minutes) Stats:
-     Minimum: 5.00 minutes
-     Maximum: 336.12 minutes
-     Range:   331.12 minutes
-     Count:   2,910 
-     Mean:    27.62 minutes
-     Median:  17.42 minutes

## What this revised plan still does not solve
The truck-stop actuals JSON is still only part of the full vegetation-cutting business problem. It does not directly include the main physical productivity drivers discussed earlier, such as:
- encroachment quantity,
- terrain roughness,
- weather,
- crew size/type,
- fence or drop-zone constraints,
- time since LiDAR scan,
- species information.

So the revised JSON-first trainer should be treated as the **actuals-layer model**, not yet the full vegetation productivity estimator.
