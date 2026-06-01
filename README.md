# Truck-Stop Duration Modeling (JSON-First)

This project trains a CatBoost regression model to predict **truck-stop duration in minutes** from raw truck-stop actuals data.

The pipeline has been revised from a flat-CSV workflow to a **JSON-first** workflow because the upstream data originates from the truck-stop actuals endpoint.

## What changed
The old prototype assumed a fixed CSV schema. The new version:

- reads **raw JSON**, **JSONL**, **CSV**, or a **live API URL**,
- flattens nested JSON records,
- canonicalizes multiple possible source field names into one stable training schema,
- computes `duration_minutes` from timestamps when possible,
- and separates **planning-safe**, **operational**, and **diagnostic-leaky** feature sets.

## Main training script

- `train_duration_from_json_catboost.py`

## Supported inputs

### 1. Local JSON export
```bash
python train_duration_from_json_catboost.py \
  --input_path truckstops.json \
  --mode operational \
  --output_dir duration_model_outputs
```

### 2. Local CSV fallback
```bash
python train_duration_from_json_catboost.py \
  --input_path truckstops.csv \
  --mode operational \
  --output_dir duration_model_outputs
```

### 3. Live API URL
```bash
python train_duration_from_json_catboost.py \
  --api_url https://centralserver.vegetationassurance.com/actuals/truckstops/ \
  --mode operational \
  --output_dir duration_model_outputs
```

If the endpoint is paginated with `next`, the script will follow pagination automatically.

## Modes

### `planning_strict`
Use this when you want the cleanest approximation of a future planning model.

Included features are limited to:
- vehicle
- location
- calendar/time features
- prior-stop context
- route-progress proxies

Example:
```bash
python train_duration_from_json_catboost.py \
  --input_path truckstops.json \
  --mode planning_strict \
  --output_dir duration_model_outputs
```

### `operational`
Recommended default for current truck-stop actuals.

Adds actuals-only summary signals such as:
- `stop_reason`
- `confidence`
- `reason_confidence`
- `low_speed_ratio`
- `stopped_signal_ratio`
- `max_speed_mph`

Example:
```bash
python train_duration_from_json_catboost.py \
  --input_path truckstops.json \
  --mode operational \
  --output_dir duration_model_outputs
```

### `diagnostic_leaky`
For QA and upper-bound benchmarking only.

Adds duration-adjacent count features such as:
- `point_count`
- `low_speed_count`
- `stopped_signal_count`

These often inflate performance and should **not** be treated as deployable planning features.

Example:
```bash
python train_duration_from_json_catboost.py \
  --input_path truckstops.json \
  --mode diagnostic_leaky \
  --output_dir duration_model_outputs
```

## Optional flags

### Keep lunch / traffic stops
By default, the script keeps only `work` and `recurring_site` rows.

To keep all stop reasons:
```bash
python train_duration_from_json_catboost.py \
  --input_path truckstops.json \
  --mode operational \
  --include_non_productive \
  --output_dir duration_model_outputs
```

### Tune train/validation split
```bash
python train_duration_from_json_catboost.py \
  --input_path truckstops.json \
  --mode operational \
  --train_frac 0.75 \
  --val_frac 0.10 \
  --output_dir duration_model_outputs
```

### Tune CatBoost parameters
```bash
python train_duration_from_json_catboost.py \
  --input_path truckstops.json \
  --mode operational \
  --iterations 300 \
  --learning_rate 0.03 \
  --depth 6 \
  --output_dir duration_model_outputs
```

## Output files
Each run writes the following to `--output_dir`:

- `canonicalized_truckstops.csv` — normalized flat table used for modeling
- `raw_columns.json` — raw source column inventory from the JSON/CSV input
- `catboost_duration_<mode>.cbm` — trained CatBoost model
- `metrics_<mode>.json` — validation/test metrics and metadata
- `feature_importance_<mode>.csv` — feature importance ranking
- `test_predictions_<mode>.csv` — holdout predictions and absolute errors

## Canonical field handling
The script tries to map multiple possible aliases to a stable set of fields, including:
- `vehicle`
- `start_timestamp`
- `end_timestamp`
- `duration_minutes` / `duration_time`
- `lat` / `lon`
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

This makes the training workflow more resilient to JSON schema drift.

## Dependencies
Install the required packages first:

```bash
pip install pandas numpy scikit-learn catboost
```

## Notes on leakage
The JSON payload can contain fields that are effectively measured during or after the stop, rather than before the stop. Those are useful for retrospective actuals modeling but not for forward planning.

### Safe to treat as diagnostic only
- `point_count`
- `low_speed_count`
- `stopped_signal_count`

### Better suited to operational retrospective mode
- `stop_reason`
- `confidence`
- `reason_confidence`
- `low_speed_ratio`
- `stopped_signal_ratio`

## Project positioning
This model is best treated as an **actuals-layer duration model** built from truck-stop JSON.

It is **not yet** the full vegetation productivity model, because the actuals endpoint does not include the main work-content drivers such as:
- LiDAR encroachment quantity
- terrain / roughness
- weather
- crew size / crew type
- fencing or access constraints
- time since scan
- species information

Those should be added in a later integration step if the goal is true forward estimation of vegetation-cutting effort.
