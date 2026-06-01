# Truck-Stop Duration Modeling from Live API JSON

This repository now uses the **live truck-stops JSON endpoint** as the primary source instead of a CSV snapshot.

Endpoint:

```text
https://centralserver.vegetationassurance.com/actuals/truckstops/
```

Expected payload shape:

```json
{
  "success": true,
  "source": "stored",
  "records": [
    {
      "id": "2240079-2026-01-05T12:20:46Z-2026-01-05T13:02:57Z",
      "vehicle": 2240079,
      "start_timestamp": "2026-01-05T12:20:46Z",
      "end_timestamp": "2026-01-05T13:02:57Z",
      "duration_minutes": 42.18,
      "lat": 41.8332266,
      "lon": -86.3548629,
      "max_radius_feet": 0.0,
      "point_count": 37,
      "low_speed_count": 36,
      "low_speed_ratio": 0.973,
      "stopped_signal_count": 0,
      "stopped_signal_ratio": 0.0,
      "max_speed_mph": 5.1,
      "confidence": 0.9,
      "stop_reason": "recurring_site",
      "reason_confidence": 0.8
    }
  ]
}
```

## Files

- `train_duration_from_truckstops_api_catboost.py`
  - fetches the live JSON payload or reads the same payload from a local `.json`
  - engineers features
  - trains a CatBoost regressor on `log1p(duration_minutes)`
  - saves model, metrics, metadata, feature importance, and test predictions

- `predict_duration_from_truckstops_api_catboost.py`
  - loads the saved `.cbm` model and metadata
  - scores the live API JSON or a local JSON payload
  - computes evaluation metrics **only if** `duration_minutes` is present

## Feature modes

### `planning_strict`
Uses only safer, planning-style features derived from time, location, route position, and lagged history.

### `operational`
Adds operational features available in the truck-stop records:
- `stop_reason`
- `confidence`
- `reason_confidence`
- `low_speed_ratio`
- `stopped_signal_ratio`
- `max_speed_mph`

### `diagnostic_leaky`
Adds strong but leakage-prone counts:
- `point_count`
- `low_speed_count`
- `stopped_signal_count`

Use `operational` as the default. Use `diagnostic_leaky` only to understand the upper bound from within-stop signals.

## Install

```bash
pip install pandas numpy scikit-learn catboost
```

## Train from the live API

```bash
python train_duration_from_truckstops_api_catboost.py \
  --api_url https://centralserver.vegetationassurance.com/actuals/truckstops/ \
  --mode operational \
  --output_dir truckstops_api_run
```

## Train from a local JSON payload with the same shape

```bash
python train_duration_from_truckstops_api_catboost.py \
  --input_path truckstops_payload.json \
  --mode operational \
  --output_dir truckstops_api_run
```

## Predict with the saved model on the live API

```bash
python predict_duration_from_truckstops_api_catboost.py \
  --model_path truckstops_api_run/catboost_duration_model.cbm \
  --metadata_path truckstops_api_run/model_metadata.json \
  --api_url https://centralserver.vegetationassurance.com/actuals/truckstops/ \
  --output_dir truckstops_api_pred
```

## Predict with the saved model on a local JSON payload

```bash
python predict_duration_from_truckstops_api_catboost.py \
  --model_path truckstops_api_run/catboost_duration_model.cbm \
  --metadata_path truckstops_api_run/model_metadata.json \
  --input_path truckstops_payload.json \
  --output_dir truckstops_api_pred
```

## Artifacts written by training

Training writes:
- `catboost_duration_model.cbm`
- `model_metadata.json`
- `metrics.json`
- `raw_columns.json`
- `feature_importance.csv`
- `test_predictions.csv`

Prediction writes:
- `predictions.csv`
- `prediction_metrics.json`

## Notes

- The trainer recomputes `duration_minutes` from `start_timestamp` and `end_timestamp` when possible.
- The predictor skips evaluation when the source data does not contain a usable `duration_minutes` label.
- Categorical features are passed to CatBoost by **column index** after explicit string normalization to avoid the earlier `Cannot convert 'work' to float` error.
