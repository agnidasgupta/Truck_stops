# Truck-Stop Duration Modeling (Live API + Local JSON)

This project trains and scores a CatBoost regression model to predict **truck-stop duration in minutes** from the truck-stops actuals payload.

Primary live source:
- `https://centralserver.vegetationassurance.com/actuals/truckstops/`

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

## Main scripts

- `train_duration_from_truckstops_api_catboost.py`
- `predict_duration_from_truckstops_api_catboost.py`

## Training

### Train from the live API
```bash
python train_duration_from_truckstops_api_catboost.py \
  --api_url https://centralserver.vegetationassurance.com/actuals/truckstops/ \
  --mode operational \
  --output_dir truckstops_api_run
```

### Train from a saved local JSON payload
```bash
python train_duration_from_truckstops_api_catboost.py \
  --input_path truckstops_payload.json \
  --mode operational \
  --output_dir truckstops_api_run
```

## SSL / certificate options
If your local Python install cannot verify the site certificate chain, try these in order.

### 1. Install certifi
```bash
pip install certifi catboost
```

### 2. Use certifi's CA bundle explicitly
```bash
python train_duration_from_truckstops_api_catboost.py \
  --api_url https://centralserver.vegetationassurance.com/actuals/truckstops/ \
  --cafile "$(python -c 'import certifi; print(certifi.where())')" \
  --mode operational \
  --output_dir truckstops_api_run
```

### 3. macOS python.org installs
Run the bundled certificate installer once:
```bash
open "/Applications/Python 3.13/Install Certificates.command"
```

### 4. Last resort for trusted internal testing only
```bash
python train_duration_from_truckstops_api_catboost.py \
  --api_url https://centralserver.vegetationassurance.com/actuals/truckstops/ \
  --insecure \
  --mode operational \
  --output_dir truckstops_api_run
```

## Prediction

### Predict from the live API using a saved model
```bash
python predict_duration_from_truckstops_api_catboost.py \
  --model_path truckstops_api_run/catboost_duration_model.cbm \
  --metadata_path truckstops_api_run/model_metadata.json \
  --api_url https://centralserver.vegetationassurance.com/actuals/truckstops/ \
  --output_dir truckstops_api_pred
```

### Predict from a saved local JSON payload
```bash
python predict_duration_from_truckstops_api_catboost.py \
  --model_path truckstops_api_run/catboost_duration_model.cbm \
  --metadata_path truckstops_api_run/model_metadata.json \
  --input_path truckstops_payload.json \
  --output_dir truckstops_api_pred
```

### Predict from the live API with an explicit CA bundle
```bash
python predict_duration_from_truckstops_api_catboost.py \
  --model_path truckstops_api_run/catboost_duration_model.cbm \
  --metadata_path truckstops_api_run/model_metadata.json \
  --api_url https://centralserver.vegetationassurance.com/actuals/truckstops/ \
  --cafile "$(python -c 'import certifi; print(certifi.where())')" \
  --output_dir truckstops_api_pred
```

Prediction evaluates metrics **only if** an explicit source target field is present (`duration_minutes` or `duration_time`).

## Modes

### `planning_strict`
Uses only the cleaner feature subset.

### `operational`
Recommended default for actuals modeling. Adds operational signals such as:
- `stop_reason`
- `confidence`
- `reason_confidence`
- `low_speed_ratio`
- `stopped_signal_ratio`
- `max_speed_mph`

### `diagnostic_leaky`
For QA / upper-bound benchmarking only. Adds:
- `point_count`
- `low_speed_count`
- `stopped_signal_count`

## Other useful flags

### Keep lunch / traffic stops
```bash
python train_duration_from_truckstops_api_catboost.py \
  --api_url https://centralserver.vegetationassurance.com/actuals/truckstops/ \
  --include_non_productive \
  --mode operational \
  --output_dir truckstops_api_run
```

### Change split ratios
```bash
python train_duration_from_truckstops_api_catboost.py \
  --input_path truckstops_payload.json \
  --train_frac 0.75 \
  --val_frac 0.10 \
  --mode operational \
  --output_dir truckstops_api_run
```

### Tune CatBoost parameters
```bash
python train_duration_from_truckstops_api_catboost.py \
  --input_path truckstops_payload.json \
  --iterations 300 \
  --learning_rate 0.03 \
  --depth 6 \
  --mode operational \
  --output_dir truckstops_api_run
```

## Outputs

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

## Dependencies
```bash
pip install pandas numpy scikit-learn catboost certifi
```

## Notes
- The code forces categorical columns explicitly before CatBoost training/prediction.
- Numeric columns are coerced explicitly with `pd.to_numeric(errors="coerce")`.
- The live fetch path now supports `--cafile` and `--insecure` to handle local TLS trust-store problems.
- The model is still an **actuals-layer** duration model, not yet the full vegetation productivity model with LiDAR encroachment, weather, terrain, crew, fencing, and time-since-scan features.
