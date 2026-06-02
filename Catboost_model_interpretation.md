# No-Location Duration Model Results

## Change made

The training and prediction code were rewritten to remove all raw and derived latitude/longitude variables from the model feature set.

Removed variables:

- `lat`
- `lon`
- `geo_cell_3dp`
- `prev_stop_distance_miles`
- `prior_visits_same_geo_cell`
- `prior_visits_same_geo_cell_vehicle`

No other model factors were intentionally changed. The script still supports the same input modes as before:

- live API URL
- local JSON / JSONL / NDJSON
- local CSV

## Training data used for this run

The model was trained in `operational` mode on the saved local truck-stops dataset available in the workspace.

After filtering to productive stop reasons (`work` and `recurring_site`), the training pipeline used 2,472 rows.

## Evaluation metrics

```json
{
  "source_type": "input_path",
  "row_count_after_filtering": 2472,
  "validation": {
    "mae": 4.718782012415744,
    "rmse": 8.405820721806018,
    "r2": 0.8646753841908309
  },
  "test": {
    "mae": 5.300281280933852,
    "rmse": 10.2997400406997,
    "r2": 0.7953595078415034
  },
  "test_baseline_median": {
    "mae": 14.000494159928126,
    "rmse": 23.21554145243267,
    "r2": -0.03967279214960229
  },
  "test_baseline_mean": {
    "mae": 18.27222291468665,
    "rmse": 23.66673336502304,
    "r2": -0.08047738299074059
  }
}
```

## Interpretation

Dropping all raw and derived lat/lon variables did not materially hurt model quality on the saved dataset. The no-location model achieved a test MAE of about 5.30 minutes and a test R² of about 0.795, which is slightly better than the previously reported location-enabled API run with test MAE about 5.50 minutes and test R² about 0.789.

This suggests that the model's useful predictive signal is primarily coming from operational and temporal features such as confidence, stopped-signal ratio, low-speed ratio, max speed, stop sequence, and time-of-day features rather than actual geographic coordinates.

The median baseline remains much worse than the trained model, with test MAE about 14.00 minutes compared with 5.30 minutes for the no-location CatBoost model. This means the model remains far better than a constant-duration estimator even without location information.

## Top no-location features in this run

The top feature importances were:

1. `confidence`
2. `stopped_signal_ratio`
3. `low_speed_ratio`
4. `max_speed_mph`
5. `hour_sin`
6. `hour_cos`
7. `gap_since_prev_stop_min`
8. `stop_index_mod4`
9. `stop_index_mod5`
10. `stop_index_vehicle_day`

## Recommended conclusion

Use the no-location model going forward if raw coordinates should not influence predictions. It preserves the same input flexibility and model workflow while avoiding geographic leakage or privacy concerns.
