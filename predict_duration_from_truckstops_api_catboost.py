#!/usr/bin/env python3
from __future__ import annotations

"""
Predict truck-stop duration using a saved CatBoost model trained from the truck-stops API.

Evaluation metrics are reported only if an explicit duration label field is present in the
source payload (`duration_minutes` or `duration_time`). If the source lacks that label,
predictions are still written but evaluation is skipped.

Example Usage:
python predict_duration_from_truckstops_api_catboost.py \
  --model_path truckstops_api_run/catboost_duration_model.cbm \
  --metadata_path truckstops_api_run/model_metadata.json \
  --api_url https://centralserver.vegetationassurance.com/actuals/truckstops/ \
  --output_dir truckstops_api_pred

python predict_duration_from_truckstops_api_catboost.py \
  --model_path truckstops_api_run/catboost_duration_model.cbm \
  --metadata_path truckstops_api_run/model_metadata.json \
  --input_path truckstops_payload.json \
  --output_dir truckstops_api_pred
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_duration_from_truckstops_api_catboost as trainer


TARGET_ALIASES = {"duration_minutes", "duration_time"}


def evaluate(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_prediction_matrix(df: pd.DataFrame, model_features: Sequence[str], categorical_features: Sequence[str]) -> tuple[pd.DataFrame, list[int]]:
    X = pd.DataFrame(index=df.index)
    for col in model_features:
        if col in df.columns:
            X[col] = df[col]
        elif col in categorical_features:
            X[col] = "__missing__"
        else:
            X[col] = np.nan

    categorical_features = [c for c in categorical_features if c in X.columns]
    numeric_features = [c for c in X.columns if c not in categorical_features]

    for col in categorical_features:
        X[col] = X[col].astype("string").fillna("__missing__")
        X[col] = X[col].replace({"<NA>": "__missing__", "nan": "__missing__", "None": "__missing__"})
        X[col] = X[col].astype(str)

    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)

    cat_idx = [X.columns.get_loc(c) for c in categorical_features]
    return X, cat_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Score truck-stop duration from live API JSON or a local JSON payload.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--api_url", type=str, default=None)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="prediction_outputs")
    parser.add_argument("--include_non_productive", action="store_true")
    args = parser.parse_args()

    if args.api_url is None and args.input_path is None:
        args.api_url = trainer.DEFAULT_API_URL

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(Path(args.metadata_path).read_text())
    model = CatBoostRegressor()
    model.load_model(args.model_path)

    raw, source_meta = trainer.load_raw_dataframe(input_path=args.input_path, api_url=args.api_url)
    raw_has_explicit_target = any(col in raw.columns for col in TARGET_ALIASES)

    canon = trainer.canonicalize(raw, require_target=False)
    df = trainer.engineer_features(
        canon,
        productive_only=(False if args.include_non_productive else bool(metadata.get("productive_only", True))),
        timezone=str(metadata.get("timezone", trainer.DEFAULT_TIMEZONE)),
    )

    model_features = list(metadata["features_used"])
    categorical_features = list(metadata["categorical_features"])
    X, cat_idx = make_prediction_matrix(df, model_features, categorical_features)
    pred = np.clip(np.expm1(model.predict(Pool(X, cat_features=cat_idx))), 0.0, None)

    output_cols = [c for c in ["id", "vehicle", "start_timestamp", "end_timestamp", "stop_reason", "lat", "lon"] if c in df.columns]
    output = df[output_cols].copy()
    output["pred_duration_minutes"] = pred

    result: dict[str, Any] = {
        **source_meta,
        "row_count_after_filtering": int(len(df)),
        "features_used": model_features,
        "categorical_features": categorical_features,
        "productive_only": bool(metadata.get("productive_only", True)) and (not args.include_non_productive),
        "timezone": str(metadata.get("timezone", trainer.DEFAULT_TIMEZONE)),
        "evaluation_performed": False,
        "explicit_target_present_in_source": bool(raw_has_explicit_target),
    }

    if raw_has_explicit_target:
        y_true = pd.to_numeric(df[trainer.TARGET_NAME], errors="coerce")
        eval_mask = y_true.notna() & np.isfinite(y_true)
        if bool(eval_mask.any()):
            y_eval = y_true.loc[eval_mask].to_numpy(dtype=float)
            pred_eval = output.loc[eval_mask, "pred_duration_minutes"].to_numpy(dtype=float)
            output.loc[eval_mask, "actual_duration_minutes"] = y_eval
            output.loc[eval_mask, "abs_error_minutes"] = np.abs(y_eval - pred_eval)
            result["evaluation_performed"] = True
            result["evaluation_rows"] = int(eval_mask.sum())
            result["evaluation"] = evaluate(y_eval, pred_eval)

    output.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "prediction_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
