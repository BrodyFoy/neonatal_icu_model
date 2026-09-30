"""Script file to fit ML models for predicting ventilatory outcomes in 
a neonatal ICU dataset. 

This code describes the primary analysis for the paper: 
Computer Vision–Enabled Early Prediction of Neonatal Respiratory Escalation 
in Resource-Limited Settings: An Indian Multi-Center Cohort Study.

Primary code was written by Amrit Sharma, supervised by Prof Brody H Foy.

Contact: Brody H Foy, DPhil. brodyfoy@uw.edu

The core file uses a neonatal ICU dataset (cannot be shared due to PHI restrictions) 
and fits one of three model types (random_forest, decision_tree, logistic_regression) 
to predict outcomes (mortality, escalation of ventilatory support, future requirement of 
intubation or non-invasive ventilation [NIV]) at one of three timepoints (6h, 12h or 
24h post birth), using demographics and vitals measurements.

Example use
--------
python NeonateICUSep26_primary_analysis.py \
    --timepoint 6 --outcome mortality --model random_forest

Notes
----------------------------------------------------
* Transfers to another ICU or a higher centre of care are excluded, due to indeterminate
outcomes.
* Discharge against medical advice (DAMA) is not excluded. DAMA is a
  non-death for the mortality target, while respiratory outcomes use the
  patient's observed subsequent ventilation records up to time of dicharge.
* Escalation excludes patients already invasively ventilated at time of prediction
(since by definition they cannot escalate). Similarly, NIV/intubation outcome 
excludes patients already receiving
  NIV or invasive ventilation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


DEFAULT_SOURCE = Path("source_data.csv")
RANDOM_STATE = 42

# Ventilatory support used many text terms, so we map them to four categories:
ROOM_AIR = {"Room air", "RA"}
CANNULA = {"Nasal cannula", "Nasal Bubble CPAP", "HFNC", "High Flow", "Cannula"}
NIV = {"NIV", "Face mask", "Venturi mask", "Ambu bag", "Oronasal Mask (NIV)",
    "Oxygen hood", "Oral Mask (NIV)", "Trach mask", "Non-rebreathing mask",
    "Mask", "Simple Mask", "NRB", "Venturi",
}
INTUBATED = {"Invasive ventilation", "T piece", "ETT", "ETT+"}

# Define levels of escalation
VENT_RANK = {value: 1 for value in ROOM_AIR}
VENT_RANK.update({value: 2 for value in CANNULA})
VENT_RANK.update({value: 3 for value in NIV})
VENT_RANK.update({value: 4 for value in INTUBATED})

# Define 'discharged to ICU or higher center of care' (to exclude)
TRANSFER_DISPOSITIONS = {
    "Discharged to other ICU within the same hospital",
    "Discharged to higher centre of care",
}

# Which dataset columns to use
USE_COLUMNS = [
    "cpmrn",
    "timestamp",
    "daysVentAirway",
    "daysHR",
    "daysSpO2",
    "daysRR",
    "daysFiO2",
    "weight",
    "gestation_age",
    "apache_score",
    "rox_score",
    "crib2_score",
    "actual_disposition",
]
# Which features to input into the model (some of these need to first be constructed)
FEATURES = [
    "HR",
    "SPO2",
    "RR",
    "FiO2",
    "Weight",
    "Gestation age",
    "APACHE score",
    "ROX score",
    "crib2_score",
    "Highest past HR",
    "Highest past RR",
    "Lowest past SPO2",
    "Highest past FiO2",
    "Current RA Binary",
    "Current Cannula Binary",
    "Current NIV Binary",
    "Current Intubated Binary",
    "Past RA Binary",
    "Past Cannula Binary",
    "Past NIV Binary",
    "Past Intubated Binary",
]

OUTCOMES = {
    "mortality": "Mortality Binary",
    "escalation": "Worse than current",
    "niv_intubation": "Reaches NIV/Intubated",
}

MODEL_NAMES = {
    "random_forest": "Random Forest",
    "decision_tree": "Decision Tree",
    "logistic_regression": "Logistic Regression",
}

RF_PARAMETERS = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False],
}

DT_PARAMETERS = {
    "max_depth": [None, 5, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": [None, "sqrt", "log2"],
    "criterion": ["gini", "entropy", "log_loss"],
    "splitter": ["best", "random"],
}

LR_PARAMETERS = {
    "classifier__C": loguniform(1e-3, 1e3),
    "classifier__penalty": ["l1", "l2"],
}


# Read data in
def read_source(path: Path) -> pd.DataFrame:
    """Read only the columns required for the primary analysis."""
    df = pd.read_csv(path, usecols=USE_COLUMNS, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    missing_id_or_time = df["cpmrn"].isna() | df["timestamp"].isna()
    if missing_id_or_time.any():
        df = df.loc[~missing_id_or_time].copy()

    df["vent_rank"] = df["daysVentAirway"].map(VENT_RANK)
    if df["vent_rank"].isna().any():
        unmapped = sorted(
            df.loc[df["vent_rank"].isna(), "daysVentAirway"]
            .dropna()
            .astype(str)
            .unique()
        )
        raise ValueError(f"Unmapped ventilation labels: {unmapped}")

    # Stable ordering fixes the notebook's list(set(...)) reproducibility issue.
    return df.sort_values(["cpmrn", "timestamp"], kind="stable").reset_index(drop=True)


def build_interval_dataset(df: pd.DataFrame, timepoint: int) -> pd.DataFrame:
    """Create the patient-level dataset at the selected ordered record."""
    counts = df.groupby("cpmrn", sort=False).size()
    eligible_ids = counts[counts >= timepoint].index
    work = df[df["cpmrn"].isin(eligible_ids)].copy()
    work["row_number"] = work.groupby("cpmrn", sort=False).cumcount()

    current = (
        work.loc[work["row_number"] == timepoint - 1]
        .copy()
        .set_index("cpmrn")
    )
    # Construct temporal variables (i.e., from heart rate to worst prior heart rate)
    through_timepoint = work.loc[work["row_number"] < timepoint]
    history = through_timepoint.groupby("cpmrn", sort=False).agg(
        **{
            "Highest past HR": ("daysHR", "max"),
            "Highest past RR": ("daysRR", "max"),
            "Lowest past SPO2": ("daysSpO2", "min"),
            "Highest past FiO2": ("daysFiO2", "max"),
            "past_rank": ("vent_rank", "max"),
        }
    )

    current_timestamp = current["timestamp"]
    patient_cutoff = work["cpmrn"].map(current_timestamp)
    future = work.loc[work["timestamp"] > patient_cutoff]
    future_rank = (
        future.groupby("cpmrn", sort=False)["vent_rank"]
        .max()
        .reindex(current.index)
        .fillna(1)
    )

    interval = pd.DataFrame(index=current.index)
    interval["HR"] = current["daysHR"]
    interval["SPO2"] = current["daysSpO2"]
    interval["RR"] = current["daysRR"]
    interval["FiO2"] = current["daysFiO2"]
    interval["Weight"] = current["weight"]
    interval["Gestation age"] = current["gestation_age"]
    interval["APACHE score"] = current["apache_score"]
    interval["ROX score"] = current["rox_score"]
    interval["crib2_score"] = current["crib2_score"]
    interval = interval.join(history)
    interval["current_rank"] = current["vent_rank"]
    interval["future_rank"] = future_rank
    interval["Discharge Outcome"] = current["actual_disposition"]

    for label, rank in (("RA", 1), ("Cannula", 2), ("NIV", 3), ("Intubated", 4)):
        interval[f"Current {label} Binary"] = (interval["current_rank"] == rank).astype(int)
        interval[f"Past {label} Binary"] = (interval["past_rank"] == rank).astype(int)

    interval["Mortality Binary"] = (
        interval["Discharge Outcome"].eq("Death").astype(int)
    )
    interval["Worse than current"] = (
        interval["future_rank"] > interval["current_rank"]
    ).astype(int)
    interval["Reaches NIV/Intubated"] = (
        (interval["future_rank"] >= 3) & (interval["current_rank"] < 3)
    ).astype(int)

    # This is the notebook's explicit discharge exclusion. DAMA remains.
    interval = interval.loc[
        ~interval["Discharge Outcome"].isin(TRANSFER_DISPOSITIONS)
    ].copy()
    return interval.reset_index()


def restrict_to_outcome_cohort(
    interval: pd.DataFrame, outcome_key: str
) -> tuple[pd.DataFrame, str]:
    """Apply the outcome-specific clinical eligibility rules."""
    target = OUTCOMES[outcome_key]
    cohort = interval.copy()

    if outcome_key == "escalation":
        cohort = cohort.loc[cohort["Current Intubated Binary"] == 0].copy()
    elif outcome_key == "niv_intubation":
        cohort = cohort.loc[
            (cohort["Current NIV Binary"] == 0)
            & (cohort["Current Intubated Binary"] == 0)
        ].copy()

    return cohort, target


def model_and_parameters(model_key: str) -> tuple[Any, dict[str, Any]]:
    """Return the model and its randomized-search space."""
    if model_key == "random_forest":
        return (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            RF_PARAMETERS,
        )
    if model_key == "decision_tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE), DT_PARAMETERS
    if model_key == "logistic_regression":
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        solver="liblinear",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        return pipeline, LR_PARAMETERS
    raise ValueError(f"Unknown model: {model_key}")


def train_and_evaluate(
    interval: pd.DataFrame, outcome_key: str, model_key: str
) -> dict[str, Any]:
    """Tune on 10% (of training) of the cohort (validation), fit on 70% (train), and evaluate on 30% (test)."""
    cohort, target = restrict_to_outcome_cohort(interval, outcome_key)
    X = cohort[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Preserved from the notebook: means are calculated before train/test split.
    X = X.fillna(X.mean())
    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        raise ValueError(f"Features contain no usable values: {all_missing}")

    y = cohort[target].astype(int)
    if y.nunique() < 2:
        raise ValueError(f"Outcome {target!r} has fewer than two classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    tune_fraction_of_train = 0.1 / 0.7
    X_tune, _, y_tune, _ = train_test_split(
        X_train,
        y_train,
        test_size=1 - tune_fraction_of_train,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    estimator, parameter_space = model_and_parameters(model_key)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameter_space,
        n_iter=30,
        scoring="roc_auc",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_tune, y_tune)

    final_model = clone(search.best_estimator_)
    final_model.fit(X_train, y_train)
    probability = final_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probability)

    return {
        "model": MODEL_NAMES[model_key],
        "outcome": outcome_key,
        "target_column": target,
        "analysis_n": int(len(cohort)),
        "analysis_events": int(y.sum()),
        "train_n": int(len(y_train)),
        "test_n": int(len(y_test)),
        "test_events": int(y_test.sum()),
        "roc_auc": float(auc),
        "best_parameters": search.best_params_,
        "random_state": RANDOM_STATE,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit one neonatal ICU outcome model at one timepoint."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Input CSV (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--timepoint",
        type=int,
        choices=(6, 12, 24),
        required=True,
        help="Ordered-record timepoint used by the notebook.",
    )
    parser.add_argument(
        "--outcome",
        choices=tuple(OUTCOMES),
        required=True,
        help="Prediction outcome.",
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_NAMES),
        required=True,
        help="Model family.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for the analysis result JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    print(f"Reading required columns from {source}...", flush=True)
    source_df = read_source(source)
    print(
        f"Loaded {len(source_df):,} rows across "
        f"{source_df['cpmrn'].nunique():,} CPMRN values.",
        flush=True,
    )

    interval = build_interval_dataset(source_df, args.timepoint)
    del source_df
    print(
        f"Built {args.timepoint}-record dataset with {len(interval):,} patients "
        "after transfer exclusions.",
        flush=True,
    )

    result = train_and_evaluate(interval, args.outcome, args.model)
    result["timepoint"] = args.timepoint
    result["source"] = str(source)

    rendered = json.dumps(result, indent=2, default=str)
    print(rendered)
    if args.output_json:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
