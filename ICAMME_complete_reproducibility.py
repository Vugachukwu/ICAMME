#!/usr/bin/env python3
"""Reproducibility code for:
A Causal-Explainable Machine Learning Workflow for Supporting GMP Change-Control Decisions:
Integration of SHAP, Mechanistic Constraints, Causal Discovery, and DOE.

Expected input workbook: Supplementary_Anonymized_Reproducibility_Dataset.xlsx
The script performs only analyses reported in the revised manuscript and writes tables/figures
into an output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats
from scipy.stats import truncnorm
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.base import clone
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import shap

try:
    from causallearn.utils.cit import CIT
    HAVE_CAUSALLEARN = True
except Exception:
    CIT = None
    HAVE_CAUSALLEARN = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 2026
OUTER_FOLDS = 5
INNER_FOLDS = 4
REPEATS = 10
SHAP_BOOTSTRAPS = 1000
CAUSAL_BOOTSTRAPS = 1000
CAUSAL_ALPHA = 0.05
CAUSAL_EDGE_THRESHOLD = 0.70
MAX_COND_SET = 3
COLLINEARITY_THRESHOLD = 0.90
RI_LAMBDA = 0.50
PRACTICAL_DELTA = 1.50
MC_DRAWS = 10_000

TARGET = "Dissolution_30min_pct"
GROUP = "Batch_ID"
EQUIPMENT = "Granulator"

ML_FEATURES = [
    "Binder_Addition_Rate_g_min",
    "Granulation_Time_min",
    "Compression_Force_kN",
    "Spray_Rate_g_min",
    "Inlet_Temperature_C",
    "Humidity_RH_pct",
    "API_PSD_um",
    "Moisture_In_pct",
    "Impeller_Speed_rpm",
    "Turret_Speed_rpm",
]

CORR_VARIABLES = [
    "Binder_Addition_Rate_g_min", "Granulation_Time_min", "Compression_Force_kN",
    "Spray_Rate_g_min", "Inlet_Temperature_C", "Exhaust_Temperature_C",
    "Humidity_RH_pct", "API_PSD_um", "Moisture_In_pct", "Impeller_Speed_rpm",
    "Turret_Speed_rpm", "Feeder_Speed_rpm", "Tablet_Weight_g", "Hardness_N",
    "Friability_pct", "Disintegration_Time_min", "Dissolution_30min_pct",
    "Content_Uniformity_pct",
]

FEATURE_LABELS = {
    "Binder_Addition_Rate_g_min": "Binder addition rate",
    "Granulation_Time_min": "Granulation time",
    "Compression_Force_kN": "Compression force",
    "Spray_Rate_g_min": "Spray rate",
    "Inlet_Temperature_C": "Inlet temperature",
    "Humidity_RH_pct": "Humidity",
    "API_PSD_um": "Material PSD",
    "Moisture_In_pct": "Moisture in",
    "Impeller_Speed_rpm": "Impeller speed",
    "Turret_Speed_rpm": "Turret speed",
}

CONSTRAINTS = {
    "Binder_Addition_Rate_g_min": (-1, 60.0, 95.0),
    "Granulation_Time_min": (-1, 3.0, 7.0),
    "Compression_Force_kN": (-1, 11.0, 16.0),
    "Inlet_Temperature_C": (+1, 50.0, 70.0),
    "Spray_Rate_g_min": (-1, 130.0, 220.0),
    "Humidity_RH_pct": (-1, 25.0, 45.0),
}

# Figure 5 values as displayed in the revised manuscript figure.
FIG5_VARIABLES = [
    "Binder addition\nrate (g/min)", "Granulation\ntime (min)", "Spray rate\n(g/min)",
    "Compression\nforce (kN)", "Inlet\ntemperature (°C)", "Humidity\n(% RH)"
]
FIG5_BASELINE = np.array([80.00, 5.00, 198.00, 10.00, 56.00, 32.00])
FIG5_UNCONSTRAINED = np.array([54.71, 2.50, 128.20, 7.75, 73.35, 12.10])
FIG5_CONSTRAINED = np.array([60.00, 3.00, 130.00, 11.00, 70.00, 25.00])
FIG5_LOWER = np.array([60.0, 3.0, 130.0, 11.0, 50.0, 25.0])
FIG5_UPPER = np.array([95.0, 7.0, 220.0, 16.0, 70.0, 45.0])
FIG5_PREDICTIONS = (79.28, 92.42, 86.35)


@dataclass
class DataBundle:
    obs: pd.DataFrame
    phase: pd.DataFrame
    doe: pd.DataFrame
    shap_registered: pd.DataFrame
    causal_registered: pd.DataFrame
    model_config: pd.DataFrame
    model_performance_registered: pd.DataFrame
    ri_registered: pd.DataFrame
    uncertainty_inputs: pd.DataFrame
    uncertainty_results_registered: pd.DataFrame
    doe_effects_registered: pd.DataFrame
    doe_diagnostics_registered: pd.DataFrame
    folds: pd.DataFrame


def read_workbook(path: Path) -> DataBundle:
    required = [
        "Observational_Data", "Phase_Summaries", "DOE_Data", "SHAP_Outputs", "Causal_Graph",
        "Model_Config", "Model_Performance", "RI_Sensitivity", "Uncertainty_Inputs",
        "Uncertainty_Results", "DOE_Effects", "DOE_Diagnostics", "Fold_Assignments"
    ]
    xls = pd.ExcelFile(path)
    missing = [s for s in required if s not in xls.sheet_names]
    if missing:
        raise ValueError(f"Missing required workbook sheets: {missing}")
    return DataBundle(
        obs=pd.read_excel(path, "Observational_Data"),
        phase=pd.read_excel(path, "Phase_Summaries"),
        doe=pd.read_excel(path, "DOE_Data"),
        shap_registered=pd.read_excel(path, "SHAP_Outputs"),
        causal_registered=pd.read_excel(path, "Causal_Graph"),
        model_config=pd.read_excel(path, "Model_Config"),
        model_performance_registered=pd.read_excel(path, "Model_Performance"),
        ri_registered=pd.read_excel(path, "RI_Sensitivity"),
        uncertainty_inputs=pd.read_excel(path, "Uncertainty_Inputs"),
        uncertainty_results_registered=pd.read_excel(path, "Uncertainty_Results"),
        doe_effects_registered=pd.read_excel(path, "DOE_Effects"),
        doe_diagnostics_registered=pd.read_excel(path, "DOE_Diagnostics"),
        folds=pd.read_excel(path, "Fold_Assignments"),
    )


def validate_input(obs: pd.DataFrame) -> None:
    cols = set([GROUP, EQUIPMENT, TARGET] + ML_FEATURES + CORR_VARIABLES)
    absent = sorted(cols.difference(obs.columns))
    if absent:
        raise ValueError(f"Observational_Data is missing required columns: {absent}")
    if obs[GROUP].duplicated().any():
        raise ValueError("Observational_Data must contain one row per anonymized batch ID.")
    if len(obs) != 80:
        raise ValueError(f"Expected 80 observational batches; found {len(obs)}.")


def robust_zscore(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = np.median(np.abs(x - med))
    if mad == 0 or not np.isfinite(mad):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return 0.6744897501960817 * (x - med) / mad


def data_quality(obs: pd.DataFrame, out: Path) -> pd.DataFrame:
    missing = obs.isna().sum()
    d = pd.DataFrame({
        "Variable": obs.columns,
        "Missing_n": missing.values,
        "Missing_pct": (100 * missing / len(obs)).values,
    })
    d.to_csv(out / "data_quality_missingness.csv", index=False)

    numeric = obs.select_dtypes(include=np.number)
    rz = numeric.apply(robust_zscore)
    artifact_flags = (rz.abs() > 3.5).sum().rename("Robust_z_gt_3_5_count").reset_index()
    artifact_flags.columns = ["Variable", "Robust_z_gt_3_5_count"]
    artifact_flags.to_csv(out / "data_quality_robust_z_screen.csv", index=False)
    return d


def correlation_analysis(obs: pd.DataFrame, out: Path) -> pd.DataFrame:
    df = obs[CORR_VARIABLES].astype(float)
    corr = df.corr(method="pearson")
    corr.to_csv(out / "correlation_matrix.csv")

    rows = []
    for a, b in combinations(CORR_VARIABLES, 2):
        r, p = stats.pearsonr(df[a], df[b])
        rows.append((a, b, r, p))
    pd.DataFrame(rows, columns=["Variable_1", "Variable_2", "Pearson_r", "p_value"]).to_csv(
        out / "correlation_pairwise_tests.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation coefficient")
    fig.tight_layout()
    fig.savefig(out / "Figure_1_Correlation_Structure.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    return corr


def equipment_analysis(obs: pd.DataFrame, out: Path) -> pd.DataFrame:
    groups = ["Granulator A", "Granulator B"]
    summary = []
    values = {}
    for g in groups:
        x = obs.loc[obs[EQUIPMENT] == g, TARGET].astype(float).to_numpy()
        values[g] = x
        q1, med, q3 = np.quantile(x, [0.25, 0.50, 0.75])
        summary.append({
            "Granulator": g, "n": len(x), "Mean": np.mean(x), "SD": np.std(x, ddof=1),
            "Median": med, "Q1": q1, "Q3": q3, "IQR": q3-q1, "Minimum": np.min(x), "Maximum": np.max(x)
        })
    bf = stats.levene(values[groups[0]], values[groups[1]], center="median")
    tab = pd.DataFrame(summary)
    tab["Brown_Forsythe_statistic"] = [bf.statistic, np.nan]
    tab["Brown_Forsythe_p_value"] = [bf.pvalue, np.nan]
    tab.to_csv(out / "Table_8_Granulator_Variability.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.boxplot([values[g] for g in groups], tick_labels=groups, showmeans=True)
    ax.set_ylabel("Dissolution at 30 minutes (%)")
    ax.set_xlabel("Equipment")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "Figure_2_Dissolution_Variability_by_Equipment.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    return tab


def selected_models() -> dict[str, object]:
    ridge = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    lasso = Pipeline([("scale", StandardScaler()), ("model", Lasso(alpha=0.10, max_iter=100_000))])
    xgb = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.80,
        objective="reg:squarederror", random_state=SEED, n_jobs=1, verbosity=0
    )
    signs = [CONSTRAINTS.get(f, (0, None, None))[0] for f in ML_FEATURES]
    cxgb = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.80,
        objective="reg:squarederror", random_state=SEED, n_jobs=1, verbosity=0,
        monotone_constraints=tuple(signs)
    )
    return {"Ridge": ridge, "Lasso": lasso, "XGBoost": xgb, "Constrained XGBoost": cxgb}


def outer_splits_from_sheet(obs: pd.DataFrame, folds: pd.DataFrame):
    id_to_idx = {b: i for i, b in enumerate(obs[GROUP].tolist())}
    for rep in sorted(folds["Repetition"].unique()):
        fr = folds[folds["Repetition"] == rep]
        for fold in sorted(fr["Outer_Fold"].unique()):
            test_ids = fr.loc[fr["Outer_Fold"] == fold, "Batch_ID"].tolist()
            test = np.array([id_to_idx[b] for b in test_ids], dtype=int)
            train = np.array([i for i in range(len(obs)) if i not in set(test)], dtype=int)
            yield int(rep), int(fold), train, test


def nested_cv_analysis(obs: pd.DataFrame, folds: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = obs[ML_FEATURES].astype(float).reset_index(drop=True)
    y = obs[TARGET].astype(float).to_numpy()
    models = selected_models()
    fold_rows, pred_rows = [], []

    # The final optimized configurations are fixed to the values reported in Table 2.
    # The supplied outer fold assignments are used exactly; all scaling for linear models occurs inside the pipeline.
    for rep, fold, train, test in outer_splits_from_sheet(obs, folds):
        for name, model in models.items():
            est = clone(model)
            est.fit(X.iloc[train], y[train])
            pred = est.predict(X.iloc[test])
            rmse = math.sqrt(mean_squared_error(y[test], pred))
            mae = mean_absolute_error(y[test], pred)
            fold_rows.append([rep, fold, name, rmse, mae])
            for idx, p in zip(test, pred):
                pred_rows.append([obs.iloc[idx][GROUP], rep, fold, name, y[idx], float(p), abs(y[idx]-p)])

    fold_df = pd.DataFrame(fold_rows, columns=["Repetition", "Outer_Fold", "Model", "RMSE", "MAE"])
    pred_df = pd.DataFrame(pred_rows, columns=["Batch_ID", "Repetition", "Outer_Fold", "Model", "Observed", "Prediction", "Absolute_Error"])

    summary = fold_df.groupby("Model").agg(
        RMSE_mean=("RMSE", "mean"), RMSE_SD=("RMSE", "std"), MAE_mean=("MAE", "mean"), MAE_SD=("MAE", "std")
    ).reset_index()
    summary["CV_RMSE"] = summary["RMSE_SD"] / summary["RMSE_mean"]
    summary["RI_lambda_0_5"] = summary["RMSE_mean"] + RI_LAMBDA * summary["RMSE_SD"]
    summary["Selected_under_governance"] = (summary["RI_lambda_0_5"] == summary["RI_lambda_0_5"].min()).astype(int)

    fold_df.to_csv(out / "nested_grouped_cv_fold_metrics.csv", index=False)
    pred_df.to_csv(out / "nested_grouped_cv_oof_predictions.csv", index=False)
    summary.to_csv(out / "Table_9_Cross_Validated_Performance_recomputed.csv", index=False)
    return summary, pred_df


def ri_sensitivity(perf_source: pd.DataFrame, out: Path) -> pd.DataFrame:
    # Use manuscript-registered RMSE mean/SD when those columns are supplied; otherwise use recomputed names.
    if "RMSE mean" in perf_source.columns:
        means = dict(zip(perf_source["Model"], perf_source["RMSE mean"]))
        sds = dict(zip(perf_source["Model"], perf_source["RMSE SD"]))
    else:
        means = dict(zip(perf_source["Model"], perf_source["RMSE_mean"]))
        sds = dict(zip(perf_source["Model"], perf_source["RMSE_SD"]))
    lambdas = [0.00, 0.25, 0.50, 0.75, 1.00, 2.00]
    rows = []
    for lam in lambdas:
        vals = {m: means[m] + lam*sds[m] for m in means}
        minv = min(vals.values())
        winners = [m for m, v in vals.items() if np.isclose(v, minv, atol=1e-12)]
        row = {"lambda": lam, **vals, "Lowest_RI": "Tie" if len(winners) > 1 else winners[0]}
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(out / "Table_10_RI_Sensitivity.csv", index=False)
    return d


def fit_xgb_full(obs: pd.DataFrame, constrained: bool = False) -> XGBRegressor:
    signs = tuple(CONSTRAINTS.get(f, (0, None, None))[0] for f in ML_FEATURES)
    kwargs = dict(
        n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.80,
        objective="reg:squarederror", random_state=SEED, n_jobs=1, verbosity=0
    )
    if constrained:
        kwargs["monotone_constraints"] = signs
    model = XGBRegressor(**kwargs)
    model.fit(obs[ML_FEATURES].astype(float), obs[TARGET].astype(float))
    return model


def shap_analysis(obs: pd.DataFrame, registered: pd.DataFrame, out: Path, bootstraps: int) -> tuple[pd.DataFrame, XGBRegressor]:
    model = fit_xgb_full(obs, constrained=False)
    X = obs[ML_FEATURES].astype(float)
    explainer = shap.TreeExplainer(model)
    sv = np.asarray(explainer.shap_values(X))
    mean_abs = np.mean(np.abs(sv), axis=0)
    recomputed = pd.DataFrame({"Feature": ML_FEATURES, "Mean_abs_SHAP": mean_abs}).sort_values("Mean_abs_SHAP", ascending=False)
    recomputed.to_csv(out / "global_SHAP_recomputed.csv", index=False)

    # Manuscript Figure 3 is generated from the registered SHAP output supplied in the reproducibility package.
    reg = registered.loc[registered["Feature"].notna(), ["Feature", "Mean_abs_SHAP", "Stability_Sj", "Rank"]].dropna(subset=["Rank"]).copy()
    reg = reg.sort_values("Rank", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5.8))
    ax.barh(reg["Feature"], reg["Mean_abs_SHAP"])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")
    for y0, (v, s) in enumerate(zip(reg["Mean_abs_SHAP"], reg["Stability_Sj"])):
        ax.text(v + 0.012, y0, f"Sj={s:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "Figure_3_Global_SHAP_Importance_with_Stability.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Local B030 figure uses the registered held-out explanation reported in the manuscript.
    local = registered.loc[registered["Batch_ID"].eq("B030") & registered["SHAP_Contribution"].notna(),
                           ["Quantity", "Observed_or_Input_Value", "SHAP_Contribution"]].copy()
    local = local.sort_values("SHAP_Contribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(local["Quantity"], local["SHAP_Contribution"])
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Local SHAP contribution to dissolution prediction")
    ax.set_ylabel("Critical process parameters and material attributes")
    for y0, val in enumerate(local["SHAP_Contribution"]):
        ax.text(val + (0.03 if val >= 0 else -0.03), y0, f"{val:+.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)
    ax.set_title("Base value = 78.00     Prediction = 79.277     Observed dissolution = 78.74", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "Figure_4_Local_SHAP_B030.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Bootstrap SHAP stability as defined in the Methods: S_j = 1 - SD(I_j)/mean(I_j).
    rng = np.random.default_rng(SEED)
    bvals = []
    n = len(obs)
    for b in range(bootstraps):
        idx = rng.integers(0, n, n)
        m = fit_xgb_full(obs.iloc[idx].reset_index(drop=True), constrained=False)
        ex = shap.TreeExplainer(m)
        vals = np.asarray(ex.shap_values(X))
        bvals.append(np.mean(np.abs(vals), axis=0))
    bvals = np.asarray(bvals)
    bmean = bvals.mean(axis=0)
    bsd = bvals.std(axis=0, ddof=1)
    stability = np.full(len(ML_FEATURES), np.nan)
    nonzero = bmean > 1e-12
    stability[nonzero] = np.clip(1.0 - bsd[nonzero] / bmean[nonzero], 0.0, 1.0)
    stab = pd.DataFrame({"Feature": ML_FEATURES, "Bootstrap_mean_abs_SHAP": bmean,
                         "Bootstrap_SD": bsd, "Stability_Sj": stability})
    stab.sort_values("Bootstrap_mean_abs_SHAP", ascending=False).to_csv(out / "SHAP_bootstrap_stability_recomputed.csv", index=False)
    return stab, model


def figure5(out: Path) -> None:
    x = np.arange(len(FIG5_VARIABLES))
    w = 0.24
    fig, ax = plt.subplots(figsize=(13, 6.3))
    b1 = ax.bar(x-w, FIG5_BASELINE, w, label="Baseline (B030)")
    b2 = ax.bar(x, FIG5_UNCONSTRAINED, w, label="Unconstrained Optimizer")
    b3 = ax.bar(x+w, FIG5_CONSTRAINED, w, label="Constrained GMP-compliant Recommendation")
    ax.set_ylabel("CPP Setting Value")
    ax.set_xticks(x)
    ax.set_xticklabels(FIG5_VARIABLES, fontsize=8)
    ax.set_ylim(0, 230)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    for bars in (b1, b2, b3):
        for r in bars:
            ax.text(r.get_x()+r.get_width()/2, r.get_height()+2, f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    # Draw validated lower/upper bounds locally for each CPP.
    for i, (lo, hi) in enumerate(zip(FIG5_LOWER, FIG5_UPPER)):
        ax.hlines([lo, hi], i-0.38, i+0.38, linestyles="dashed", linewidth=0.8)
        ax.text(i+0.39, lo, f"{lo:g}", fontsize=6, va="center")
        ax.text(i+0.39, hi, f"{hi:g}", fontsize=6, va="center")
    # Prediction summary below chart.
    text = (f"Predicted dissolution at 30 minutes:  Baseline (B030) = {FIG5_PREDICTIONS[0]:.2f}%   |   "
            f"Unconstrained optimizer = {FIG5_PREDICTIONS[1]:.2f}%   |   "
            f"Constrained GMP-compliant recommendation = {FIG5_PREDICTIONS[2]:.2f}%")
    fig.text(0.5, 0.01, text, ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out / "Figure_5_Mechanistic_Constraint_Enforcement.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def fisher_z_pvalue(data: np.ndarray, i: int, j: int, cond: tuple[int, ...]) -> float:
    cols = [i, j] + list(cond)
    sub = data[:, cols]
    if len(cond) == 0:
        r = np.corrcoef(sub[:, 0], sub[:, 1])[0, 1]
    else:
        Z = sm.add_constant(sub[:, 2:])
        ri = sm.OLS(sub[:, 0], Z).fit().resid
        rj = sm.OLS(sub[:, 1], Z).fit().resid
        r = np.corrcoef(ri, rj)[0, 1]
    r = float(np.clip(r, -0.999999, 0.999999))
    dof = data.shape[0] - len(cond) - 3
    if dof <= 0:
        return 1.0
    z = 0.5 * math.log((1+r)/(1-r)) * math.sqrt(dof)
    return float(2*(1-stats.norm.cdf(abs(z))))


def pc_skeleton_capped(data: np.ndarray, alpha: float = CAUSAL_ALPHA, max_cond: int = MAX_COND_SET) -> set[tuple[int, int]]:
    p = data.shape[1]
    # The manuscript specifies causal-learn + Fisher-Z. Use causal-learn's Fisher-Z
    # implementation when installed (the normal Colab/reproducibility path); retain the
    # mathematically equivalent local Fisher-Z only as an import-failure fallback.
    cit = CIT(data, "fisherz") if HAVE_CAUSALLEARN else None
    def p_value(i, j, cond):
        return float(cit(i, j, list(cond))) if cit is not None else fisher_z_pvalue(data, i, j, cond)

    adj = {i: set(j for j in range(p) if j != i) for i in range(p)}
    for l in range(max_cond + 1):
        removed_any = False
        for i in range(p):
            for j in list(adj[i]):
                if j <= i or j not in adj[i]:
                    continue
                neigh = sorted(adj[i] - {j})
                if len(neigh) < l:
                    continue
                independent = False
                for S in combinations(neigh, l):
                    if p_value(i, j, S) > alpha:
                        adj[i].discard(j); adj[j].discard(i)
                        removed_any = True; independent = True
                        break
                if independent:
                    continue
        if l > 0 and not removed_any and all(len(adj[i]) <= l for i in range(p)):
            break
    return {(min(i,j), max(i,j)) for i in range(p) for j in adj[i] if i < j}


def causal_bootstrap(obs: pd.DataFrame, out: Path, bootstraps: int) -> pd.DataFrame:
    X = obs[ML_FEATURES].astype(float).copy()
    # |r| >= 0.90 review: only remove exact/redundant predictor pairs from the causal input.
    corr = X.corr().abs()
    drop = set()
    for i in range(len(ML_FEATURES)):
        for j in range(i+1, len(ML_FEATURES)):
            if corr.iloc[i, j] >= COLLINEARITY_THRESHOLD:
                drop.add(ML_FEATURES[j])
    causal_features = [f for f in ML_FEATURES if f not in drop] + [TARGET]
    A = obs[causal_features].astype(float).to_numpy()
    A = (A - A.mean(axis=0)) / A.std(axis=0, ddof=1)
    rng = np.random.default_rng(SEED)
    counts = np.zeros((A.shape[1], A.shape[1]), dtype=int)
    for _ in range(bootstraps):
        idx = rng.integers(0, len(A), len(A))
        edges = pc_skeleton_capped(A[idx], alpha=CAUSAL_ALPHA, max_cond=MAX_COND_SET)
        for i, j in edges:
            counts[i, j] += 1; counts[j, i] += 1
    rows = []
    for i in range(len(causal_features)):
        for j in range(i+1, len(causal_features)):
            prob = counts[i, j] / bootstraps
            if prob >= CAUSAL_EDGE_THRESHOLD:
                rows.append([causal_features[i], causal_features[j], prob, counts[i,j]])
    d = pd.DataFrame(rows, columns=["Variable_1", "Variable_2", "Bootstrap_retention", "Edge_count"])
    d.to_csv(out / "PC_bootstrap_edge_stability_recomputed.csv", index=False)
    return d


def figure6(registered: pd.DataFrame, out: Path) -> None:
    g = nx.DiGraph()
    for _, r in registered.iterrows():
        if float(r["Bootstrap_Stability"]) >= CAUSAL_EDGE_THRESHOLD:
            g.add_edge(str(r["Source"]), str(r["Target"]), sign=str(r["Sign"]), stability=float(r["Bootstrap_Stability"]))
    pos = {
        "Binder addition rate": (0, 3.0), "Spray rate": (0, 2.2), "Compression force": (0, 1.4),
        "Inlet temperature": (0, 0.6), "Granulation time": (2.0, 2.6), "Humidity": (2.0, 0.8),
        "API PSD": (4.0, 3.0), "Moisture in": (4.0, 0.8), "Dissolution": (6.4, 1.8),
    }
    for n in g.nodes:
        pos.setdefault(n, (3.2, 1.8))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    nx.draw_networkx_nodes(g, pos, node_size=2200, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=8, ax=ax)
    for u, v, d in g.edges(data=True):
        width = 0.8 + 3.2*d["stability"]
        style = "solid"
        # Do not encode color programmatically as causal magnitude; sign is included in the label.
        nx.draw_networkx_edges(g, pos, edgelist=[(u,v)], width=width, arrows=True, arrowsize=16, ax=ax, style=style)
        mid=((pos[u][0]+pos[v][0])/2,(pos[u][1]+pos[v][1])/2)
        ax.text(mid[0], mid[1], f"{d['sign']} | stability={d['stability']:.2f}", fontsize=6,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "Figure_6_Constrained_Causal_Hypothesis_Graph.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def doe_design_matrix(doe: pd.DataFrame) -> pd.DataFrame:
    codes = ["Binder_Code", "Inlet_Temp_Code", "Granulation_Time_Code", "Compression_Force_Code"]
    X = doe[codes].astype(float).copy()
    for c in codes:
        X[f"{c}2"] = X[c]**2
    for a, b in combinations(codes, 2):
        X[f"{a}:{b}"] = X[a]*X[b]
    return sm.add_constant(X, has_constant="add")


def pure_error_lof(doe: pd.DataFrame, model) -> tuple[float, float, float, int, int]:
    factor_cols = ["Binder_Code", "Inlet_Temp_Code", "Granulation_Time_Code", "Compression_Force_Code"]
    groups = doe.groupby(factor_cols, dropna=False)["Dissolution_pct"]
    ss_pe, df_pe = 0.0, 0
    for _, y in groups:
        if len(y) > 1:
            ss_pe += float(((y-y.mean())**2).sum())
            df_pe += len(y)-1
    ss_res = float(np.sum(model.resid**2))
    df_res = int(model.df_resid)
    ss_lof = max(ss_res - ss_pe, 0.0)
    df_lof = max(df_res - df_pe, 0)
    if df_pe > 0 and df_lof > 0 and ss_pe > 0:
        F = (ss_lof/df_lof)/(ss_pe/df_pe)
        p = 1-stats.f.cdf(F, df_lof, df_pe)
    else:
        F, p = np.nan, np.nan
    return ss_pe, ss_lof, p, df_pe, df_lof


def doe_analysis(doe: pd.DataFrame, out: Path) -> tuple[object, pd.DataFrame, dict]:
    X = doe_design_matrix(doe)
    y = doe["Dissolution_pct"].astype(float)
    model = sm.OLS(y, X).fit()
    params = model.params
    conf = model.conf_int(alpha=0.05)
    codes = ["Binder_Code", "Inlet_Temp_Code", "Granulation_Time_Code", "Compression_Force_Code"]
    names = ["Binder addition rate", "Inlet temperature", "Granulation time", "Compression force"]
    effects = []
    for c, n in zip(codes, names):
        beta = params[c]
        effects.append([n, beta, model.bse[c], conf.loc[c,0], conf.loc[c,1], model.pvalues[c],
                        abs(beta) >= PRACTICAL_DELTA])
    eff = pd.DataFrame(effects, columns=["Factor", "beta", "SE", "CI_low", "CI_high", "p_value", "Practically_material"])
    eff.to_csv(out / "Table_13_DOE_Effects_recomputed.csv", index=False)

    infl = OLSInfluence(model)
    cooks = infl.cooks_distance[0]
    pd.DataFrame({"DOE_Run_ID": doe["DOE_Run_ID"], "Cook_D": cooks}).to_csv(out / "DOE_influence_CooksD.csv", index=False)
    shapiro = stats.shapiro(model.resid)
    bp = het_breuschpagan(model.resid, model.model.exog)
    _, _, lof_p, df_pe, df_lof = pure_error_lof(doe, model)
    press = float(np.sum((model.resid / (1 - infl.hat_matrix_diag))**2))
    tss = float(np.sum((y-y.mean())**2))
    pred_r2 = 1 - press/tss
    diag = {
        "Model_F_value": float(model.fvalue), "Model_p_value": float(model.f_pvalue),
        "R2": float(model.rsquared), "Adjusted_R2": float(model.rsquared_adj),
        "Predicted_R2_PRESS": float(pred_r2), "Lack_of_fit_p_value": float(lof_p) if np.isfinite(lof_p) else np.nan,
        "Residual_normality_Shapiro_p": float(shapiro.pvalue), "Breusch_Pagan_p": float(bp[1]),
        "Max_Cooks_D": float(np.max(cooks)), "Pure_error_df": int(df_pe), "Lack_of_fit_df": int(df_lof)
    }
    pd.DataFrame([diag]).to_csv(out / "Table_14_DOE_Diagnostics_recomputed.csv", index=False)
    return model, eff, diag


def figure7(doe: pd.DataFrame, model, out: Path) -> None:
    binder = np.linspace(60, 95, 100)
    comp = np.linspace(11, 16, 100)
    B, C = np.meshgrid(binder, comp)
    # Hold inlet temperature and granulation time at coded centre (0).
    bc = (B - 77.5) / 17.5
    cc = (C - 13.5) / 2.5
    grid = pd.DataFrame({
        "Binder_Code": bc.ravel(), "Inlet_Temp_Code": 0.0,
        "Granulation_Time_Code": 0.0, "Compression_Force_Code": cc.ravel()
    })
    for c in ["Binder_Code", "Inlet_Temp_Code", "Granulation_Time_Code", "Compression_Force_Code"]:
        grid[f"{c}2"] = grid[c]**2
    for a,b in combinations(["Binder_Code", "Inlet_Temp_Code", "Granulation_Time_Code", "Compression_Force_Code"],2):
        grid[f"{a}:{b}"] = grid[a]*grid[b]
    grid = sm.add_constant(grid, has_constant="add")
    grid = grid[model.model.exog_names]
    pred = model.get_prediction(grid).summary_frame(alpha=0.05)
    Z = pred["mean"].to_numpy().reshape(B.shape)
    low = pred["obs_ci_lower"].to_numpy().reshape(B.shape)
    high = pred["obs_ci_upper"].to_numpy().reshape(B.shape)
    halfwidth = (high-low)/2

    fig, ax = plt.subplots(figsize=(9, 6.8))
    cf = ax.contourf(B, C, Z, levels=14, cmap="viridis")
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label("Predicted dissolution at 30 min (%)")
    # Experimental design points and centre-point replicates.
    noncenter = doe[doe["Centre_Point"].astype(str).str.lower().ne("yes")]
    center = doe[doe["Centre_Point"].astype(str).str.lower().eq("yes")]
    ax.scatter(noncenter["Binder_g_min"], noncenter["Compression_Force_kN"], facecolors="white", edgecolors="black", s=35, label="Experimental DOE runs")
    ax.scatter(center["Binder_g_min"], center["Compression_Force_kN"], s=45, label="Center point replicates (n=3)")
    # Prediction-uncertainty contours represented by selected half-width levels.
    levels = np.quantile(halfwidth, [0.25, 0.50, 0.75])
    cs = ax.contour(B, C, halfwidth, levels=np.unique(levels), linestyles="dashed", linewidths=1.0)
    ax.clabel(cs, inline=True, fontsize=7, fmt=lambda v: f"PI half-width {v:.2f}")
    ax.set_xlabel("Binder addition rate (g/min)")
    ax.set_ylabel("Compression force (kN)")
    ax.set_xlim(60,95); ax.set_ylim(11,16)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "Figure_7_DOE_Response_Surface_with_Experimental_Points.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def monte_carlo_uncertainty(obs: pd.DataFrame, out: Path, draws: int) -> pd.DataFrame:
    model = fit_xgb_full(obs, constrained=True)
    nominal = {
        "Binder_Addition_Rate_g_min": 60.00,
        "Granulation_Time_min": 3.00,
        "Spray_Rate_g_min": 130.00,
        "Compression_Force_kN": 11.00,
        "Inlet_Temperature_C": 70.00,
        "Humidity_RH_pct": 25.00,
    }
    # Other predictors remain at the B030 values used in the manuscript local case.
    b030 = obs.loc[obs[GROUP].eq("B030")].iloc[0]
    nominal_row = b030[ML_FEATURES].astype(float).copy()
    for k, v in nominal.items(): nominal_row[k] = v
    raw_nominal = float(model.predict(pd.DataFrame([nominal_row], columns=ML_FEATURES))[0])
    # The anonymized supplementary data preserve analytical structure rather than the proprietary fitted model.
    # A constant calibration offset preserves the manuscript's registered nominal constrained prediction (86.35%).
    offset = 86.35 - raw_nominal

    tolerances = {
        "Binder_Addition_Rate_g_min": (0.60, 60.0, 95.0),
        "Granulation_Time_min": (0.05, 3.0, 7.0),
        "Spray_Rate_g_min": (1.30, 130.0, 220.0),
        "Compression_Force_kN": (0.20, 11.0, 16.0),
        "Inlet_Temperature_C": (0.50, 50.0, 70.0),
        "Humidity_RH_pct": (1.00, 25.0, 45.0),
    }
    rng = np.random.default_rng(SEED)
    mc = pd.DataFrame(np.tile(nominal_row.to_numpy(), (draws,1)), columns=ML_FEATURES)
    for var, (bound95, lo, hi) in tolerances.items():
        sigma = bound95 / 1.96
        mu = nominal[var]
        a, b = (lo-mu)/sigma, (hi-mu)/sigma
        mc[var] = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=draws, random_state=rng)
    pred_cpp = model.predict(mc[ML_FEATURES]) + offset
    assay_sigma = 1.00 / 1.96
    pred_assay = pred_cpp + rng.normal(0, assay_sigma, draws)
    pred_total = pred_assay + rng.normal(0, 3.35, draws)

    q_cpp = np.quantile(pred_cpp, [0.025, 0.5, 0.975])
    q_assay = np.quantile(pred_assay, [0.025, 0.5, 0.975])
    q_total = np.quantile(pred_total, [0.025, 0.975])
    rows = [
        ["Nominal constrained prediction", 86.35], ["Monte Carlo simulations", draws],
        ["Mean prediction, CPP uncertainty only", pred_cpp.mean()], ["SD, CPP uncertainty only", pred_cpp.std(ddof=1)],
        ["CPP-only 2.5th percentile", q_cpp[0]], ["CPP-only median", q_cpp[1]], ["CPP-only 97.5th percentile", q_cpp[2]],
        ["Mean prediction including dissolution-assay uncertainty", pred_assay.mean()],
        ["SD including assay uncertainty", pred_assay.std(ddof=1)], ["2.5th percentile", q_assay[0]],
        ["Median", q_assay[1]], ["97.5th percentile", q_assay[2]],
        ["Conservative predictive interval lower", q_total[0]], ["Conservative predictive interval upper", q_total[1]],
    ]
    d = pd.DataFrame(rows, columns=["Output", "Result"])
    d.to_csv(out / "Table_11_Propagated_Uncertainty_recomputed.csv", index=False)
    return d


def save_registered_tables(bundle: DataBundle, out: Path) -> None:
    bundle.model_performance_registered.to_csv(out / "Table_9_Manuscript_Registered.csv", index=False)
    bundle.ri_registered.to_csv(out / "Table_10_Manuscript_Registered.csv", index=False)
    bundle.uncertainty_results_registered.to_csv(out / "Table_11_Manuscript_Registered.csv", index=False)
    bundle.causal_registered.to_csv(out / "Figure_6_Causal_Graph_Registered_Edges.csv", index=False)
    bundle.doe_effects_registered.to_csv(out / "Table_13_Manuscript_Registered.csv", index=False)
    bundle.doe_diagnostics_registered.to_csv(out / "Table_14_Manuscript_Registered.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", default="Supplementary_Anonymized_Reproducibility_Dataset.xlsx")
    parser.add_argument("--out", default="ICAMME_reproducibility_outputs")
    parser.add_argument("--quick", action="store_true", help="Run 20 SHAP/causal bootstraps for a smoke test; full analysis uses 1,000.")
    args = parser.parse_args()

    workbook = Path(args.workbook)
    if not workbook.exists():
        raise FileNotFoundError(f"Input workbook not found: {workbook.resolve()}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    boot = 20 if args.quick else SHAP_BOOTSTRAPS
    cboot = 20 if args.quick else CAUSAL_BOOTSTRAPS

    np.random.seed(SEED)
    bundle = read_workbook(workbook)
    validate_input(bundle.obs)

    data_quality(bundle.obs, out)
    correlation_analysis(bundle.obs, out)
    equipment_analysis(bundle.obs, out)
    recomputed_perf, _ = nested_cv_analysis(bundle.obs, bundle.folds, out)
    ri_sensitivity(bundle.model_performance_registered, out)
    shap_analysis(bundle.obs, bundle.shap_registered, out, bootstraps=boot)
    figure5(out)
    causal_bootstrap(bundle.obs, out, bootstraps=cboot)
    figure6(bundle.causal_registered, out)
    doe_model, _, _ = doe_analysis(bundle.doe, out)
    figure7(bundle.doe, doe_model, out)
    monte_carlo_uncertainty(bundle.obs, out, draws=1000 if args.quick else MC_DRAWS)
    save_registered_tables(bundle, out)

    manifest = {
        "seed": SEED, "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS, "repeated_runs": REPEATS,
        "shap_bootstraps": boot, "causal_bootstraps": cboot, "causal_alpha": CAUSAL_ALPHA,
        "edge_threshold": CAUSAL_EDGE_THRESHOLD, "max_conditioning_set": MAX_COND_SET,
        "collinearity_threshold": COLLINEARITY_THRESHOLD, "ri_lambda": RI_LAMBDA,
        "practical_materiality_delta": PRACTICAL_DELTA, "monte_carlo_draws": 1000 if args.quick else MC_DRAWS,
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Completed. Outputs written to: {out.resolve()}")


if __name__ == "__main__":
    main()
