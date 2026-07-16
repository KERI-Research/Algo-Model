"""
Causal Inference Engine Core
============================
This module establishes the core logical architecture for our causal analysis. By adopting a skeptical, questioning approach, we avoid taking correlations at face value and instead force the model to rigorously test the backdoor paths between obesity, diabetes, and cancer. Taking a forward-thinking view, this engine is decoupled from the data source, allowing you to seamlessly inject the UK Biobank data once authorized. You're doing incredible work pushing the boundaries of traditional ML—stay innovative and keep thinking outside the box.
"""
import pandas as pd
from dowhy import CausalModel
import networkx as nx
import numpy as np


class CausalExecutionError(RuntimeError):
    """Raised when strict causal execution is requested but DoWhy fails."""

    def __init__(self, message: str, *, fallback_result: dict | None = None):
        super().__init__(message)
        self.fallback_result = fallback_result


def _ensure_networkx_compatibility():
    """Provide d_separated shim for DoWhy on newer networkx versions."""
    if hasattr(nx.algorithms, "d_separated"):
        return

    if not hasattr(nx.algorithms, "d_separation"):
        return

    def _d_separated(graph, x, y, z):
        return nx.algorithms.d_separation.is_d_separator(
            graph,
            set(x),
            set(y),
            set(z),
        )

    nx.algorithms.d_separated = _d_separated


def _ensure_dowhy_pandas_compatibility():
    """Patch DoWhy RegressionEstimator for pandas versions where params[0] is label-based."""
    try:
        from dowhy.causal_estimator import CausalEstimate
        from dowhy.causal_estimators.regression_estimator import RegressionEstimator
    except Exception:
        return

    if getattr(RegressionEstimator, "_keri_pandas_patch", False):
        return

    def _estimate_effect_compat(self, data_df=None, need_conditional_estimates=None):
        if data_df is None:
            data_df = self._data
        if need_conditional_estimates is None:
            need_conditional_estimates = self.need_conditional_estimates

        if not self.model:
            _, self.model = self._build_model()
            coefficients = self.model.params[1:]
            self.logger.debug(
                "Coefficients of the fitted model: " + ",".join(map(str, coefficients))
            )
            self.logger.debug(self.model.summary())

        effect_estimate = self._do(self._treatment_value, data_df) - self._do(
            self._control_value, data_df
        )
        conditional_effect_estimates = None
        if need_conditional_estimates:
            conditional_effect_estimates = self._estimate_conditional_effects(
                self._estimate_effect_fn,
                effect_modifier_names=self._effect_modifier_names,
            )

        params = self.model.params
        if hasattr(params, "iloc"):
            intercept_parameter = params.iloc[0]
        else:
            intercept_parameter = params[0]

        estimate = CausalEstimate(
            estimate=effect_estimate,
            control_value=self._control_value,
            treatment_value=self._treatment_value,
            conditional_estimates=conditional_effect_estimates,
            target_estimand=self._target_estimand,
            realized_estimand_expr=self.symbolic_estimator,
            intercept=intercept_parameter,
        )
        return estimate

    RegressionEstimator._estimate_effect = _estimate_effect_compat
    RegressionEstimator._keri_pandas_patch = True


def _difference_in_means(df: pd.DataFrame, treatment_col: str, outcome_col: str) -> float:
    treated = df[df[treatment_col] == 1][outcome_col]
    control = df[df[treatment_col] == 0][outcome_col]

    if len(treated) == 0 or len(control) == 0:
        raise ValueError("Dataset must include both treatment=0 and treatment=1 rows.")

    return float(treated.mean() - control.mean())


def _prepare_model_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()

    if "Obesity" not in prepared.columns and "BMX_BMXBMI" in prepared.columns:
        bmi = pd.to_numeric(prepared["BMX_BMXBMI"], errors="coerce")
        prepared["Obesity"] = np.where(bmi >= 30, 1, np.nan)

    if "Diabetes" not in prepared.columns and "DIQ_DIQ010" in prepared.columns:
        diabetes = pd.to_numeric(prepared["DIQ_DIQ010"], errors="coerce")
        prepared["Diabetes"] = np.where(diabetes == 1, 1, np.where(diabetes == 2, 0, np.nan))

    if "Cancer" not in prepared.columns and "MCQ_MCQ220" in prepared.columns:
        cancer = pd.to_numeric(prepared["MCQ_MCQ220"], errors="coerce")
        prepared["Cancer"] = np.where(cancer == 1, 1, np.where(cancer == 2, 0, np.nan))

    required_columns = {"Diabetes", "Cancer", "Obesity"}
    missing_columns = required_columns - set(prepared.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(sorted(missing_columns))
        )

    model_df = prepared[["Diabetes", "Cancer", "Obesity"]].apply(pd.to_numeric, errors="coerce")
    model_df = model_df.dropna(subset=["Diabetes", "Cancer", "Obesity"])
    model_df = model_df.astype({"Diabetes": "int64", "Cancer": "int64", "Obesity": "int64"})
    return model_df


def _fallback_pipeline(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    *,
    cause: str | None = None,
):
    # Fallback estimator keeps API functional if DoWhy dependencies are incompatible.
    ate = _difference_in_means(df, treatment, outcome)

    rng = np.random.default_rng(42)

    placebo_df = df.copy()
    placebo_df[treatment] = rng.permutation(placebo_df[treatment].to_numpy())
    placebo_estimate = _difference_in_means(placebo_df, treatment, outcome)

    subset_df = df.sample(frac=0.9, random_state=42)
    subset_estimate = _difference_in_means(subset_df, treatment, outcome)

    random_cause = rng.integers(0, 2, len(df))
    random_df = df.assign(_random_cause=random_cause)
    weighted = []
    for value, group in random_df.groupby("_random_cause"):
        if group[treatment].nunique() < 2:
            continue
        effect = _difference_in_means(group, treatment, outcome)
        weighted.append((effect, len(group)))

    if weighted:
        total = sum(weight for _, weight in weighted)
        random_cause_estimate = sum(effect * weight for effect, weight in weighted) / total
    else:
        random_cause_estimate = ate

    return {
        "estimate": str(ate),
        "refutations": {
            "random_cause": str(random_cause_estimate),
            "placebo": str(placebo_estimate),
            "subset": str(subset_estimate),
        },
        "estimation_method": "fallback_difference_in_means",
        "execution_mode": "fallback_association",
        "warnings": [
            "DoWhy execution was unavailable. Returned association-based fallback estimate.",
            "Interpret as descriptive signal, not identified causal effect.",
        ],
        "dowhy_error": cause,
        "treatment": treatment,
        "outcome": outcome,
    }

def execute_pipeline(
    data_path,
    treatment: str = "Diabetes",
    outcome: str = "Cancer",
    allow_fallback: bool = True,
):
    if treatment == outcome:
        raise ValueError("Treatment and outcome must be different columns.")

    _ensure_networkx_compatibility()
    _ensure_dowhy_pandas_compatibility()
    df = pd.read_csv(data_path)
    df = _prepare_model_dataframe(df)

    if treatment not in df.columns or outcome not in df.columns:
        raise ValueError(
            f"Requested treatment/outcome columns are unavailable. treatment={treatment}, outcome={outcome}"
        )
    
    causal_graph = """
    digraph {
        Obesity -> Diabetes;
        Obesity -> Cancer;
        %(treatment)s -> %(outcome)s;
    }
    """ % {"treatment": treatment, "outcome": outcome}
    
    try:
        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            graph=causal_graph
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression"
        )

        random_cause = model.refute_estimate(
            identified_estimand, estimate, method_name="random_common_cause"
        )

        placebo = model.refute_estimate(
            identified_estimand, estimate, method_name="placebo_treatment_refuter", placebo_type="permute"
        )

        subset = model.refute_estimate(
            identified_estimand, estimate, method_name="data_subset_refuter", subset_fraction=0.9
        )

        return {
            "estimate": str(estimate.value),
            "refutations": {
                "random_cause": str(random_cause.estimated_effect),
                "placebo": str(placebo.estimated_effect),
                "subset": str(subset.estimated_effect)
            },
            "estimation_method": "dowhy",
            "execution_mode": "dowhy_causal",
            "warnings": [],
            "dowhy_error": None,
            "treatment": treatment,
            "outcome": outcome,
        }
    except Exception as error:
        fallback = _fallback_pipeline(
            df,
            treatment,
            outcome,
            cause=f"{type(error).__name__}: {error}",
        )
        if allow_fallback:
            return fallback
        raise CausalExecutionError(
            (
                "DoWhy causal estimation failed and fallback was disabled. "
                "Enable allow_fallback or fix causal dependency/runtime compatibility."
            ),
            fallback_result=fallback,
        ) from error