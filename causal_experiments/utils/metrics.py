"""Synthetic data quality evaluation metrics using SynthEval."""
from __future__ import annotations

import itertools
import random
import warnings

import matplotlib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import chi2_contingency

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from syntheval import SynthEval

plt.ioff()
matplotlib.rcParams["figure.max_open_warning"] = 0


def _quantile_edges_from_real(series: pd.Series, n_bins: int) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if values.size == 0:
        return np.array([])
    quantiles = np.linspace(0.0, 1.0, num=n_bins + 1)
    edges = np.quantile(values, quantiles)
    return np.unique(edges)


def _digitize_with_edges(values: pd.Series, edges: np.ndarray) -> pd.Series:
    encoded = pd.Series(-1, index=values.index, dtype=int)
    if edges.size < 2:
        return encoded
    numeric = pd.to_numeric(values, errors="coerce")
    mask = numeric.notna()
    if not mask.any():
        return encoded
    bins = edges[1:-1]
    indices = np.digitize(numeric.loc[mask].to_numpy(), bins=bins, right=False)
    indices = np.clip(indices, 0, len(edges) - 2)
    encoded.loc[mask] = indices
    return encoded


def _cramers_v(values_a: pd.Series, values_b: pd.Series) -> float:
    table = pd.crosstab(values_a, values_b)
    if table.empty:
        return 0.0
    stat = chi2_contingency(table)[0]
    obs = table.to_numpy().sum()
    if obs == 0:
        return 0.0
    d = min(table.shape) - 1
    if d <= 0:
        return 0.0
    return float(stat / (obs * d + 1e-16))


def _correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    cat = pd.Series(categories).reset_index(drop=True)
    meas = pd.to_numeric(measurements, errors="coerce").reset_index(drop=True)
    mask = cat.notna() & meas.notna()
    cat = cat[mask]
    meas = meas[mask]
    if cat.empty or meas.empty:
        return 0.0

    codes, uniques = pd.factorize(cat)
    counts = np.bincount(codes)
    means = np.zeros_like(counts, dtype=float)
    for idx in range(len(uniques)):
        means[idx] = meas[codes == idx].mean()

    global_mean = meas.mean()
    numerator = np.sum(counts * (means - global_mean) ** 2)
    denominator = np.sum((meas - global_mean) ** 2)
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _mixed_correlation_matrix(
    data: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    *,
    method: str = "pearson",
) -> pd.DataFrame:
    numeric_cols = [col for col in numeric_cols if col in data.columns]
    categorical_cols = [col for col in categorical_cols if col in data.columns]

    num_frame = data[numeric_cols] if numeric_cols else pd.DataFrame(index=data.index)
    corr_num = num_frame.corr(method=method) if not num_frame.empty else pd.DataFrame()

    corr_cat = pd.DataFrame()
    if categorical_cols:
        corr_cat = pd.DataFrame(
            {
                col_a: {
                    col_b: _cramers_v(data[col_a], data[col_b])
                    for col_b in categorical_cols
                }
                for col_a in categorical_cols
            }
        ).astype(float)

    corr_cross = pd.DataFrame()
    if categorical_cols and numeric_cols:
        corr_cross = pd.DataFrame(
            {
                num_col: {
                    cat_col: _correlation_ratio(data[cat_col], data[num_col])
                    for cat_col in categorical_cols
                }
                for num_col in numeric_cols
            }
        ).T.astype(float)

    if corr_cat.empty and corr_num.empty:
        return pd.DataFrame()

    if corr_cat.empty:
        corr = corr_num.copy()
    elif corr_num.empty:
        corr = corr_cat.copy()
    else:
        top = pd.concat([corr_cat, corr_cross.T], axis=1)
        bottom = pd.concat([corr_cross, corr_num], axis=1)
        corr = pd.concat([top, bottom], axis=0)

    if corr.empty:
        return corr
    np.fill_diagonal(corr.values, 1.0)
    return corr


def frobenius_corr_mixed_spearman(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    cat_cols: list[str] | None = None,
) -> tuple[float, int]:
    """Frobenius norm between mixed correlation matrices (Spearman for numeric pairs)."""
    if cat_cols is None:
        cat_cols = []
    valid_cats = [col for col in cat_cols if col in real.columns]

    numeric_cols = [
        col
        for col in real.columns
        if col not in valid_cats and (is_numeric_dtype(real[col]) or is_numeric_dtype(synth[col]))
    ]

    feature_count = len(valid_cats) + len(numeric_cols)
    if feature_count < 2:
        return -1.0, -1

    real_mat = _mixed_correlation_matrix(real, numeric_cols, valid_cats, method="spearman")
    synth_mat = _mixed_correlation_matrix(synth, numeric_cols, valid_cats, method="spearman")

    if real_mat.empty or synth_mat.empty:
        return -1.0, -1

    ordered_cols = list(real_mat.columns)
    synth_mat = synth_mat.reindex(index=ordered_cols, columns=ordered_cols)
    diff = real_mat - synth_mat
    return float(np.linalg.norm(diff.values, ord="fro")), diff.shape[0]

def _discretize_for_kmarginal(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cat_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discretize numerical columns and encode categorical columns for k-marginal TVD calculation.

    Args:
        real_data: DataFrame of real data
        synthetic_data: DataFrame of synthetic data
        cat_cols: List of categorical column names

    Returns:
        Tuple of discretized real and synthetic DataFrames
    """
    numeric_features = [
        col for col in real_data.columns
        if col not in cat_cols
        and real_data[col].nunique() >= 20
        and real_data[col].dtype.kind in ("f", "i", "u")
    ]

    real_binned = real_data.copy()
    syn_binned = synthetic_data.copy()

    # Process numeric features (discretize)
    for col in numeric_features:
        edges = _quantile_edges_from_real(real_data[col], 20)
        if edges.size < 2:
            real_binned[col] = -1
            syn_binned[col] = -1
            continue

        real_binned[col] = _digitize_with_edges(real_data[col], edges)
        syn_binned[col] = _digitize_with_edges(synthetic_data[col], edges)

    # Process categorical features (encode to integers)
    for col in cat_cols:
        # Get all unique values from both datasets
        all_unique_values = list(set(real_data[col].unique()) | set(synthetic_data[col].unique()))
        all_unique_values = [v for v in all_unique_values if pd.notna(v)]  # Remove NaN values
        all_unique_values.sort()  # Sort for consistency

        # Create mapping
        value_to_int = {val: i for i, val in enumerate(all_unique_values)}

        # Apply mapping to real data
        real_binned[col] = real_data[col].map(value_to_int).fillna(-1).astype(int)

        # Apply mapping to synthetic data
        syn_binned[col] = synthetic_data[col].map(value_to_int).fillna(-1).astype(int)

    all_cols = numeric_features + cat_cols
    return real_binned[all_cols].astype(int), syn_binned[all_cols].astype(int)

def calculate_kmarginal_tvd(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cat_cols: list[str] | None = None,
    k: int = 2,
    random_seed: int | None = None
) -> float:
    """Calculate k-marginal Total Variation Distance.

    Args:
        real_data: DataFrame of real data
        synthetic_data: DataFrame of synthetic data
        cat_cols: List of categorical column names
        k: Order of marginals to compute
        random_seed: Random seed for reproducible sampling when > 1000 marginals

    Returns:
        Mean TVD score
    """
    if cat_cols is None:
        cat_cols = []

    real_processed, syn_processed = _discretize_for_kmarginal(real_data, synthetic_data, cat_cols)
    features = real_processed.columns.tolist()

    if len(features) < k: return 1.0

    marginals = list(itertools.combinations(sorted(features), k))
    if len(marginals) > 1000:
        # Set seed for reproducible sampling if provided
        if random_seed is not None:
            random.seed(random_seed)
        marginals = random.sample(marginals, 1000)

    if not marginals: return 1.0

    total_density_diff_sum = 0.0
    for marg in marginals:
        marg = list(marg)
        t_den = real_processed.groupby(marg).size() / len(real_processed)
        s_den = syn_processed.groupby(marg).size() / len(syn_processed)
        # t_den and s_den are already Series with groupby().size()
        if not (isinstance(t_den, pd.Series) and isinstance(s_den, pd.Series)):
            continue
        if not (t_den.dtype.kind in ("f", "i") and s_den.dtype.kind in ("f", "i")):
            continue
        t_den, s_den = t_den.align(s_den, fill_value=0)
        abs_den_diff = t_den.subtract(s_den).abs()
        total_density_diff_sum += float(abs_den_diff.sum())
    if len(marginals) == 0:
        return 1.0
    mean_total_density_diff = float(total_density_diff_sum) / len(marginals)
    mean_tvd = mean_total_density_diff / 2.0
    return float(mean_tvd)

def calculate_nnaa(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cat_cols: list[str] | None = None,
    random_seed: int | None = None
) -> dict[str, float]:
    """Calculate Nearest Neighbour Adversarial Accuracy using SynthEval.

    Args:
        real_data: DataFrame of real data
        synthetic_data: DataFrame of synthetic data
        cat_cols: List of categorical column names
        random_seed: Random seed for reproducible calculations (unused here, for API consistency)

    Returns:
        Dict with 'nnaa' score
    """
    if cat_cols is None:
        cat_cols = []

    evaluator = SynthEval(real_data, cat_cols=cat_cols)
    original_savefig = plt.savefig
    plt.savefig = lambda *args, **kwargs: None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            evaluator.evaluate(
                synthetic_data,
                "",
                nnaa={"n_resample": 30}
            )
    finally:
        plt.savefig = original_savefig

    try:
        val = float(evaluator._raw_results["nnaa"]["avg"])
    except Exception:
        val = -1.0
    return {"nnaa": val}

class FaithfulDataEvaluator:
    """Wrapper class for comprehensive synthetic data evaluation."""
    def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        categorical_columns: list[str] | None = None,
        k_for_kmarginal: int = 2,
        random_seed: int | None = None
    ) -> dict[str, float]:
        """Evaluate synthetic data quality using available metrics.

        Args:
            real_data: DataFrame of real data
            synthetic_data: DataFrame of synthetic data
            categorical_columns: List of categorical column names
            k_for_kmarginal: Order of marginals for k-marginal TVD
            random_seed: Random seed for reproducible metric calculations

        Returns:
            Dict containing calculated metric scores:
            - correlation_matrix_difference
            - k_marginal_tvd
            - nnaa
        """
        if categorical_columns is None:
            categorical_columns = []

        results = {}

        # Calculate correlation_matrix_difference (Frobenius norm of mixed Spearman correlation matrix)
        try:
            mixed_spear_fro, _ = frobenius_corr_mixed_spearman(
                real_data, synthetic_data, categorical_columns
            )
            results["correlation_matrix_difference"] = mixed_spear_fro
        except Exception:
            results["correlation_matrix_difference"] = -1.0

        # Calculate nnaa using SynthEval
        evaluator = SynthEval(real_data, cat_cols=categorical_columns, verbose=False)
        original_savefig = plt.savefig
        plt.savefig = lambda *args, **kwargs: None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                evaluator.evaluate(
                    synthetic_data, "",
                    nnaa={"n_resample": 30}
                )

                try:
                    nnaa_val = float(evaluator._raw_results["nnaa"]["avg"])
                    results["nnaa"] = nnaa_val
                except Exception:
                    results["nnaa"] = -1.0
        finally:
            plt.savefig = original_savefig

        # K-marginal TVD (custom implementation)
        results["k_marginal_tvd"] = calculate_kmarginal_tvd(
            real_data, synthetic_data, categorical_columns, k=k_for_kmarginal, random_seed=random_seed
        )

        return results
