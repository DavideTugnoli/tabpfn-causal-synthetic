"""CAUSAL DISCOVERY EXPERIMENT WITH P-VALUE ANALYSIS.

Runs causal discovery on continuous and mixed data with detailed p-value reporting.
Allows selecting different conditional independence tests (Fisher-Z, KCI, G^2, ...).
"""
from __future__ import annotations

import io
import random
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import CIT, CIT_Base, register_ci_test
from causallearn.utils.GraphUtils import GraphUtils
from scipy.special import digamma
from scipy.spatial import cKDTree
from scipy.stats import chi2_contingency, pearsonr
from sklearn.preprocessing import KBinsDiscretizer

# Add the causal_experiments directory to the path for absolute imports
current_dir = Path(__file__).parent
causal_experiments_dir = current_dir.parent
sys.path.insert(0, str(causal_experiments_dir))

# Set random seeds for reproducibility
_rng = np.random.default_rng(42)
random.seed(42)

def get_pairwise_pvalues(
    data: np.ndarray,
    col_names: list[str],
    _categorical_cols: list[int],
    test_type: str = "fisherz",
    _alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute pairwise p-values between all variables.

    Returns:
        DataFrame with p-values for each variable pair
    """
    n_vars = data.shape[1]
    p_values = np.ones((n_vars, n_vars))

    # Initialize CIT object
    cit = CIT(data, test_type)

    # Compute pairwise tests
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            # Test independence between i and j
            p_value = cit(i, j, [])  # type: ignore[call-arg]
            p_values[i, j] = p_value
            p_values[j, i] = p_value

    # Create DataFrame for better display
    p_values_df = pd.DataFrame(p_values, index=col_names, columns=col_names)
    return p_values_df


def _discretize_for_gsq(data: np.ndarray, categorical_cols: list[int], n_bins: int = 8) -> np.ndarray:
    continuous_indices = [i for i in range(data.shape[1]) if i not in categorical_cols]
    discretized = data.copy()
    if continuous_indices:
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        discretized[:, continuous_indices] = discretizer.fit_transform(data[:, continuous_indices])
    return discretized.astype(int)


def _build_ci_callable(
    data: np.ndarray,
    categorical_cols: list[int],
    indep_test: str,
    alpha: float,
    hybrid_params: dict[str, Any] | None = None,
    random_state: int | None = 42,
):
    test_lower = indep_test.lower()
    if test_lower == "hybrid":
        params = hybrid_params.copy() if hybrid_params else {}
        params.setdefault("random_state", random_state)
        return HybridCIT(
            data,
            categorical_cols=categorical_cols,
            ci_alpha=alpha,
            **params,
        )

    if test_lower in {"fisherz", "kci", "chisq", "gsq"}:
        if test_lower in {"chisq", "gsq"}:
            categorical_cols = sorted(set(categorical_cols))
            discrete_data = _discretize_for_gsq(data, categorical_cols)
            return CIT(discrete_data, method=test_lower)
        return CIT(data, method=test_lower)

    raise ValueError(f"Unsupported independence test '{indep_test}' for analysis")


def analyze_edge_discovery(
    data: np.ndarray,
    true_dag: dict[int, list[int]],
    col_names: list[str],
    categorical_cols: list[int],
    cg: Any,
    indep_test: str,
    alpha: float,
    hybrid_params: dict[str, Any] | None = None,
) -> None:
    """Analyze why certain edges were or weren't discovered using actual separating sets."""

    sepsets = getattr(cg, "sepset", None)
    if sepsets is None and hasattr(cg, "G"):
        sepsets = getattr(cg.G, "sepset", None)

    if sepsets is None:
        print("[analyze_edge_discovery] Separation sets unavailable; skipping analysis.")
        return

    tester = _build_ci_callable(
        data,
        categorical_cols,
        indep_test,
        alpha,
        hybrid_params=hybrid_params,
    )

    for child_idx, parent_indices in true_dag.items():
        for parent_idx in parent_indices:
            sep = sepsets[child_idx][parent_idx]
            if not sep:
                sep = sepsets[parent_idx][child_idx]
            if sep is None:
                sep = []
            sepset_indices = [int(s) for s in sep]

            try:
                p_value = tester(parent_idx, child_idx, sepset_indices)
            except Exception as exc:  # pragma: no cover
                print(
                    f"[analyze_edge_discovery] Failed to evaluate edge ({col_names[parent_idx]} → {col_names[child_idx]}): {exc}"
                )
                continue

            status = "kept" if cg.G.graph[parent_idx, child_idx] != 0 or cg.G.graph[child_idx, parent_idx] != 0 else "removed"
            print(
                f"Edge {col_names[parent_idx]} ↔ {col_names[child_idx]} ({status}) | "
                f"sepset={ [col_names[i] for i in sepset_indices] } | p-value={p_value:.4g}"
            )


def discover_cpdag(
    data: np.ndarray,
    alpha: float,
    indep_test: str,
    *,
    return_pvalues: bool = False,
    categorical_cols: list[int] | None = None,
    hybrid_params: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any | None:
    """Runs PC algorithm with optional p-value collection."""
    try:
        pc_kwargs = dict(
            alpha=alpha,
            show_progress=False,
            stable=True,          # PC-stable (order-independence in skeleton)
            uc_rule=1,            # 'maxP' rule for collider orientation
        )
        pc_kwargs.update(kwargs)

        if indep_test == "hybrid":
            categorical_cols = categorical_cols or []
            params = (hybrid_params or {}).copy()
            ci_kwargs = dict(params)
            ci_kwargs.setdefault("ci_alpha", alpha)
            ci_kwargs.pop("alpha", None)
            ci_kwargs["categorical_cols"] = categorical_cols
            cg = pc(
                data,
                indep_test="hybrid",
                **pc_kwargs,
                **ci_kwargs,
            )
        elif return_pvalues:
            cg = pc(
                data,
                indep_test=indep_test,
                verbose=True,
                **pc_kwargs,
            )
        else:
            cg = pc(
                data,
                indep_test=indep_test,
                **pc_kwargs,
            )
        return cg
    except (ValueError, RuntimeError, ImportError):
        return None

def plot_graphs(
    true_dag_def: dict[int, list[int]] | dict[str, list[str]],
    discovered_cg: Any,
    col_names: list[str],
    filename: str | Path,
) -> None:
    """Creates comparison plot between true and discovered graphs."""
    # True Graph
    dot_true = pydot.Dot(
        graph_type="digraph",
        label="True Causal Graph",
        fontsize=20,
        labeljust="t",
    )
    for name in col_names:
        dot_true.add_node(pydot.Node(name))
    
    # Handle both index-based and name-based DAGs
    if true_dag_def:
        first_key = next(iter(true_dag_def.keys()))
        uses_indices = isinstance(first_key, int)
        
        if uses_indices:
            # Index-based DAG: {0: [1, 2], 1: [3], ...}
            for child_idx, parent_indices in true_dag_def.items():
                for parent_idx in parent_indices:
                    parent_idx_int = int(parent_idx)
                    child_idx_int = int(child_idx)
                    if parent_idx_int >= len(col_names) or child_idx_int >= len(col_names):
                        print('ERROR: index out of range!', parent_idx_int, child_idx_int)
                        continue
                    dot_true.add_edge(
                        pydot.Edge(col_names[parent_idx_int], col_names[child_idx_int])
                    )
        else:
            # Name-based DAG: {"X1": ["X2", "X3"], "X2": ["X4"], ...}
            for child_name, parent_names in true_dag_def.items():
                for parent_name in parent_names:
                    dot_true.add_edge(
                        pydot.Edge(parent_name, child_name)
                    )

    # Discovered Graph
    dot_discovered = GraphUtils.to_pydot(discovered_cg.G, labels=col_names)
    # Cast to Any to handle pydot method access
    dot_discovered_any = cast("Any", dot_discovered)
    dot_discovered_any.set_label("Discovered Graph (CPDAG)")
    dot_discovered_any.set_fontsize(20)
    dot_discovered_any.set_labeljust("t")

    # Create image
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # Cast to Any to handle pydot method access
    dot_true_any = cast("Any", dot_true)
    axes[0].imshow(plt.imread(io.BytesIO(dot_true_any.create_png(prog="dot"))))
    axes[0].axis("off")
    axes[1].imshow(plt.imread(io.BytesIO(dot_discovered_any.create_png(prog="dot"))))
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def run_mixed_data_discovery(
    data: np.ndarray,
    col_names: list[str],
    categorical_cols: list[int],
    alpha: float,
    *,
    indep_test: str = "gsq",
) -> tuple[Any, str]:
    """Deterministic, single-pass mixed-data discovery (fast).

    Policy:
    - Discretize continuous variables into quantiles (8 bins, encode=ordinal)
    - Use G^2 independence test (gsq)
    """
    n_vars = data.shape[1]
    # Remove unused variable warning
    _ = col_names

    continuous_indices = [i for i in range(n_vars) if i not in categorical_cols]

    if continuous_indices:
        discretized_data = data.copy()
        discretizer = KBinsDiscretizer(n_bins=8, encode="ordinal", strategy="quantile")
        discretized_data[:, continuous_indices] = discretizer.fit_transform(
            data[:, continuous_indices]
        )
        discretized_data = discretized_data.astype(int)
    else:
        discretized_data = data.astype(int)

    cg = discover_cpdag(discretized_data, alpha=alpha, indep_test=indep_test)
    if cg is None:
        # Fast fallback: empty CPDAG
        return (
            np.zeros((n_vars, n_vars)),
            f"Deterministic discretized (8 bins) + {indep_test.upper()} failed",
        )
    return cg, f"Deterministic discretized (8 bins) + {indep_test.upper()}"


def run_causal_discovery_experiment(
    include_categorical: bool,
    *,
    alpha: float = 0.05,
    n_samples: int = 1000,
) -> None:
    """Runs causal discovery experiment with detailed analysis."""
    # NOTE: This function is disabled due to missing imports
    # It was only used for standalone testing, not for our discovery pipeline
    raise NotImplementedError("This function requires imports that are not available")
    

def run_pc_discovery_on_dataset(
    dataset_name: str,
    data: np.ndarray,
    col_names: list[str],
    categorical_cols: list[int],
    true_dag: dict[int, list[int]] | None = None,
    *,
    task_type: str = "unsupervised",
    target_column: str | None = None,
    verbose: bool = False,
    output_dir: str | Path | None = None,
    alpha: float = 0.05,
    indep_test: str | None = None,
    hybrid_params: dict[str, Any] | None = None,
) -> Any:
    """Run PC algorithm on a dataset and return the discovered CPDAG.

    Args:
        dataset_name: Name of the dataset ("mixed" or "continuous")
        data: Data array for causal discovery
        true_dag: True DAG structure (for validation/comparison). If None, no comparison is made.
        col_names: List of column names (must match data)
        categorical_cols: List of categorical column indices
        task_type: Type of task (unused, kept for compatibility)
        target_column: Target column name (unused, kept for compatibility)
        verbose: Whether to print detailed output
        output_dir: Output directory for plots (if None, no plots saved)
        alpha: Significance level for PC algorithm (default: 0.05)
        indep_test: Optional independence test identifier (e.g., "fisherz", "kci", "gsq", "hybrid").
            When ``None`` the function defaults to Fisher-Z for continuous datasets and
            the hybrid tester for mixed datasets.

    Returns:
        CausalGraph object (contains the adjacency matrix in .G.graph)
    """
    # Remove unused parameter warnings
    _ = task_type, target_column, output_dir

    include_categorical = dataset_name == "mixed"
    requested_test = (indep_test or "").lower() or None
    if requested_test is None:
        selected_test = "hybrid" if include_categorical and categorical_cols else "fisherz"
    else:
        selected_test = requested_test
    # No fallback to get_dag_and_config: col_names and categorical_cols must be provided

    # Run causal discovery (deterministic fast path)
    if include_categorical and categorical_cols and selected_test == "hybrid":
        params = hybrid_params or {}
        cg = discover_cpdag(
            data,
            alpha=alpha,
            indep_test="hybrid",
            categorical_cols=categorical_cols,
            hybrid_params=params,
        )
    elif include_categorical and categorical_cols and selected_test in {"gsq", "chisq"}:
        # Default or discrete-friendly tests: discretise and run with provided/gsq test
        mixed_test = selected_test
        cg, _method_used = run_mixed_data_discovery(
            data,
            col_names,
            categorical_cols,
            alpha,
            indep_test=mixed_test,
        )
    else:
        Xc = data.copy()
        if selected_test == "fisherz" and Xc.size > 0:
            for j in range(Xc.shape[1]):
                mu = float(np.mean(Xc[:, j]))
                sd = float(np.std(Xc[:, j]))
                if sd > 0:
                    Xc[:, j] = (Xc[:, j] - mu) / sd
        cg = discover_cpdag(
            Xc,
            alpha=alpha,
            indep_test=selected_test,
            categorical_cols=categorical_cols,
            hybrid_params=hybrid_params,
        )

    if cg is None:
        # Return empty CPDAG object
        return np.zeros((len(col_names), len(col_names)))

    if verbose:
        # Count discovered edges
        _discovered_edges = 0
        for i in range(len(col_names)):
            for j in range(len(col_names)):
                if cg.G.graph[i, j] != 0:
                    _discovered_edges += 1

    # Save plot only if output_dir is provided and true_dag is available
    if output_dir is not None and true_dag is not None:
        plot_filename = Path(output_dir) / f"{dataset_name}_discovery_result.png"
        plot_graphs(true_dag, cg, col_names, plot_filename)

    return cg


def main() -> None:
    """Main function."""
    # NOTE: Main function disabled due to missing imports
    print("This script is now used as a library. Use run_pc_discovery_on_dataset() function directly.")


if __name__ == "__main__":
    main()

class HybridCIT(CIT_Base):
    """Hybrid conditional independence tester supporting mixed data.

    Strategy:
    - (discrete, discrete | discrete): G^2 on discretized table
    - (continuous, continuous | continuous): KCI (causal-learn)
    - mixed cases (any of X/Y/Z has mixed types): kNN-based CMI with permutation p-values
      using a jittered & scaled metric space to avoid ties and scale dominance.
    """

    def __init__(
        self,
        data: np.ndarray,
        categorical_cols: list[int] | None = None,
        *,
        ci_alpha: float = 0.05,
        k: int = 5,
        permutations: int = 100,
        random_state: int | None = 42,
        jitter: float = 1e-6,
        cache_path: str | None = None,
        **_: Any,
    ) -> None:
        data_array = np.asarray(data, dtype=float)
        super().__init__(data_array, cache_path=cache_path)

        # Store canonical data representations expected by base helpers
        self.data = data_array
        self.sample_size, self.num_features = self.data.shape
        self.n_samples, self.n_features = self.data.shape
        self.method = "hybrid"

        self.alpha = float(ci_alpha)
        self.k = int(k)
        self.permutations = int(permutations)
        self.rng = np.random.default_rng(random_state)
        self.jitter = float(jitter)

        categorical_indices = categorical_cols or []

        # Indices bookkeeping
        self.categorical_indices = sorted(set(int(i) for i in categorical_indices))
        self.continuous_indices = [i for i in range(self.n_features) if i not in self.categorical_indices]
        self.discrete_index_map: dict[int, int] = {
            old: new for new, old in enumerate(self.categorical_indices)
        }

        # --- Data views for different testers ---

        # 1) Discrete view for G^2
        if self.categorical_indices:
            disc = self.data[:, self.categorical_indices].copy()
            # Ensure valid integer coding for contingency tests
            disc = np.asarray(np.round(disc), dtype=int)
            self.data_discrete = disc
            from causallearn.utils.cit import CIT
            self.cit_gsq = CIT(self.data_discrete, method="gsq")
        else:
            self.data_discrete = None
            self.cit_gsq = None

        # 2) Continuous-standardized view for KCI
        Xc = self.data.copy()
        if self.continuous_indices:
            mu = np.mean(Xc[:, self.continuous_indices], axis=0, keepdims=True)
            sd = np.std(Xc[:, self.continuous_indices], axis=0, keepdims=True)
            sd[sd == 0] = 1.0
            Xc[:, self.continuous_indices] = (Xc[:, self.continuous_indices] - mu) / sd
        self.data_kci = Xc
        from causallearn.utils.cit import CIT
        self.cit_kci = CIT(self.data_kci, method="kci")

        # 3) Jittered & scaled view for kNN–CMI (used only in mixed cases)
        Xn = self.data.copy()
        # z-score all columns to avoid scale dominance
        mu_all = np.mean(Xn, axis=0, keepdims=True)
        sd_all = np.std(Xn, axis=0, keepdims=True)
        sd_all[sd_all == 0] = 1.0
        Xn = (Xn - mu_all) / sd_all

        # add tiny jitter ONLY to categorical columns to break ties
        if self.categorical_indices:
            noise = self.rng.uniform(-self.jitter, self.jitter, size=(self.n_samples, len(self.categorical_indices)))
            Xn[:, self.categorical_indices] += noise

        self.data_knn = Xn

        # simple memoization
        self.knn_cache: dict[tuple[int, int, tuple[int, ...]], float] = {}

    # ---- public call ----
    def __call__(self, x_idx: int, y_idx: int, s_indices: list[int] | tuple[int, ...]) -> float:
        S = tuple(sorted(int(s) for s in s_indices))
        x_discrete = x_idx in self.discrete_index_map
        y_discrete = y_idx in self.discrete_index_map
        S_discrete = all(s in self.discrete_index_map for s in S)
        x_cont = x_idx in self.continuous_indices
        y_cont = y_idx in self.continuous_indices
        S_cont = all(s in self.continuous_indices for s in S)

        if x_discrete and y_discrete and S_discrete:
            return self._gsq_test(x_idx, y_idx, S)
        if x_cont and y_cont and S_cont:
            return self.cit_kci(x_idx, y_idx, list(S))
        return self._knn_cmi_pvalue(x_idx, y_idx, S)

    # ---- discrete branch ----
    def _gsq_test(self, x_idx: int, y_idx: int, S: tuple[int, ...]) -> float:
        if self.cit_gsq is None:
            raise RuntimeError("G^2 test requested but no categorical variables were registered.")
        x_new = self.discrete_index_map[x_idx]
        y_new = self.discrete_index_map[y_idx]
        S_new = [self.discrete_index_map[s] for s in S]
        return float(self.cit_gsq(x_new, y_new, S_new))

    # ---- mixed branch: kNN–CMI with permutation p-value ----
    def _knn_cmi_pvalue(self, x_idx: int, y_idx: int, S: tuple[int, ...]) -> float:
        key = (x_idx, y_idx, S)
        if key in self.knn_cache:
            return self.knn_cache[key]

        X = self.data_knn[:, [x_idx]]
        Y = self.data_knn[:, [y_idx]]
        Z = self.data_knn[:, list(S)] if S else None

        I_obs = self._estimate_cmi_knn(X, Y, Z)

        # permutation p-value on Y | (X,Z), with early stopping
        greater = 1
        total = 1
        for _ in range(self.permutations):
            perm = self.rng.permutation(self.n_samples)
            I_perm = self._estimate_cmi_knn(X, Y[perm], Z)
            total += 1
            if I_perm >= I_obs - 1e-9:
                greater += 1
            if greater / total > self.alpha + 0.01:
                break

        pval = float(min(1.0, max(1e-12, greater / total)))
        self.knn_cache[key] = pval
        return pval

    # ---- KSG-style estimators (sup-norm radius) ----
    def _estimate_cmi_knn(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray | None) -> float:
        if Z is None or Z.size == 0:
            return self._estimate_mi_knn(X, Y)

        XZ = np.concatenate([X, Z], axis=1)
        YZ = np.concatenate([Y, Z], axis=1)
        XYZ = np.concatenate([X, Y, Z], axis=1)

        tree_xyz = cKDTree(XYZ)
        distances, _ = tree_xyz.query(XYZ, k=self.k + 1, p=np.inf)
        eps = distances[:, -1] - 1e-12
        eps[eps <= 0] = 1e-12

        tree_xz = cKDTree(XZ)
        tree_yz = cKDTree(YZ)
        tree_z = cKDTree(Z)

        nx = self._count_neighbors(tree_xz, XZ, eps)
        ny = self._count_neighbors(tree_yz, YZ, eps)
        nz = self._count_neighbors(tree_z, Z, eps)

        value = digamma(self.k) - np.mean(digamma(nx + 1) + digamma(ny + 1) - digamma(nz + 1))
        return float(max(0.0, value))

    def _estimate_mi_knn(self, X: np.ndarray, Y: np.ndarray) -> float:
        XY = np.concatenate([X, Y], axis=1)
        tree_xy = cKDTree(XY)
        distances, _ = tree_xy.query(XY, k=self.k + 1, p=np.inf)
        eps = distances[:, -1] - 1e-12
        eps[eps <= 0] = 1e-12

        tree_x = cKDTree(X)
        tree_y = cKDTree(Y)

        nx = self._count_neighbors(tree_x, X, eps)
        ny = self._count_neighbors(tree_y, Y, eps)

        value = digamma(self.k) + digamma(len(X)) - np.mean(digamma(nx + 1) + digamma(ny + 1))
        return float(max(0.0, value))

    @staticmethod
    def _count_neighbors(tree: cKDTree, points: np.ndarray, radii: np.ndarray) -> np.ndarray:
        counts = np.empty(len(points), dtype=int)
        for idx, (point, radius) in enumerate(zip(points, radii)):
            neighbors = tree.query_ball_point(point, radius, p=np.inf)
            counts[idx] = max(0, len(neighbors) - 1)
        return counts


# Register the hybrid tester so causallearn can instantiate it via string identifier
register_ci_test("hybrid", HybridCIT)
