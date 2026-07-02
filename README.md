# Improving TabPFN's Synthetic Data Generation by Integrating Causal Structure

This repository contains the code and experimental results for the paper **"Improving TabPFN's Synthetic Data Generation by Integrating Causal Structure"**.

## Overview

TabPFN is an autoregressive foundation model for tabular data that generates features sequentially, conditioning each on previously generated ones. When feature order conflicts with causal structure, this conditioning induces spurious correlations that impair synthetic data quality and causal effect preservation.

This project integrates causal knowledge into TabPFN's generation process through two approaches:
- **Directed Acyclic Graph (DAG)-aware generation**: Conditions each variable only on its causal parents
- **Completed Partially Directed Acyclic Graph (CPDAG)-based generation**: Handles partial causal knowledge when the complete DAG is unknown

We evaluate these methods on synthetic data quality (Correlation Matrix Difference (CMD), k-Marginal Total Variation Distance (kMTVD, $k=2$), Nearest-Neighbor Adversarial Accuracy (NNAA)) and Average Treatment Effect (ATE) preservation.

## Experimental Conditions

We compare the following generation strategies:
- **Vanilla TabPFN**: With *original*, *topological*, and *reverse topological* orderings
- **DAG-aware TabPFN**: Using ground-truth DAG constraints
- **CPDAG-aware TabPFN**: Using minimal or discovered CPDAG constraints

## Repository Structure

The core experimental code is located in `causal_experiments/`.

### Experiment Scripts

*   **`causal_experiments/csuite_experiment/`**
    *   Experiments on **CSuite** benchmark datasets.
    *   `comparison_experiment_csuite/`: Distribution metrics comparison.
    *   `intervention_experiment_csuite/`: ATE preservation evaluation.

*   **`causal_experiments/custom_scm_experiment/`**
    *   Experiments on **Custom Structural Causal Model (SCM)** (collider structure).
    *   `comparison_experiment/`: Distribution metrics comparison.
    *   `interventional_experiment/`: ATE preservation evaluation.

*   **`causal_experiments/real_dataset_simglucose/`**
    *   Experiments on **SimGlucose**, derived from an FDA-approved Type 1 Diabetes simulator.
    *   `acyclic_scm_simglucose_complete/`: Comparison and interventional experiments.

### Results

*   **`causal_experiments/results/comparison_experiment/`**: Distribution-metric results (cleaned CSVs, forest plots, statistics), including the `|NNAA - 0.5|` builds (`nnaa_distance/`) and the NNAA train-versus-test analysis (`nnaa_test_train/`).
*   **`causal_experiments/results/interventional_experiment/`**: ATE-preservation results, including the outcome-scale reference tables (`ate_outcome_scale/`).
*   **`causal_experiments/results/additional_experiments/`**: Appendix experiments (REX discovery, PC-recoverable Custom SCM variant, random-order sensitivity, conditional-independence preservation, external-baseline tables).
*   **`causal_experiments/results/correlations/`**: Spurious-correlation tables for the Custom SCM.
*   **`external_baselines/`**: External tabular generators (TabularARGN, CausalDiffTab, CTGAN, DATGAN, DECAF) with a shared evaluation protocol and final result bundles.
*   Running experiments generates outputs in each experiment's `results/` subdirectory.

## Implementation

The core implementation of DAG-aware and CPDAG-aware generation methods is located in **`src/tabpfn_extensions/unsupervised/unsupervised.py`**. This file contains:

*   **DAG-aware generation**: Implementation of topological ordering and parent-based conditioning using Directed Acyclic Graphs (see `_get_generation_order` and `generate_synthetic_data` methods)
*   **CPDAG-aware generation**: Implementation of CPDAG parsing, ordering strategies, and handling of partially directed causal structures (see `_get_cpdag_original_ordering` and related methods)
*   **Integration with TabPFN**: Modifications to TabPFN's autoregressive generation process to respect causal constraints

The `TabPFNUnsupervisedModel` class in this file extends TabPFN's capabilities to support causal structure-aware synthetic data generation, which is the main contribution of this work.

## Usage

Experiments are designed to be run using `uv`.

### Comparison Experiments (Distribution Metrics)

**CSuite:**
```bash
uv run causal_experiments/csuite_experiment/comparison_experiment_csuite/csuite_comparison_experiment.py
```

**Custom SCM:**
```bash
uv run causal_experiments/custom_scm_experiment/comparison_experiment/comparison_experiment.py
```

**SimGlucose:**
```bash
uv run causal_experiments/real_dataset_simglucose/acyclic_scm_simglucose_complete/comparison_experiment/comparison_experiment.py
```

### Interventional Experiments (ATE Preservation)

**CSuite:**
```bash
uv run causal_experiments/csuite_experiment/intervention_experiment_csuite/intervention_experiment_csuite.py
```

**Custom SCM:**
```bash
uv run causal_experiments/custom_scm_experiment/interventional_experiment/interventional_experiment.py
```

**SimGlucose:**
```bash
uv run causal_experiments/real_dataset_simglucose/acyclic_scm_simglucose_complete/interventional_experiment/interventional_experiment.py
```

### Options

Most scripts support:
*   `--test`: Test mode (fewer repetitions, smaller data)
*   `--algorithm [vanilla|dag|...]`: Run specific algorithm only
*   `--dataset [name]`: Select specific dataset (where applicable)
*   `--save-synthetic`: Save synthetic datasets to disk (required for spurious correlation analysis)

### Spurious Correlation Analysis

The script `causal_experiments/results/correlations/analyze_spurious_correlations.py` generates the spurious correlation table shown in the paper's appendix. This analysis requires:

1. **Synthetic datasets**: Run the comparison experiment for the custom SCM with `--save-synthetic` flag:
   ```bash
   uv run causal_experiments/custom_scm_experiment/comparison_experiment/comparison_experiment.py --save-synthetic
   ```
   This will generate the synthetic data files (`.npz`) and add `synthetic_data_path` and `test_dataset_path` columns to the results CSV.

2. **Run the analysis**:
   ```bash
   uv run causal_experiments/results/correlations/analyze_spurious_correlations.py
   ```
   This will generate `spurious_correlations_summary.csv` containing the table used in the paper's appendix.

**Note**: The synthetic datasets are large (~6GB total) and are not included in this repository. They must be regenerated by running the experiments with the `--save-synthetic` flag.

## Datasets

### CSuite Benchmark
We evaluate on six CSuite datasets: `csuite_large_backdoor`, `csuite_mixed_confounding`, `csuite_mixed_simpson`, `csuite_nonlin_simpson`, `csuite_symprod_simpson`, and `csuite_weak_arrows`. These datasets cover various causal structures including colliders, backdoors, and confounding scenarios.

### Custom SCM
A controlled collider structure (X₄ → X₃ → X₂ ← X₁) optimized to test spurious correlation detection. The structure achieves 0.9292 conditional correlation for collider bias testing.

### SimGlucose
Real-world dataset derived from the FDA-approved UVA/Padova Type 1 Diabetes simulator, converted to an acyclic SCM structure for causal-aware generation evaluation.

## Evaluation Metrics

### Distribution Quality Metrics
- **Correlation Matrix Difference (CMD)**: Frobenius norm of the difference between correlation matrices
- **k-Marginal Total Variation Distance (kMTVD)**: Total variation distance for k-way marginals (k=2)
- **Nearest-Neighbor Adversarial Accuracy (NNAA)**: Adversarial accuracy using nearest-neighbor classifier

### Causal Effect Preservation
- **Average Treatment Effect (ATE) Difference**: Absolute difference between true and synthetic ATE estimates

## Requirements

- Python ≥3.9
- `uv` package manager (recommended) or `pip`
- See `pyproject.toml` for full dependency list

### Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Reproducibility

All experiments use fixed random seeds for reproducibility. Results are averaged over 100 repetitions per configuration. The consolidated results used in the paper are available in `causal_experiments/results/`:
- **`comparison_experiment/data/`**: Raw CSV results for distribution quality metrics
- **`interventional_experiment/data/`**: Raw CSV results for ATE preservation
- **`comparison_experiment/forest_plots/paper/`**: Forest plots (PDF) and summary CSVs
- **`interventional_experiment/forest_plots/paper/`**: Forest plots (PDF) and summary CSVs

## Acknowledgments

This project builds upon [tabpfn-extensions](https://github.com/PriorLabs/tabpfn-extensions) by PriorLabs.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.