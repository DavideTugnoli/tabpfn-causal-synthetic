# IIT_simglucose

## Overview

This repository explores the application of **Interchange Intervention Training (IIT)** for enhancing the interpretability and predictive performance of neural networks in the context of **blood glucose prediction** for **Type 1 Diabetes Mellitus (T1DM)** patients. By leveraging causal structures from the **simglucose** simulator, the study demonstrates how IIT can impose expert knowledge onto machine learning models, improving their reliability in healthcare settings.

## Key Features
- Utilization of the **simglucose simulator** (UVA/Padova T1DM simulator) for modeling glucose-insulin dynamics.
- Application of **IIT** to enforce causal relationships within a **Multi-Layer Perceptron (MLP)** model.
- Comparative analysis between **IIT-trained models** and **standard-trained models** across different **prediction horizons (PHs)**.
- Detailed **counterfactual loss (LINT)** tracking to evaluate causal abstraction effectiveness.

## Repository Structure
- **/data**: Contains the in-silico patient data used for training and validation.
- **/data**: Contains the in-silico patient dataset used for training and validation.
- **/models**: Includes the MLP architectures implemented for the experiments.
- **/results**: Stores the output metrics and visualizations from the experiments.

## Data

The dataset comprises **in-silico data** from **30 T1DM patients** (10 children, 10 adolescents, 10 adults) generated using the UVA/Padova T1DM joint distributions. To enhance model generalization, an additional **200 synthetic patients** were generated reflecting on `sampled_insilico_quest.csv` and `sampled_insilico_vparams.csv` files.

### Preprocessing Steps:
- Selection of **9 dynamic patient state variables** and **2 variables** for insulin and carbohydrate intake.
- **Z-score standardization** applied to blood glucose values.
- **80/20 split** for training and validation, with the original 30 FDA-approved patients used as the test set.

## Methodology

### Causal Model: simglucose

The **simglucose** simulator models physiological processes in T1DM patients. An additional **amended version** of the simulator was used to ensure acyclicity, aligning with the non-recurrent neural network architecture.

### Neural Network Architectures
1. **MLP Tree**: Mimics the causal structure by aligning each module to specific patient state parameters.
2. **MLP Parallel**: Baseline model with independent modules, lacking causal connections.
3. **MLP Joint**: Addresses cyclic dependencies by merging related modules.

### Interchange Intervention Training (IIT)

IIT imposes causal structures on neural networks using **counterfactual reasoning**. By swapping specific parts of the input and comparing outcomes, IIT guides the model to align with the underlying causal mechanism. The **LINT (counterfactual loss)** metric quantifies the effectiveness of this alignment.

## How to Run Experiments

### Running with `main_batch.py`

The primary script for executing experiments is **`main_batch.py`**. This script allows for flexible configuration of model training parameters.

#### Basic Command:
```bash
python main_batch.py
```

#### Example:
To train the **MLP Tree 256** model with **IIT** and seed **256**, run:
```bash
python main_MLP --model tree --neuro_mapping train_config/MLP_tree.nm --seed 56 --pred_horizon 30
```
To train the **MLP Tree** model with **Standard** and a hidden size of **256**, run:
```bash
python main_MLP --model tree --seed 56 --pred_horizon 30
```
## Experiments and Results

### Performance Metrics:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Percentage of predictions in clinically acceptable EGA classes A and B**

### Key Findings:
1. **IIT-trained models consistently outperformed** standard models across all prediction horizons (30, 45, 60, 120 minutes post-meal) for MLP tree (256) using the **amended** (acyclic) simglucose version.
2. The **MLP Tree model with 256 hidden units** showed significant improvements in RMSE, especially at shorter prediction horizons.
3. **LINT values decreased** during training, indicating successful causal abstraction.


## Limitations and Future Work

- **Outdated simglucose version (S2008)**: Future studies should leverage more recent versions (S2017) for improved clinical relevance.
- **Model simplicity**: Incorporating **recurrent architectures** like LSTM or DRNN could better capture time-dependent relationships.
- **Synthetic data limitations**: Testing on real-world datasets (e.g., Ohio dataset) is recommended.

## Conclusion

This study demonstrates the potential of **Interchange Intervention Training** in improving both the **predictive accuracy** and **interpretability** of neural networks for **blood glucose prediction** in **T1DM patients**. By embedding expert knowledge through causal structures, IIT offers a promising approach for developing reliable, interpretable models in high-stakes healthcare applications.

## References
- Jinyu Xie. Simglucose v0.2.1 (2018) [Online]. Available: https://github.com/jxx123/simglucose. Accessed on: 02-14-2025.
- Geiger et al. (2021). Inducing Causal Structure for Interpretable Neural Networks.
- Liu et al. (2023). Machine learning models for blood glucose level prediction in patients with diabetes mellitus: Systematic review and network meta-analysis.


