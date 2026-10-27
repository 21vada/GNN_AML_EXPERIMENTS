# AML Graph Neural Network (GNN) Pipeline & Explainability Engine

An end-to-end machine learning framework for detecting complex anti-money laundering (AML) typologies in temporal transaction graphs. This repository integrates **PyTorch Geometric (PyG)** edge-level GNNs, **XGBoost** ensembles with out-of-fold stacking, **Neo4j** graph database extraction, and **SHAP** glass-box explainability.

---

## 📌 Executive Summary

Traditional AML transaction monitoring models heavily rely on static tabular attributes (e.g., payment format, transaction currency) that can easily be obfuscated by illicit networks or introduce synthetic dataset artifacts. This pipeline addresses these limitations by:

1. **Stripping Synthetic Fingerprints:** Benchmarking a strict *Structural Feature Set* (pure graph topology, time gaps, velocity, and causal behavior) against the *Full Feature Set* to prevent shortcut learning.
2. **Temporal EdgeGNN Architecture:** Leveraging `TransformerConv` layers on dynamic temporal subgraphs using strictly past transaction history ($t_{edge} < t_{target}$).
3. **Out-of-Fold Stacking & Blending:** Stacking time-forward out-of-fold GNN predictions directly into an XGBoost classifier and Logistic Regression logit blend.
4. **Audit-Grade Explainability:** Applying SHAP hardware-accelerated `TreeExplainer` to visualize macro-feature importances and micro-level "waterfall" decisions for flagged laundering rings.

---

## 🏗️ Architecture & Pipeline Overview

```
                          ┌────────────────────────┐
                          │     Neo4j Database     │
                          └───────────┬────────────┘
                                      │ (Cypher Extraction)
                                      ▼
                          ┌────────────────────────┐
                          │ Temporal CSV / PyG Data│
                          └───────────┬────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
┌──────────────────────────────┐              ┌──────────────────────────────┐
│  Causal Feature Engineering  │              │    Autoencoder Anomaly Score │
│ (Numba-Accelerated Sweeps)   │              │   (Pre-filtering PyG Weight) │
└──────────────┬───────────────┘              └──────────────┬───────────────┘
               │                                             │
               ├──────────────────────┐                      │
               ▼                      ▼                      ▼
┌──────────────────────────────┐  ┌──────────────────────────────────────────┐
│ Tabular XGBoost (Baseline)   │  │ PyG EdgeGNN (TransformerConv + Sampler)  │
└──────────────┬───────────────┘  └──────────────────────┬───────────────────┘
               │                                         │
               │ (Tabular Features)                      │ (GNN Out-of-Fold Score)
               └──────────────────────┬──────────────────┘
                                      ▼
                       ┌────────────────────────────┐
                       │   Stacked OOF XGBoost /    │
                       │   Logistic Logit Blend     │
                       └──────────────┬─────────────┘
                                      ▼
                       ┌────────────────────────────┐
                       │  SHAP Transparency Layer   │
                       │ (Global & Local Waterfall) │
                       └────────────────────────────┘

```

---

## 📂 Repository Layout

| File | Description |
| --- | --- |
| **`aml_addressed_concerns_with_shap.ipynb`** | Primary production Jupyter Notebook (Google Colab A100/GPU ready). Runs data ingestion, causal feature generation, GNN sampling/training, stacked XGBoost ensemble, and SHAP visualization. |
| **`export_csv.py`** | Python script connecting to Neo4j via Bolt to pull temporal transaction edges, account IDs, amounts, timestamps, and backfilled node properties into CSV format. |
| **`graph_extraction.py`** | PyTorch & PyG pipeline that connects to Neo4j, converts raw graph streams to `torch_geometric.data.Data`, trains a reconstruction Autoencoder for anomaly weighting, and sets up PyG link neighbor loaders. |
| **`summary.json`** | Output metrics detailing PR-AUC validation/test scores, precision, recall, F1, and GNN performance gains across structural and full feature sets. |
| **`shap_summary_global_structural.png`** | High-resolution SHAP summary plot ranking feature impacts for the structural model. |
| **`shap_waterfall_local_structural.png`** | Local SHAP waterfall plot demonstrating GNN score contribution to a True Positive Gather-Scatter laundering attempt. |

---

## ⚙️ Core Methodology & Design Choices

### 1. Causal Feature Engineering (Numba-Accelerated)

To eliminate data leakage, every feature is computed strictly using events occurring prior to the current transaction timestamp ($t$).

* **Windowed Aggregations:** Rolling transaction counts and amount sums across 1h, 24h, 96h, and 168h windows for both senders ($s$) and receivers ($r$).
* **Velocity & Pass-Through Ratios:** Ratios capturing rapid incoming-to-outgoing money forwarding (`s_fwd_ratio`, `s_pass_96h`).
* **Counterparty Diversity:** Exact Numba-accelerated two-pointer sweeps calculating unique counterparties over rolling windows (`s_out_nuniq_24h`, `r_in_nuniq_96h`).

### 2. Temporal EdgeGNN Architecture

* **Graph Convolution:** Uses multi-layer `TransformerConv` with BatchNorm and residual skip-connections.
* **Link Neighbor Sampling:** Utilizes PyG's `LinkNeighborLoader` constrained by `time_attr="time"` and `edge_label_time = ts - 1` to strictly restrict neighbor search to past edges.
* **Ego Markers:** Injects binary markers for seed edge source and destination nodes directly into the input embeddings to assist link property prediction.

### 3. Structural vs. Full Feature Evaluation

* **Structural Feature Set (80 Features):** Drops all synthetic artifacts (e.g., payment format, currency types, bank codes, amount equality flags). Forces the model to learn pure graph structure and transaction dynamics.
* **Full Feature Set (106 Features):** Includes transaction metadata (ACH, Wire, Cheque, Currency exchange flags).

---

## 📊 Experimental Results

Benchmarked on the IBM HI-Small temporal dataset (**5,078,345 transactions**, **515,088 accounts**, 60/20/20 chronological train/val/test split).

### Headline Model Performance

| Feature Set | Model Variant | Val PR-AUC | Test PR-AUC | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- | --- | --- |
| **Structural** | Tabular Baseline | 0.5646 | 0.5831 | 0.536 | 0.564 | 0.550 |
| **Structural** | GNN Alone | 0.4013 | 0.4420 | 0.370 | 0.500 | 0.425 |
| **Structural** | **Blend (Tabular + GNN Logits)** | **0.5694** | **0.6095** | **0.583** | **0.573** | **0.578** |
| **Structural** | **Stacked OOF (XGBoost + GNN Score)** | **0.5746** | **0.5670** | **0.508** | **0.572** | **0.538** |
| **Full** | Tabular (Burn-In) | 0.5857 | 0.7019 | 0.799 | 0.609 | 0.691 |
| **Full** | AUDIT (Fingerprints Only) | 0.0113 | 0.0202 | 0.065 | 0.095 | 0.077 |

*Note: The Logistic Regression Blend yields a **+0.0264 PR-AUC lift** on test data over the best purely tabular model on structural features.*

### Detection Recall Across Typologies (Structural Test Set)

| Laundering Typology | Count ($n$) | Tabular Recall | Stacked OOF Recall | Top 1% Alert Recall |
| --- | --- | --- | --- | --- |
| **BIPARTITE** | 74 | 0.51 | 0.49 | 0.84 |
| **CYCLE** | 109 | 0.66 | 0.69 | 0.95 |
| **FAN-IN** | 137 | 0.79 | 0.80 | 0.98 |
| **FAN-OUT** | 140 | 0.92 | 0.93 | 0.99 |
| **GATHER-SCATTER** | 401 | 0.84 | 0.84 | 0.97 |
| **RANDOM** | 86 | 0.52 | 0.50 | 0.94 |
| **SCATTER-GATHER** | 252 | 0.82 | 0.84 | 0.96 |
| **STACK** | 140 | 0.51 | 0.55 | 0.92 |

---

## 🔍 Explainability (SHAP Glass-Box Layer)

### Global Macro View (`shap_summary_global_structural.png`)

The global SHAP summary plot ranks `gnn_score` as the top impact feature across the stacked model, overriding traditional heuristic metrics like `pair_fwd_cnt` and `log_amount`.

### Local Micro View (`shap_waterfall_local_structural.png`)

The local waterfall plot isolates a True Positive **Gather-Scatter** transaction. The GNN Score (`Feature 80`) contributes a massive **+3.43** push to the final decision logit ($f(x) = 6.07$), enabling investigators to trace why complex network topology overrode standard transaction amounts.

---

## 🚀 Setup & Execution Guide

### Prerequisites

* Python 3.10+
* PyTorch 2.1+ with CUDA acceleration
* PyTorch Geometric & `pyg-lib`
* XGBoost, SHAP, Numba, Pandas, Scikit-Learn

### 1. Extracting Data from Neo4j

Ensure your Neo4j instance is running, then execute the temporal CSV export script:

```bash
python export_csv.py

```

Alternatively, run the PyG graph extraction script directly to output a PyG `Data` object with pre-computed autoencoder anomaly scores:

```bash
python graph_extraction.py

```

### 2. Running the Pipeline

To run the full GNN training, out-of-fold stacking, evaluation, and SHAP generation:

```bash
python aml_addressed_concerns_with_shap.py \
  --csv path/to/HI-Small_Trans.csv \
  --patterns path/to/HI-Small_Patterns.txt \
  --out ./aml_outputs \
  --sets structural \
  --stack \
  --boot 200

```

#### Command-Line Arguments

* `--csv`: Path to transaction CSV file.
* `--patterns`: Path to patterns text file (for typology validation).
* `--out`: Directory where results and plots will be saved.
* `--sets`: Feature sets to evaluate (`structural`, `full`).
* `--no-gnn`: Skip GNN training and run tabular models only.
* `--stack`: Execute time-forward out-of-fold stacking and trigger SHAP plot generation.
* `--smoke`: Run on fast synthetic data to test environment setups.
