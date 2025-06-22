# Breast Cancer Subtype Classification - TCGA-BRCA

This project uses machine learning to classify breast cancer subtypes from clinical data provided by the TCGA-BRCA dataset. The goal is to support diagnosis and treatment planning by predicting subtypes such as **Luminal A**, **Luminal B**, **HER2-enriched**, and **Triple-Negative**.

---

## 🔍 Project Overview

Breast cancer is highly heterogeneous, and proper subtype classification is critical for tailoring treatment. This project leverages clinical variables (age, gender, gene receptor status) and supervised learning to predict subtypes.

**Data Pipeline Overview:**
- Raw data: `data.json` — Raw TCGA clinical dataset
- Cleaned: `filtered_data.csv` — Preprocessed data ready for model training
- Models used: Logistic Regression, SVC, Random Forest, K-Nearest Neighbors
- Visuals: Age & subtype distributions, model performance, feature importances, confusion matrix

---

## 🧩 Project Modules

### 📁 `data_services.py`
- Loads and filters the raw JSON file into structured clinical data.
- Exports usable data to CSV for modeling.

### 📁 `data_models.py`
- Contains object classes for:
  - `Patient`, `Diagnosis`, `Treatment`, `Demographic`, and `Molecular`
- Supports structured parsing of clinical fields.

### 📁 `ml_services.py`
- Handles the full ML pipeline:
  - Data preprocessing
  - Model training
  - Hyperparameter tuning via `GridSearchCV`
  - Internal and external prediction
  - Model evaluation summaries
  - Tracks the best-performing model

### 📁 `data_visualization.py`
- Provides graphing tools using `matplotlib` and `seaborn`:
  - Subtype distribution
  - Age histogram
  - Feature distributions (e.g. ESR1, PGR)
  - Model comparison
  - Feature importances
  - Confusion matrix (best model)

### 📁 `menu_controller.py`
- CLI interface for users to:
  - Predict subtype from patient input
  - Browse trained models and metrics
  - View detailed visualization plots
  - Navigate through menus interactively

---

## 🖥️ Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/ParweshBhaggan/CapstoneProject.git

## 🛠️ Setup Instructions

Follow these steps to get started:

### 1. Clone the Repository

```bash
git clone https://github.com/ParweshBhaggan/CapstoneProject.git
```

### 2. Create a Python Environment
We recommend using venv:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv/Scripts/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
python main.py

## 🧠 Authors
Burak Kilic

Brent de Brons

Dimitri Etienne

Parwesh Bhaggan

Roderick Wilson

## 🔒 Access
This repository is private and intended only for course staff and group members.
