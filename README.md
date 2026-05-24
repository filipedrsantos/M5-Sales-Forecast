# M5 Sales Forecast

## TL;DR
End-to-end demand forecasting system for intermittent retail sales using NHITS.  
Includes data pipeline, feature engineering, API (FastAPI), dashboard (Streamlit), and testing.  
**Key result:** Lag-based features significantly improve performance on sparse time series.

---

## Overview
This project implements an end-to-end demand forecasting pipeline using the **M5 Forecasting Accuracy** dataset. 
The goal is to predict daily product sales at store level, handling highly sparse time series where most days have zero sales and occasional spikes.

The solution combines:

* Data preprocessing and feature engineering  
* Deep learning forecasting using NHITS  
* A REST API for training and inference  
* An interactive dashboard for visualization  
* Automated tests for validation
  
---

## Pipeline

### Data Ingestion
* Data is downloaded from Kaggle (M5 Forecasting dataset) using the Kaggle API  
* Due to size constraints, datasets are **not included** in this repository  

### Data Preprocessing
* Filter by store  

### Feature Engineering
**Exogenous features:**  
* sell_price → product price  
* price_change → relative price variation  

**Lag-based features (key for performance):**  
* lag_1 → yesterday’s sales  
* lag_7 → sales one week ago  
* was_zero_yesterday → binary indicator of no sale  

These features allow the model to capture:  
* demand persistence  
* weekly patterns  
* intermittent behavior  

### Model Training
* Model: NHITS (NeuralForecast)  
* Loss: MAE (robust for sparse data)  
* Multi-horizon forecasting (e.g., 7 and 30 days)  

**Key improvements:**  
* Lag features improve zero prediction  
* Clipping negative predictions  
* Rounding outputs to match discrete demand  

### Evaluation
* Metrics: sMAPE and RMSE  
* Comparison with naive baseline (last observed value)  
* Per-product performance analysis  

### API (FastAPI)
Provides endpoints for:  
* /train → train and save models  
* /forecast → retrieve forecasts  
* /health → service status  

Supports:  
* multiple horizons  
* filtering by product  
* top-K selection  

### Dashboard (Streamlit)
Interactive UI to:  
* select forecast horizon  
* choose specific products or top-K  
* visualize:  
  * real sales  
  * model predictions  
  * naive baseline  

### Testing
Automated tests for:  
* data integrity  
* time consistency  
* train/validation split (no leakage)  
* API correctness  

---

## Folder Structure

M5-Sales-Forecast/  
* data_raw/ # Raw Kaggle data (auto-created)  
* data_processed/ # Preprocessed data & reports (auto-created)  
* models/ # Trained model versions (auto-created)  
* lightning_logs/ # Training logs (auto-created)  
* import_data.py # Download & unzip data  
* data.py # Preprocess data  
* app.py # FastAPI API  
* streamlit_app.py # Dashboard UI  
* test_*.py # Test scripts  
* config.yaml # Configuration  
* requirements.txt # Python dependencies  
* pyproject.toml # Poetry project  

> Note: data_raw/, data_processed/, models/, and lightning_logs/ are automatically created when running the pipeline.

---

## Setup

### Clone repository
* git clone https://github.com/filipedrsantos/M5-Sales-Forecast.git  
* cd M5-Sales-Forecast  

### Install dependencies
(Optional) Create and activate Conda environment:  
* conda create -n m5_env python=3.10  
* conda activate m5_env  

Install packages:  
* pip install -r requirements.txt  

Or with Poetry:  
* poetry install  
* poetry shell  

### Kaggle Dataset
* Create a Kaggle account and download API token (kaggle.json)  
* Set environment variables:  
  * setx KAGGLE_USERNAME "<your_username>"  
  * setx KAGGLE_KEY "<your_key>"  

### Download and preprocess data
* python import_data.py  
* python data.py  

### Run FastAPI
* uvicorn app:app --reload  

Check health:  
* curl http://127.0.0.1:8000/health  

Train model:  
* curl -X POST http://127.0.0.1:8000/train  

Forecast example:  
* curl "http://127.0.0.1:8000/forecast?horizon=7&top_k=3"  

### Run Streamlit Dashboard
* streamlit run streamlit_app.py  
(Run from project root with API running)

### Run Tests
* pytest  

---

## Key Insights
* Lag-based features are critical to capture zero-inflated dynamics  
* Deep learning models (NHITS) require careful feature engineering to outperform naive baselines  
* Post-processing (rounding, clipping) improves real-world usability  

---

## Future Improvements
* Two-stage model (classification + regression)  
* Probabilistic forecasting  
* Hyperparameter tuning  
* Global vs hierarchical models  

---

## Author
Filipe Santos
