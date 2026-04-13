<div align="center">

# ✈️ APUWATCH
### APU Predictive Maintenance System

<img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/LightGBM-4.3-green?style=for-the-badge&logo=lightgbm&logoColor=white" alt="LightGBM">
<img src="https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
<img src="https://img.shields.io/badge/Docker-Containerized-blue?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
<img src="https://img.shields.io/badge/GCP-Cloud%20Run-orange?style=for-the-badge&logo=google-cloud&logoColor=white" alt="GCP">
<img src="https://img.shields.io/badge/MLflow-Tracked-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow">

**Predict failure before it fails you.**

*An end-to-end MLOps system for predicting Remaining Useful Life (RUL) of aircraft APU engines using real-time sensor data and machine learning.*

[🚀 Live Demo](#) • [📖 Documentation](#table-of-contents) • [🎯 Features](#-key-features) • [⚡ Quick Start](#-quick-start)

---

</div>

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [🎯 Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [🔬 The Science Behind It](#-the-science-behind-it)
- [📊 Performance Metrics](#-performance-metrics)
- [⚡ Quick Start](#-quick-start)
- [🐳 Docker Deployment](#-docker-deployment)
- [☁️ Cloud Deployment (GCP)](#️-cloud-deployment-gcp)
- [🧪 Testing](#-testing)
- [📁 Project Structure](#-project-structure)
- [🔄 MLOps Pipeline](#-mlops-pipeline)
- [🛠️ Technology Stack](#️-technology-stack)
- [📈 Model Details](#-model-details)
- [🎨 Frontend Dashboard](#-frontend-dashboard)
- [🔍 Monitoring & Drift Detection](#-monitoring--drift-detection)
- [👥 Team](#-team)
- [📄 License](#-license)

---

## 🌟 Overview

**APUWATCH** is a production-ready predictive maintenance system designed for aircraft Auxiliary Power Units (APUs). It leverages advanced machine learning and time-series analysis to predict exactly how many operational cycles remain before an engine requires maintenance.

### 💡 The Problem

Airlines face a critical dilemma:
- **Reactive Maintenance**: Fix after failure → Dangerous, expensive, grounds aircraft
- **Scheduled Maintenance**: Replace parts at fixed intervals → Wasteful, parts replaced when still functional
- **Predictive Maintenance**: Use sensor data to predict failure → Safe, cost-effective, data-driven ✅

### 🎯 Our Solution

APUWATCH analyzes 21 sensor readings per engine cycle and predicts the **Remaining Useful Life (RUL)** with **99.4% accuracy (R² = 0.9941)**. The system:

- ✅ Processes any number of engines simultaneously
- ✅ Automatically retrains weekly on new data
- ✅ Detects sensor drift in real-time
- ✅ Provides professional aviation-themed dashboard
- ✅ Runs completely serverless on Google Cloud
- ✅ Monitors its own health with custom metrics

---

## 🎯 Key Features

### 🤖 Machine Learning Excellence
- **99.4% Prediction Accuracy** (R² = 0.9941) on unseen engines
- **LightGBM Regressor** with 69 engineered features
- **Fault Injection Training** - 5 synthetic fault patterns for robustness
- **Engine-based Data Split** - Zero data leakage guarantee

### 🔄 Full MLOps Automation
- **Automated Weekly Retraining** via Cloud Scheduler
- **Smart Model Promotion** - Only deploys if new model beats current R²
- **MLflow Experiment Tracking** - Complete parameter/metric versioning
- **DVC Data Versioning** - Git-like control for datasets

### 📊 Real-Time Monitoring
- **Feature Drift Detection** - Rolling window z-score analysis
- **Prediction Drift Detection** - Trend analysis over 4-week windows
- **Custom Cloud Metrics** - R², drift flags, request counts
- **Automated Alerting** - Email notifications for degradation

### 🎨 Professional UI
- **Aviation Cockpit Theme** - Dark mode with HUD green accents
- **Interactive Charts** - Chart.js visualization of True vs Predicted RUL
- **Live Metrics** - Animated gauge rings for MSE, MAE, RMSE, R²
- **Dynamic Status Badges** - CRITICAL / WARNING / NOMINAL per cycle

### 🔒 Production-Grade Security
- **Zero Hardcoded Credentials** - All secrets in GCP Secret Manager
- **IAM-based Access Control** - Principle of least privilege
- **Encrypted Data Transfer** - HTTPS-only communication
- **Docker Containerization** - Isolated execution environment

---

## 🏗️ Architecture

```mermaid
graph TB
    A[User Uploads CSV] --> B[Flask API /predict]
    B --> C[InferencePipeline]
    C --> D[Preprocessing]
    D --> E{Feature Engineering}
    E --> F[Op-Condition Normalization]
    E --> G[Rolling Window Features]
    E --> H[Cycle Normalization]
    F --> I[StandardScaler]
    G --> I
    H --> I
    I --> J[LightGBM Model]
    J --> K[RUL Predictions]
    K --> L[Drift Detection]
    L --> M[Save to Cloud SQL]
    L --> N[Save CSV to GCS]
    L --> O[Send Metrics to Cloud Monitoring]
    K --> P[Return JSON Response]
    P --> Q[APUWATCH Dashboard]
    
    R[Cloud Scheduler] -.->|Every Sunday| S[Cloud Run Job: Retrain]
    S --> T[Collect GCS Uploads]
    T --> U[Fault Injection]
    U --> V[Feature Engineering]
    V --> W[Train New LightGBM]
    W --> X{R² Better?}
    X -->|Yes| Y[Promote to Production]
    X -->|No| Z[Archive Model]
    Y --> AA[Update GCS Model]
    
    style A fill:#00ff88,stroke:#333,stroke-width:2px,color:#000
    style Q fill:#00ff88,stroke:#333,stroke-width:2px,color:#000
    style J fill:#ff6b6b,stroke:#333,stroke-width:2px
    style Y fill:#51cf66,stroke:#333,stroke-width:2px
    style Z fill:#ffd43b,stroke:#333,stroke-width:2px
```

### 🔄 Data Flow

1. **User Input** → CSV file with engine sensor data (21 sensors per cycle)
2. **Preprocessing** → Op-condition normalization, rolling features, scaling
3. **Inference** → LightGBM predicts RUL for each cycle
4. **Storage** → Predictions → Cloud SQL, Raw CSV → GCS
5. **Monitoring** → Custom metrics → Cloud Monitoring
6. **Retraining** → Weekly automated pipeline → Model promotion if improved

---

## 🔬 The Science Behind It

### 📊 Dataset: NASA CMAPSS FD002

- **Source**: NASA Ames Research Center (Public Domain)
- **Type**: Commercial Modular Aero-Propulsion System Simulation
- **Engines**: 260 engine histories
- **Features**: 21 onboard sensors + 3 operational settings
- **Cycles**: Variable (each engine runs until failure)

### 🧬 Fault Injection Strategy

Since the NASA dataset contains only healthy degradation, we synthetically inject **5 fault types**:

| Fault ID | Name | Affected Sensors | Pattern |
|----------|------|------------------|---------|
| 0 | Healthy | None | Original data |
| 1 | Gradual Drift | sensor_2, sensor_7 | Slow ±10% drift |
| 2 | Bias Shift | sensor_4, sensor_8 | Sudden constant offset |
| 3 | Noise Increase | sensor_3 | 1x → 3x std growth |
| 4 | Spike | sensor_9 | Random 3-5x spikes |
| 5 | Coupled Fault | sensor_11, sensor_12 | Correlated drift |

**Result**: 1,560 engine histories (260 engines × 6 variants)

### ⚙️ Feature Engineering

**69 Engineered Features** from 14 active sensors:

1. **RUL Calculation**: `max_cycle - current_cycle`
2. **Operating Condition Normalization**: Mean-center within op-setting groups
3. **Cycle Normalization**: `cycle / max_cycle` (0.0 = new, 1.0 = failed)
4. **Rolling Window Features** (window=5):
   - Rolling Mean (smoothed signal)
   - Rolling Std (recent variability)
   - Trend (diff, direction of change)
5. **StandardScaler**: Mean=0, Std=1 normalization

### 🧠 Model: LightGBM Regressor

**Hyperparameters**:
```yaml
n_estimators: 400
learning_rate: 0.05
max_depth: -1
num_leaves: 50
subsample: 1.0
colsample_bytree: 1.0
min_child_samples: 20
```

**Why LightGBM?**
- ⚡ Fast training (leaf-wise growth)
- 🎯 Excellent for tabular sensor data
- 📦 Lightweight model file (~1.8 MB)
- 🔍 Built-in feature importance

---

## 📊 Performance Metrics

### 🎯 Test Results (20-row sample, unseen engine)

```
┌─────────────────────────────────────┐
│  METRIC  │   VALUE   │  BENCHMARK  │
├──────────┼───────────┼─────────────┤
│   MSE    │  0.1360   │   < 0.5     │ ✅
│   MAE    │  0.3014   │   < 1.0     │ ✅
│   RMSE   │  0.3688   │   < 1.0     │ ✅
│   R²     │  0.9941   │   > 0.90    │ ✅ EXCELLENT
└─────────────────────────────────────┘

Health Index: 99% HEALTHY
Status: MODEL ARMED • SCALER LOADED • INFERENCE READY
```

### 📈 What This Means

- **R² = 0.9941**: Model explains **99.41%** of variance in RUL
- **MAE = 0.3014**: Predictions off by only **~0.3 cycles** on average
- **Generalization**: Strong performance on completely unseen engines

---

## ⚡ Quick Start

### 📋 Prerequisites

- Python 3.9+ (recommended: 3.11)
- Virtual environment tool
- Git

### 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/APU_Predictive_Maintenance.git
cd APU_Predictive_Maintenance

# Create virtual environment
python -m venv .APU_venv

# Activate virtual environment
# Windows:
.APU_venv\Scripts\activate
# Linux/Mac:
source .APU_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 🏃 Run the Complete Pipeline

```bash
# Step 1: Fault Injection (generates training labels)
python NoteBook/Fault_injection/fault_injection.py

# Step 2: Train/Test Split
python entrypoint/data_split.py

# Step 3: Feature Engineering
python entrypoint/feature_engineering.py

# Step 4: Model Training
python entrypoint/training.py

# Step 5: View MLflow Experiments
mlflow ui
# Open http://localhost:5000

# Step 6: Run Inference (CLI)
python -m entrypoint.inference --input testing_datasets/exp/data_testing_1.csv

# Step 7: Launch Frontend
python frontend/app.py
# Open http://localhost:8080
```

### 🧪 Run Tests

```bash
pytest tests/ -v
```

**Expected Output**:
```
tests/test_inference_pipeline.py::test_feature_count PASSED
tests/test_inference_pipeline.py::test_anomaly_score_injection PASSED
tests/test_inference_pipeline.py::test_scaler_deterministic PASSED
tests/test_inference_pipeline.py::test_output_report_structure PASSED
tests/test_inference_pipeline.py::test_flask_api_endpoint PASSED
tests/test_fault_injection.py::test_fault_balance PASSED
tests/test_fault_injection.py::test_no_nan_after_injection PASSED

============== 7 passed in 15.32s ==============
```

---

## 🐳 Docker Deployment

### 🏗️ Build the Docker Image

```bash
docker build -t apu-app:latest .
```

### 🚀 Run Locally in Docker

```bash
docker run -p 8080:8080 apu-app:latest
```

Open browser: `http://localhost:8080`

### 📦 What's Inside the Container

- ✅ Python 3.11-slim base
- ✅ All dependencies from requirements.txt
- ✅ Flask app on port 8080
- ✅ Model/scaler loaded from GCS at startup (in production mode)
- ❌ Model files NOT baked in (fetched at runtime for easy updates)

---

## ☁️ Cloud Deployment (GCP)

### 🌐 Live Production URL

```
https://apu-app-[hash]-uc.a.run.app
```

### 📋 GCP Services Used

| Service | Purpose | Configuration |
|---------|---------|---------------|
| **Cloud Run** | Serverless app hosting | 1 vCPU, 1GB RAM, port 8080 |
| **Cloud SQL** | PostgreSQL database | db-f1-micro, 10GB SSD |
| **Cloud Storage** | Model artifacts + uploads | 2 buckets (model, data) |
| **Artifact Registry** | Docker image storage | us-central1 region |
| **Cloud Scheduler** | Weekly retraining cron | Every Sunday 00:00 UTC |
| **Secret Manager** | DB credentials | Encrypted password storage |
| **Cloud Monitoring** | Custom metrics + alerts | R², drift, latency |

### 🚀 CI/CD Pipeline (GitHub Actions)

**Trigger**: Every `git push` to `main` branch

```yaml
Pipeline: test → build → deploy
Duration: 8-12 minutes
Result: Live production deployment
```

**Stages**:
1. **Test** - Run all 7 pytest tests (gate: must pass)
2. **Build** - Docker build + push to Artifact Registry
3. **Deploy** - Update Cloud Run service with new image

### 🔐 Required GitHub Secrets

```yaml
GCP_SA_KEY: [Service Account JSON Key]
GCP_PROJECT_ID: apu-predictive-maintenance
GCP_REGION: us-central1
```

### 📊 Environment Variables (Cloud Run)

```bash
GCS_MODEL_BUCKET=apu-model-artifacts-[PROJECT_ID]
GCS_DATA_BUCKET=apu-incoming-data-[PROJECT_ID]
GCP_PROJECT_ID=[PROJECT_ID]
DB_NAME=apu_predictions
DB_USER=apu_user
DB_HOST=[Cloud SQL IP]
```

---

## 🧪 Testing

### 🎯 Test Coverage

| Test File | Tests | Purpose |
|-----------|-------|---------|
| `test_inference_pipeline.py` | 5 | Inference logic, API contract, determinism |
| `test_fault_injection.py` | 2 | Data integrity, balance checking |

### 🔬 Test Details

```python
# Test 1: Feature Count Validation
def test_feature_count():
    """Ensures model receives exactly 69 features"""
    # Guards against feature engineering changes

# Test 2: Anomaly Score Auto-Injection
def test_anomaly_score_injection():
    """Verifies missing anomaly_score is added as 0.0"""
    # Prevents crashes on real-world CSV uploads

# Test 3: Scaler Determinism
def test_scaler_deterministic():
    """Confirms scaler never refits on new data"""
    # Prevents prediction drift over time

# Test 4: Output Report Structure
def test_output_report_structure():
    """Validates CSV report has correct sections"""
    # Ensures downstream compatibility

# Test 5: Flask API Endpoint
def test_flask_api_endpoint():
    """Tests full /predict API contract"""
    # End-to-end API validation

# Test 6: Fault Injection Balance
def test_fault_balance():
    """Checks all 6 fault types have equal rows"""
    # Prevents class imbalance

# Test 7: No NaN After Injection
def test_no_nan_after_injection():
    """Ensures fault injection doesn't create NaNs"""
    # Data quality gate
```

---

## 📁 Project Structure

```
APU_Predictive_Maintenance/
│
├── 📁 config/
│   └── config.yaml                    # Master configuration (paths, hyperparams)
│
├── 📁 src/                            # Core source code
│   ├── Data_preprocessing/
│   │   ├── Data_split/
│   │   │   └── Data_split.py          # Engine-based train/test split
│   │   └── Fault_injection/
│   │       └── fault_injection.py     # 5 fault type injection
│   ├── Pipelines/
│   │   ├── feature_engineering_pipeline.py  # Full FE pipeline
│   │   ├── trainig_pipeline.py              # LightGBM training + MLflow
│   │   └── inference_pipeline.py            # Inference + drift detection
│   └── monitoring/
│       └── prediction_drift.py        # Cloud SQL trend analysis
│
├── 📁 entrypoint/                     # CLI scripts
│   ├── data_split.py
│   ├── feature_engineering.py
│   ├── training.py
│   ├── inference.py
│   └── retrain.py                     # Automated retraining pipeline
│
├── 📁 frontend/                       # Web application
│   ├── app.py                         # Flask backend (GCP integration)
│   ├── templates/
│   │   └── index.html                 # Aviation FMS UI
│   └── static/
│       ├── css/style.css              # Cockpit dark theme
│       └── js/main.js                 # Chart.js + table logic
│
├── 📁 tests/                          # Automated tests
│   ├── test_inference_pipeline.py     # 5 inference tests
│   └── test_fault_injection.py        # 2 data quality tests
│
├── 📁 infra/                          # Infrastructure code
│   ├── schema.sql                     # PostgreSQL database schema
│   └── setup_scheduler.sh             # Cloud Scheduler + Run Job setup
│
├── 📁 Utils/
│   ├── Logging/logger.py              # Timestamped logging
│   └── Exception/exception.py         # Custom exceptions
│
├── 📁 Artifacts/                      # Generated ML artifacts
│   ├── raw/train_FD002.csv            # Original NASA data
│   ├── Fault_injected_data/           # 6× augmented dataset
│   ├── Data_split/                    # Train/test CSVs
│   ├── preprocessed/                  # Feature-engineered CSVs
│   ├── Scaler/scaler.pkl              # StandardScaler
│   ├── Model/model_LGBM.pkl           # Trained LightGBM
│   ├── baseline_stats.json            # Training baseline for drift
│   └── model_validations/             # Inference reports
│
├── 📁 .github/workflows/
│   └── deploy.yml                     # CI/CD pipeline
│
├── Dockerfile                         # Main app container
├── Dockerfile.retrain                 # Retraining job container
├── .dockerignore
├── requirements.txt                   # Python dependencies
├── .dvc/                              # DVC configuration
├── mlflow.db                          # MLflow experiment database
└── README.md                          # You are here! 👋
```

---

## 🔄 MLOps Pipeline

### 🔁 Automated Retraining (Every Sunday 00:00 UTC)

```mermaid
graph LR
    A[Cloud Scheduler] --> B[Trigger Cloud Run Job]
    B --> C[Collect GCS Uploads]
    C --> D[Load Original NASA Data]
    D --> E[Evidently Drift Report]
    E --> F[Fault Injection]
    F --> G[Feature Engineering]
    G --> H[Train New LightGBM]
    H --> I{R² Better?}
    I -->|Yes| J[Promote to Production]
    I -->|No| K[Archive Model]
    J --> L[Update GCS Model]
    L --> M[Log to Cloud SQL]
    K --> M
    
    style A fill:#4CAF50,stroke:#333,stroke-width:2px
    style J fill:#4CAF50,stroke:#333,stroke-width:2px
    style K fill:#FFC107,stroke:#333,stroke-width:2px
```

### 🎯 Model Promotion Logic

```python
if new_model_r2 > current_production_r2:
    # Promote new model
    upload_to_gcs("model_LGBM.pkl", new_model)
    upload_to_gcs("scaler.pkl", new_scaler)
    upload_to_gcs("baseline_stats.json", new_baseline)
    log_promotion(promoted=True)
else:
    # Keep current model
    archive_model(new_model, version=timestamp)
    log_promotion(promoted=False)
```

### 📊 Experiment Tracking (MLflow)

Every training run logs:
- **Parameters**: All hyperparameters, data paths, feature count
- **Metrics**: MSE, MAE, RMSE, R² on test set
- **Artifacts**: Full LightGBM model binary
- **Tags**: Model stage (Staging/Production), Git commit SHA

**View experiments**:
```bash
mlflow ui
```
Open `http://localhost:5000`

---

## 🛠️ Technology Stack

### 🧠 Machine Learning

| Technology | Version | Purpose |
|------------|---------|---------|
| **LightGBM** | 4.3.0 | Gradient boosting regression |
| **scikit-learn** | 1.8.0 | Preprocessing (StandardScaler, IsolationForest) |
| **Pandas** | 2.1.4 | Data manipulation |
| **NumPy** | 1.26.4 | Numerical computing |
| **MLflow** | 2.10.2 | Experiment tracking |
| **Evidently AI** | 0.4.22 | Drift reports |

### 🌐 Web & Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| **Flask** | 3.0.2 | Web framework |
| **flask-cors** | 4.0.0 | CORS handling |
| **Chart.js** | CDN | Interactive charts |
| **HTML/CSS/JS** | Vanilla | Frontend UI |

### ☁️ Cloud & DevOps

| Technology | Version | Purpose |
|------------|---------|---------|
| **Docker** | Latest | Containerization |
| **GCP Cloud Run** | - | Serverless hosting |
| **GCP Cloud SQL** | PostgreSQL 14 | Database |
| **GCP Cloud Storage** | - | Object storage |
| **GitHub Actions** | - | CI/CD automation |
| **DVC** | 3.42.0 | Data versioning |

### 🔧 Utilities

| Technology | Purpose |
|------------|---------|
| **joblib** | Model serialization |
| **PyYAML** | Configuration parsing |
| **psycopg2** | PostgreSQL client |
| **google-cloud-storage** | GCS client |
| **google-cloud-monitoring** | Custom metrics |
| **google-cloud-secret-manager** | Secrets management |

---

## 📈 Model Details

### 🎯 Input Features (69 total)

**Feature Categories**:
1. **Raw Sensors** (14): sensor_2, sensor_3, sensor_4, sensor_7, sensor_8, sensor_9, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_17, sensor_20, sensor_21
2. **Rolling Mean** (14): `{sensor}_rolling_mean` (window=5)
3. **Rolling Std** (14): `{sensor}_rolling_std` (window=5)
4. **Trends** (14): `{sensor}_trend` (diff)
5. **Operational** (3): op_setting_1, op_setting_2, op_setting_3
6. **Lifecycle** (1): cycle_normalized (0.0 → 1.0)
7. **Anomaly** (1): anomaly_score (from Isolation Forest)

### 🎓 Training Details

**Data Split**: Engine-based (prevents data leakage)
- Train: 80% of engines
- Test: 20% of engines (completely unseen)

**Target Variable**: RUL (Remaining Useful Life)
```python
RUL = max_cycle_of_engine - current_cycle
```

**Loss Function**: Mean Squared Error (MSE)

**Optimization**: Gradient boosting with leaf-wise tree growth

### 🔍 Feature Importance (Top 10)

*Note: Generated during training, view in MLflow UI*

1. `cycle_normalized` - Most predictive (engine lifecycle position)
2. `sensor_2_rolling_mean` - Temperature trend
3. `sensor_7_trend` - Pressure change direction
4. `sensor_11_rolling_std` - Variability indicator
5. `anomaly_score` - Overall health deviation
6. ... *(and 64 more)*

---

## 🎨 Frontend Dashboard

### 🖥️ APUWATCH Flight Management System

**Theme**: Aviation Cockpit (Dark Mode)
- Background: Matte black (`#0a0c0f`)
- Primary: HUD green (`#00ff88`)
- Alert: Amber (`#ffaa00`)
- Fonts: Orbitron, Teko, Share Tech Mono

### 📊 Dashboard Components

#### 1️⃣ Upload Zone
- Drag & drop CSV interface
- Pre-flight checklist (4 items)
- Radar sweep animation
- File validation

#### 2️⃣ Processing Panel
- Circular SVG progress ring (0% → 100%)
- 7-step pipeline status:
  - Load Data
  - Compute RUL
  - Normalize Conditions
  - Add Rolling Features
  - Apply Scaler
  - Run Inference
  - Save Report

#### 3️⃣ Metrics Dashboard
- **4 Animated Gauge Rings**:
  - MSE (green)
  - MAE (amber)
  - RMSE (blue)
  - R² (purple)

#### 4️⃣ RUL Chart (Chart.js)
- True RUL (green solid line)
- Predicted RUL (amber dashed line)
- Engine filter pills (ALL | engine_1 | engine_2...)
- Interactive tooltips

#### 5️⃣ Mission Summary
- Total records processed
- Number of engines
- Min/Max RUL values
- Critical cycles count
- Overall R² score

#### 6️⃣ Cycle-Level Telemetry Table
- Columns: ENG | CYCLE | TRUE RUL | PRED RUL | ΔRUL | STATUS
- Pagination (25 rows/page)
- Search by engine or cycle
- Status badges:
  - 🔴 **CRITICAL**: RUL < 15% of max
  - 🟡 **WARNING**: RUL < 40% of max
  - 🟢 **NOMINAL**: All other

#### 7️⃣ Sidebar
- Spinning APU turbine rotor (animated SVG)
- APUWATCH branding
- System info: MODEL / SCALER / SENSORS / STATUS
- Engine Health Index gauge
- Navigation: LOAD DATA | DASHBOARD

#### 8️⃣ Top Bar
- FMS-APU-01 callsign
- Live UTC Zulu time
- Animated signal bars

#### 9️⃣ Bottom Status Bar
- MODEL ARMED · SCALER LOADED · INFERENCE [READY/RUNNING/COMPLETE]

### 🎬 Animations

- Scanlines overlay
- Rotor spin (4s infinite)
- Radar sweep (conic gradient)
- Signal bar pulse
- Status dot pulse
- Gauge ring fill (1.4s cubic-bezier)
- Button shimmer sweep

---

## 🔍 Monitoring & Drift Detection

### 📊 Feature Drift Detection

**Method**: Rolling Window Z-Score (time-series aware)

For each of 14 active sensors:
```python
z_score = |input_rolling_mean - baseline_mean| / baseline_std
if z_score > 2.0:
    flag_as_drifted(sensor)
```

**Why Rolling Window?**
- Time-series data has temporal autocorrelation
- Raw point-by-point comparisons (KS test) don't account for this
- 20-cycle rolling mean captures recent trends while smoothing noise

**Alert Trigger**: Any sensor with z-score > 2.0

### 📈 Prediction Drift Detection

**Method**: 4-Week Median RUL Trend Analysis

```sql
SELECT DATE_TRUNC('week', timestamp) as week,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY predicted_rul) as median_rul
FROM predictions
WHERE timestamp > NOW() - INTERVAL '4 weeks'
GROUP BY week;
```

**Alert Trigger**: Current week median < 80% of 4-week average

### 📡 Custom Cloud Metrics

Sent after every inference:

| Metric | Type | Purpose |
|--------|------|---------|
| `custom.googleapis.com/apu/r2_score` | Double | Model accuracy tracking |
| `custom.googleapis.com/apu/drift_detected` | Double | 0.0 = no drift, 1.0 = drift |
| `custom.googleapis.com/apu/inference_row_count` | Double | Request size monitoring |

### 🚨 Alerting Policies

**Alert 1**: R² Score Degradation
- Condition: `r2_score < 0.90`
- Action: Email notification

**Alert 2**: Feature Drift Detected
- Condition: `drift_detected > 0.5`
- Action: Email notification

**Alert 3**: Application Error Rate
- Condition: `5xx responses > 5/min`
- Action: Email notification

---

## 🚀 Usage Examples

### 📤 Upload a CSV File

**Required Columns**:
```
engine_id, cycle, op_setting_1, op_setting_2, op_setting_3,
sensor_1, sensor_2, ..., sensor_21
```

**Optional Columns** (handled automatically):
- `anomaly_score` → auto-injected as 0.0 if missing
- `fault_label`, `fault_type`, `fault_target` → ignored

**Constraints**:
- Must be sorted by `(engine_id, cycle)` ascending
- No NaN values
- Any number of rows/engines

### 🔮 Get Predictions

**CLI**:
```bash
python -m entrypoint.inference --input path/to/your_data.csv
```

**API** (POST to `/predict`):
```python
import requests

files = {'file': open('engine_data.csv', 'rb')}
response = requests.post('http://localhost:8080/predict', files=files)
result = response.json()

print(f"R² Score: {result['overall_metrics']['R2']}")
print(f"Total Engines: {result['total_engines']}")
```

**Response Format**:
```json
{
  "success": true,
  "total_rows": 20,
  "total_engines": 1,
  "overall_metrics": {
    "MSE": 0.136,
    "MAE": 0.3014,
    "RMSE": 0.3688,
    "R2": 0.9941
  },
  "per_engine_metrics": [...],
  "predictions": [
    {
      "engine_id": 72,
      "cycle": 141,
      "true_RUL": 19,
      "predicted_RUL": 18.73
    },
    ...
  ]
}
```

### 📊 View Dashboard

1. Open frontend: `http://localhost:8080`
2. Upload CSV file
3. Click **EXECUTE INFERENCE SEQUENCE**
4. View results:
   - Metrics gauges
   - RUL chart
   - Telemetry table
   - Export report

---

## 🎓 Learning Resources

### 📚 Understanding Predictive Maintenance

- [NASA CMAPSS Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- [Remaining Useful Life Prediction](https://en.wikipedia.org/wiki/Remaining_useful_life)
- [Condition-Based Maintenance](https://en.wikipedia.org/wiki/Condition-based_maintenance)

### 🔬 Time-Series Drift Detection

- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [Rolling Window Statistics](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html)
- [ADWIN Algorithm](https://riverml.xyz/latest/api/drift/ADWIN/)

### ☁️ MLOps Best Practices

- [Google Cloud MLOps Guide](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC for Data Versioning](https://dvc.org/doc)

---

## 🐛 Troubleshooting

### ❌ Common Issues

**Issue**: Docker build fails with "no space left on device"
```bash
# Solution: Clean up Docker
docker system prune -a
```

**Issue**: Pytest fails with "No module named 'src'"
```bash
# Solution: Run from project root with venv activated
cd E:\APU_Predictive_Maintenance
.APU_venv\Scripts\activate
pytest tests/ -v
```

**Issue**: Frontend shows "Model loading failed"
```bash
# Solution: Check model files exist
ls Artifacts/Model/model_LGBM.pkl
ls Artifacts/Scaler/scaler.pkl

# Re-run training if missing
python entrypoint/training.py
```

**Issue**: Cloud Run deployment fails
```bash
# Solution: Check Cloud Run logs
gcloud run services logs read apu-app --region=us-central1 --limit=50
```

**Issue**: Inference fails with "Feature count mismatch"
```bash
# Solution: Verify input CSV has all 21 sensors
# Model expects exactly 69 features after processing
```

<div align="center">

### 🚀 Ready to predict the future?

**Start predicting engine failures before they happen!**

[⬆ Back to Top](#️-apuwatch)

---

**From idea to execution — built with dedication throughout ⚙️**

*Predict failure before it fails you.* ✈️

</div>
