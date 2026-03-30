# 🚦 ModelShift
### Label-Free Monitoring for Deployed Machine Learning Models

[![PyPI version](https://badge.fury.io/py/modelshift.svg)](https://badge.fury.io/py/modelshift)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A lightweight, behavior-centric system to detect **silent reliability degradation** in deployed machine learning models — without requiring ground-truth labels.

---

## 📌 Why ModelShift?

Machine learning models rarely fail loudly after deployment. Instead, they **silently degrade** as real-world data changes — while true labels are unavailable for continuous evaluation.

**ModelShift addresses this blind spot.**

---

## 🧩 Problem Statement & Objective

Deployed machine learning models often degrade silently over time due to changing data distributions. Design a **label-free, post-deployment monitoring system** that tracks:
- Data distribution shifts 
- Prediction behavior instability 
- Model reliability trends 

...to provide **early warning signals** of degradation **without modifying the deployed model**.

> **Note:** ModelShift explicitly does *not* retrain models, correct predictions, or compute accuracy on production data. It focuses solely on deterministic monitoring, telemetry, and interpretability.

---

## 🚀 The Architecture (SaaS Infrastructure)

ModelShift is built as a distributed, full-stack monitoring ecosystem:

1. **The PyPI SDK (`modelshift`)**: A lightweight Python package installed on the data scientist's local machine or cloud pipeline. It computes complex statistical drift (KS-Statistics, Entropy) locally.
2. **The API Bridge**: The SDK authenticates via a secure `API_KEY` and transmits JSON telemetry payloads across the web.
3. **The Web Hub (FastAPI)**: A secure cloud server equipped with SQLite, bcrypt cryptographic hashing, and automated routing.
4. **The Cyberpunk Dashboard**: A real-time HTML/JS monitoring terminal to visualize pipeline health.
5. **Automated Alerting**: Background workers that trigger styled SMTP HTML emails the second a `CRITICAL_DRIFT` event occurs.

---

## 💻 Quickstart

### 1. Install the SDK
ModelShift is published on the Python Package Index. Install it anywhere:
```bash
pip install modelshift
2. Connect Your Pipeline
Add these 5 lines of code to the end of your existing ML inference scripts to instantly beam telemetry to your dashboard:

Python
import pandas as pd
import numpy as np
from modelshift.monitor import ModelMonitor, init

# 1. Authenticate with your Cloud Dashboard API Key
init(api_key="ms_YOUR_API_KEY_HERE")

# 2. Initialize the Engine
monitor = ModelMonitor(reference_df)
monitor.set_baseline_predictions(ref_predictions)

# 3. Feed it live production data
monitor.update(live_df)
monitor.update_predictions(live_predictions)

# 4. Compute statistical drift
monitor.compute_feature_drift()
monitor.compute_prediction_drift()

# 5. Beam telemetry to the cloud
monitor.push()
🛠️ Technology Stack
Data Science & SDK Core:

Language: Python 3.8+

Math & Stats: NumPy, Pandas, SciPy

Networking: Requests

Cloud Backend & Security:

Framework: FastAPI / Uvicorn

Database: SQLite (Relational Storage)

Cryptography: bcrypt (Password Hashing)

Notifications: Python smtplib & email.mime (Background Tasks)

Frontend Dashboard:

UI: HTML5, TailwindCSS, Vanilla JS

Templating: Jinja2

Aesthetic: High-contrast, custom dark-mode / terminal UI

📂 Repository Structure
Plaintext
modelshift-lite/
├── dashboard_web/       # FastAPI Backend & Web Application
│   ├── data/            # SQLite DB & telemetry storage
│   ├── static/          # CSS and Vanilla JS for the dashboard
│   ├── templates/       # HTML Jinja2 templates (login, signup, dash)
│   ├── app.py           # Core FastAPI router & logic
│   └── email_alert.py   # Automated SMTP dispatch system
├── modelshift/          # The PyPI SDK Source Code
│   ├── drift/           # Statistical math (KS, Entropy, severity)
│   └── monitor.py       # Main ModelMonitor class and API Bridge
├── setup.py             # PyPI package configuration
└── README.md
⚙️ Running the Local Server (For Developers)
To run the full-stack dashboard and backend API on your local machine:

Clone this repository.

Install the backend requirements: pip install fastapi uvicorn bcrypt sqlalchemy jinja2

Start the server:

Bash
uvicorn dashboard_web.app:app --reload
Navigate to http://127.0.0.1:8000 in your browser.

Create an account, generate an API Key, and start tracking your models!

```mermaid
graph TD
    subgraph "Client Environment (Data Scientist's Laptop)"
        A[Live ML Pipeline] -->|1. Feed Data| B(ModelShift SDK)
        B -->|2. Compute Math| B
        B -->|3. push()| C[Transmitter]
    end

    subgraph "ModelShift Cloud (FastAPI Backend)"
        C -->|4. POST JSON + API_KEY| D{FastAPI Router}
        D -->|5. Validate Key| E[(SQLite Security Vault)]
        D -->|6. Save Telemetry| F[Time-Series JSON Storage]
    end

    subgraph "Outputs & Notification"
        F -->|7. Live Fetch| G[Cyberpunk Web Dashboard]
        D -->|8. Trigger if CRITICAL| H[Background Worker]
        H -->|9. SMTP Dispatch| I[✉️ HTML Email Alert]
    end

    classDef sdk fill:#1d2023,stroke:#d11f1f,stroke-width:2px,color:#fff;
    classDef cloud fill:#0f1112,stroke:#4a4d52,stroke-width:2px,color:#fff;
    classDef output fill:#2a2d30,stroke:#d11f1f,stroke-width:2px,color:#fff;
    
    class A,B,C sdk;
    class D,E,F cloud;
    class G,H,I output;
```