#  Crime Rate Prediction System
# Predicting City-Level Crime Rates Across India Using NCRB Data

# Overview

This project is a **Machine Learning-based Crime Rate Prediction Web Application** that forecasts crime rates for crime against women in Indian cities using historical data from the **National Crime Records Bureau (NCRB)**. Users can input a state, city, and crime category through a web interface and receive a predicted crime rate along with a year-on-year trend indicator.

# Machine Learning Pipeline

# 1. Data Preprocessing
- Loads and standardizes the NCRB CSV dataset
- Filters to `crime_category == 'total'` only
- Removes aggregate rows (e.g., "all india", "total cities")
- Back-calculates city population from crime rate and incidence
- Imputes missing population values using city-state group medians

### 2. Feature Engineering
- **`crime_rate_lag1`** — Previous year's crime rate for the same city–crime combination
- **`state_avg_crime_rate_lag1`** — Previous year's state-level average crime rate for that crime type

### 3. Train-Test Split
Splits train and test data where train will be upto previous year and test data will be of current year.

### 4. Models Evaluated
We trained 3 models - Ridge, Random Forest and Gradient Boosting out of which Ridge had the highest R2 score of 0.9040
Therefore, Ridge Regression - Best model selected for deployment.

### 5. Final Feature Set
city_encoded, state_encoded, crime_head_encoded,
crime_rate_lag1, state_avg_crime_rate_lag1

##  Model Performance

**Ridge Regression (α = 100)** was selected as the final model due to its highest R² score. Its strong performance is attributed to the near-linear relationship between lag features and the target crime rate, with L2 regularization effectively handling multicollinearity.

# Web Application

Built with Flask (backend) and HTML/CSS (frontend).

### How It Works
1. User enters **State**, **City**, and **Crime Head** in the web form
2. Flask validates inputs against known dataset values
3. Historical lag features are fetched from the preprocessed dataset
4. The Ridge model predicts the crime rate for the current year
5. The result is compared with the 2022 crime rate to determine trend

### Trend Indicators
| Trend | Condition |
|-------|-----------|
| Increasing Crime | Predicted rate > 2022 rate |
| Decreasing Crime | Predicted rate < 2022 rate |
| No Change | Predicted rate = 2022 rate |

# Installation and Setup

### Prerequisites
- Python 3.8+
- pip

### Step 1 — Clone the Repository
```bash
git clone https://github.com/your-username/crime-rate-prediction.git
cd crime-rate-prediction
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the Notebook (to generate model and encoder files)
Open and run all cells in `EPICS_FINAL.ipynb`. This will generate:
- `final_crime_data_processed.pkl`
- `encoders_epics.pkl`
- `ridge_final_model_EPICS.pkl`

### Step 4 — Start the Flask Server
```bash
python app.py
```

### Step 5 — Open in Browser
```
http://127.0.0.1:5000
