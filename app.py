from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# ── Load models & data ──────────────────────────────────────────────────────
with open("encoders_epics.pkl", "rb") as f:
    encoders = pickle.load(f)

# Keys from notebook Cell 18: {"city": le_city, "state": le_state, "crime": le_crime}
le_city  = encoders["city"]
le_state = encoders["state"]
le_crime = encoders["crime"]

with open("final_crime_data_processed.pkl", "rb") as f:
    final_df = pickle.load(f)

with open("ridge_final_model_EPICS.pkl", "rb") as f:
    best_model = pickle.load(f)

# ── Dropdown values (sorted, lowercase — exactly as stored in dataset) ───────
STATES = sorted(final_df["state"].unique().tolist())
CITIES = sorted(final_df["city"].unique().tolist())
CRIMES = sorted(final_df["crime_head"].unique().tolist())


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    context = {
        "states":      STATES,
        "cities":      CITIES,
        "crimes":      CRIMES,
        "prediction":  None,
        "trend":       None,
        "trend_class": None,
        "state_input": None,
        "city_input":  None,
        "crime_input": None,
        "error":       None,
    }

    if request.method == "POST":
        state_raw  = request.form.get("state", "").strip()
        city_raw   = request.form.get("city", "").strip()
        crime_raw  = request.form.get("crime_head", "").strip()

        # Values come from dropdowns so already lowercase; normalise anyway
        state      = state_raw.lower()
        city       = city_raw.lower()
        crime_head = crime_raw.lower()

        context["state_input"] = state_raw.title()
        context["city_input"]  = city_raw.title()
        context["crime_input"] = crime_raw.title()

        # ── Validation ───────────────────────────────────────────────────────
        if state not in final_df["state"].unique():
            context["error"] = f'State "{state_raw.title()}" not found in the dataset.'
            return render_template("result.html", **context)

        if city not in final_df["city"].unique():
            context["error"] = f'City "{city_raw.title()}" not found in the dataset.'
            return render_template("result.html", **context)

        if crime_head not in final_df["crime_head"].unique():
            context["error"] = f'Crime category "{crime_raw.title()}" not found in the dataset.'
            return render_template("result.html", **context)

        # ── Encode — exact keys from notebook Cell 18 ────────────────────────
        state_enc = le_state.transform([state])[0]
        city_enc  = le_city.transform([city])[0]
        crime_enc = le_crime.transform([crime_head])[0]

        # ── Fetch latest historical values — mirrors notebook Cell 36 ─────────
        city_hist = final_df[
            (final_df["state"]      == state) &
            (final_df["city"]       == city) &
            (final_df["crime_head"] == crime_head)
        ].sort_values("year")

        if city_hist.empty:
            context["error"] = "No historical data found for this combination."
            return render_template("result.html", **context)

        crime_rate_lag1 = city_hist.iloc[-1]["crime_rate"]

        state_hist = final_df[
            (final_df["state"]      == state) &
            (final_df["crime_head"] == crime_head)
        ].sort_values("year")

        state_avg_crime_rate_lag1 = state_hist.iloc[-1]["state_avg_crime_rate"]

        # ── Feature matrix — EXACT ORDER from notebook Cell 19 ───────────────
        X_input = pd.DataFrame([{
            "city_encoded":              city_enc,
            "state_encoded":             state_enc,
            "crime_head_encoded":        crime_enc,
            "crime_rate_lag1":           crime_rate_lag1,
            "state_avg_crime_rate_lag1": state_avg_crime_rate_lag1,
        }])

        # ── Predict ──────────────────────────────────────────────────────────
        prediction = best_model.predict(X_input)[0]

        # ── Trend vs 2022 — mirrors notebook Cell 36 ─────────────────────────
        crime_2022_row = final_df[
            (final_df["state"]      == state) &
            (final_df["city"]       == city) &
            (final_df["crime_head"] == crime_head) &
            (final_df["year"]       == 2022)
        ]
        crime_rate_2022 = (
            crime_2022_row.iloc[0]["crime_rate"]
            if not crime_2022_row.empty
            else crime_rate_lag1
        )

        if prediction > crime_rate_2022:
            trend       = "Increasing Crime"
            trend_class = "up"
        elif prediction < crime_rate_2022:
            trend       = "Decreasing Crime"
            trend_class = "down"
        else:
            trend       = "No Change"
            trend_class = "neutral"

        context["prediction"]  = round(float(prediction), 2)
        context["trend"]       = trend
        context["trend_class"] = trend_class

    return render_template("result.html", **context)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
