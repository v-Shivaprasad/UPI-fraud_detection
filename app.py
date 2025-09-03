import os
import pandas as pd
import joblib
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    df = pd.read_csv(file)
    df["transaction_time"] = pd.to_datetime(df["transaction_time"])

    grouped_features = []
    for user_id, user_txns in df.groupby("user_id"):
        user_txns = user_txns.reset_index(drop=True)

        if len(user_txns) < 2:  # skip users with <2 txns
            continue

        history = user_txns.iloc[:-1]
        current = user_txns.iloc[-1]

        time_diffs = history["transaction_time"].diff().dt.total_seconds()
        feature_dict = {
            "user_id": user_id,
            "transaction_id": current["transaction_id"],
            "current_amount": current["transaction_amount"],
            "mean_amount": history["transaction_amount"].mean(),
            "std_amount": history["transaction_amount"].std(),
            "max_amount": history["transaction_amount"].max(),
            "min_amount": history["transaction_amount"].min(),
            "unique_devices": history["device_id"].nunique(),
            "primary_device_ratio": (history["device_id"].value_counts().max() / len(history)) if len(history) > 0 else 1.0,
            "unique_locations": history["location"].nunique(),
            "location_switch_rate": ((history["location"] != history["location"].shift()).mean()) if len(history) > 1 else 0.0,
            "time_diff_mean": time_diffs.mean() if len(history) > 1 else 0.0,
            "time_diff_std": time_diffs.std() if len(history) > 1 else 0.0,
            "min_time_gap": time_diffs.min() if len(history) > 1 else 0.0,
            "txn_count": len(history),
        }

        grouped_features.append(feature_dict)

    test_features = pd.DataFrame(grouped_features)
    ids = test_features[["user_id", "transaction_id"]]
    test_features = test_features.drop(["user_id", "transaction_id"], axis=1)

    model = joblib.load("upi_fraud_model.pkl")
    y_pred = model.predict(test_features)
    y_proba = model.predict_proba(test_features)[:, 1]

    results = ids.copy()
    results["fraud_prediction"] = y_pred
    results["fraud_probability"] = y_proba

    fraud_probability = float(y_proba[0])
    prediction = int(y_pred[0])

    return render_template(
        "results.html",
        tables=[results.to_html(classes="data table table-bordered table-striped", index=False)],
        titles=results.columns.values,
        fraud_probability=fraud_probability,
        prediction=prediction,
    )

if __name__ == "__main__":
    app.run(debug=True)
