from __future__ import annotations

from io import StringIO

import pandas as pd
from flask import Flask, jsonify, render_template, request

from fraud_detection import load_training_bundle, predict_transactions


app = Flask(__name__)
bundle = load_training_bundle()


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        metrics=bundle["metrics"],
        feature_names=bundle["feature_names"],
        sample_values=bundle["sample_values"],
        prediction=None,
        batch_results=None,
        error=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    error = None
    prediction = None
    batch_results = None

    try:
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            records = payload.get("data")
            if isinstance(records, dict):
                records = [records]
            if not isinstance(records, list) or not records:
                raise ValueError("Send JSON as {'data': {...}} or {'data': [{...}, ...]}.")
            results = predict_transactions(records)
            return jsonify({"predictions": results})

        form_record = {
            feature: request.form.get(feature, "").strip()
            for feature in bundle["feature_names"]
        }
        prediction = predict_transactions([form_record])[0]
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    return render_template(
        "index.html",
        metrics=bundle["metrics"],
        feature_names=bundle["feature_names"],
        sample_values=bundle["sample_values"],
        prediction=prediction,
        batch_results=batch_results,
        error=error,
    )


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    prediction = None
    batch_results = None
    error = None

    try:
        uploaded_file = request.files.get("file")
        if uploaded_file is None or uploaded_file.filename == "":
            raise ValueError("Upload a CSV file with the model feature columns.")

        csv_text = uploaded_file.stream.read().decode("utf-8")
        dataframe = pd.read_csv(StringIO(csv_text))
        results = predict_transactions(dataframe.to_dict(orient="records"))

        batch_results = []
        for index, result in enumerate(results, start=1):
            batch_results.append(
                {
                    "row_number": index,
                    "label": result["label"],
                    "fraud_probability": result["fraud_probability"],
                }
            )
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    return render_template(
        "index.html",
        metrics=bundle["metrics"],
        feature_names=bundle["feature_names"],
        sample_values=bundle["sample_values"],
        prediction=prediction,
        batch_results=batch_results,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
