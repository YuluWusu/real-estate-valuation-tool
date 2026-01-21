from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model components
model = joblib.load("models/rf_model.joblib")
x_scaler = joblib.load("models/x_scaler.joblib")
y_scaler = joblib.load("models/y_scaler.joblib")
encoder = joblib.load("models/ordinal_encoder.joblib")
categorical_cols = joblib.load("models/categorical_cols.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        data = {
            "area": [float(request.form["area"])],
            "bedrooms": [int(request.form["bedrooms"])],
            "bathrooms": [int(request.form["bathrooms"])],
            "stories": [int(request.form["stories"])],
            "mainroad": [request.form["mainroad"]],
            "guestroom": [request.form["guestroom"]],
            "basement": [request.form["basement"]],
            "hotwaterheating": [request.form["hotwaterheating"]],
            "airconditioning": [request.form["airconditioning"]],
            "parking": [int(request.form["parking"])],
            "prefarea": [request.form["prefarea"]],
            "furnishingstatus": [request.form["furnishingstatus"]],
        }

        df = pd.DataFrame(data)

        # Encode categorical
        cat_data = df[categorical_cols]
        encoded_cat = encoder.transform(cat_data)
        encoded_cat_df = pd.DataFrame(encoded_cat, columns=categorical_cols)

        # Combine
        num_data = df.drop(columns=categorical_cols)
        X = pd.concat([num_data, encoded_cat_df], axis=1)

        # Scale + predict
        X_scaled = x_scaler.transform(X)
        y_scaled = model.predict(X_scaled)
        y_pred = y_scaler.inverse_transform(y_scaled.reshape(-1, 1))

        prediction = f"${y_pred[0][0]:,.2f}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
