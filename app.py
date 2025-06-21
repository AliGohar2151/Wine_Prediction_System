from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pickle
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "wine_model_regression.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

model = None
scaler = None

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    with open(SCALER_PATH, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Regression model and scaler loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model or scaler file not found at expected paths.")
    print(f"Expected model at: {MODEL_PATH}")
    print(f"Expected scaler at: {SCALER_PATH}")
    print(
        "Please ensure 'wine_model_regression.pkl' and 'scaler.pkl' are in the same directory as app.py"
    )
except Exception as e:
    print(f"An error occurred while loading model/scaler: {e}")


@app.route("/")
def landing():

    return render_template("index_landing.html")


@app.route("/predict_form")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return (
            "Error: The prediction system could not be initialized. Please check server logs.",
            500,
        )

    try:
        features = [
            float(request.form["fixed_acidity"]),
            float(request.form["volatile_acidity"]),
            float(request.form["citric_acid"]),
            float(request.form["residual_sugar"]),
            float(request.form["chlorides"]),
            float(request.form["free_sulfur_dioxide"]),
            float(request.form["total_sulfur_dioxide"]),
            float(request.form["density"]),
            float(request.form["pH"]),
            float(request.form["sulphates"]),
            float(request.form["alcohol"]),
        ]

        input_features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(input_features)
        prediction = model.predict(scaled_features)[0]
        predicted_quality = round(prediction)

        return render_template(
            "index.html",
            prediction_text=f"Predicted Wine Quality: {predicted_quality} (Raw: {prediction:.2f})",
        )

    except KeyError as e:
        return (
            f"Error: Missing input field - {e}. Please ensure all wine properties are entered.",
            400,
        )
    except ValueError as e:
        return (
            f"Error: Invalid input value - {e}. Please ensure all inputs are valid numbers.",
            400,
        )
    except Exception as e:
        return f"An unexpected error occurred during prediction: {str(e)}", 500


if __name__ == "__main__":
    app.run(debug=True)
