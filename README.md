# 🍷 Wine Quality Predictor Flask Application

This is a Flask web application that predicts the quality of red wine based on its physicochemical properties. The application uses a machine learning regression model trained on a dataset of red wine characteristics.

### 🌐User Manual: https://aligohar2151.github.io/Wine_prediction_System_User_Manual/

## ✨ Features

* **User-friendly Interface:** Simple web form for entering wine properties.
* **Regression Model:** Predicts a continuous wine quality score (e.g., from 3 to 8).
* **Auto-fill Buttons:** Quick testing with predefined values for low, medium, and high-quality wines.
* **Responsive Design:** Optimized for various screen sizes, from mobile to desktop.
* **Landing Page:** An attractive introduction page for the application.

## 🚀 How It Works

The application uses a `RandomForestRegressor` model, which was trained on the [Wine Quality Red dataset](https://archive.ics.uci.uci.edu/dataset/186/wine+quality). Input features are scaled using `StandardScaler` before prediction to ensure consistent results.

## 🛠️ Technologies Used

* **Python:** Programming language
* **Flask:** Web framework
* **Scikit-learn:** For machine learning model (RandomForestRegressor, StandardScaler)
* **NumPy, Pandas:** For data handling
* **HTML, CSS, JavaScript:** For the front-end interface


