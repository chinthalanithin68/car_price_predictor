# Refurbished Car Price Predictor 🚗💰

A Streamlit web application that predicts the estimated price of a refurbished car based on user inputs such as car model, year, kilometers driven, fuel type, seller type, and transmission. The app uses a machine learning model (Random Forest Regressor) trained on a dataset of refurbished cars.

---

## 🔗 Live Demo

👉 [Refurbished Car Price Predictor - Streamlit App](https://carpricepredictor-yya8uyrxaeydxzykqzim3y.streamlit.app/)

---

## ✨ Features

- **User-friendly Input Form:** Enter car details through an intuitive sidebar interface.
- **Price Prediction:** Uses a trained Random Forest Regression model to estimate the selling price of a refurbished car.
- **Dynamic Progress Bar:** Displays a loading animation while predicting.
- **Search Link:** Provides a direct Google search link to explore more about the car model.
- **Similar Cars:** Displays similar cars from the dataset within the predicted price range for comparison.
- **Expandable Sections:** Additional information about the app, features, and instructions.

---

## 📊 Dataset

The model is trained on a dataset (`car_data.csv`) containing the following features:

- `Car_Name`
- `Year`
- `Kms_Driven`
- `Fuel_Type`
- `Seller_Type`
- `Transmission`
- `Selling_Price` (target variable)

Irrelevant columns like `Owner` and `Present_Price` are dropped during preprocessing.

---

## 🤖 Model

- **Algorithm:** Random Forest Regressor
- **Preprocessing:**
  - One-hot encoding for categorical features: `Car_Name`, `Fuel_Type`, `Seller_Type`, `Transmission`
  - Numerical features are used directly.
- **Train-Test Split:** 80% training, 20% testing
- **Implementation:** All preprocessing and modeling steps are combined using a scikit-learn Pipeline for efficiency and reproducibility.

---

## 🛠 Technologies Used

- **Language:** Python 3.12
- **Web Framework:** Streamlit
- **Data Handling:** Pandas
- **Machine Learning:** scikit-learn
- **Model Persistence:** joblib (optional)

---

## 🚀 Installation and Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/refurbished-car-price-predictor.git
    cd refurbished-car-price-predictor
    ```

2. **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
