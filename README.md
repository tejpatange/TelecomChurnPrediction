# Customer Churn Prediction API

This project is a Flask-based API for predicting customer churn using a stacked ensemble machine learning model. It supports CSV input and outputs predictions via a RESTful endpoint. Designed for flexibility and extendability, this project includes automated feature engineering and supports local, LAN, and cloud-based deployment options.

---

## 🚀 Project Highlights

- **Advanced Feature Engineering**: Includes creation of ratio-based, aggregated, and plan-interaction features.
- **Modeling Techniques**:
  - Logistic Regression
  - Random Forest
  - Support Vector Classifier
  - XGBoost
  - **Stacking Classifier (Best Model)**
- **Performance Metrics**:
  - **Stacking Model Train ROC AUC**: 0.9400
  - **Stacking Model Test ROC AUC**: 0.9415
- **API for Prediction**:
  - Accepts `.csv` file input
  - Returns predictions in JSON format

---

## 📂 Project Structure

```
.
├── app.py                  # Flask API definition
├── stacking_model.pkl      # Trained stacking model
├── requirements.txt        # Python dependencies
├── Kaggle Telecom dataset  # Sample test file
├── request.py              # Example usage of API
└── README.md               # Project documentation
```

---

## 🧰 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```
---

## 🛠️ Run the API Locally

Start the Flask app:

```bash
python app.py
```

To allow network access:

```python
app.run(host='0.0.0.0', port=5000)
```

Access it via:

```
http://127.0.0.1:5000/predict
```

Or from other devices on LAN using your local IP.

---

## 🧪 Testing the API

Use the provided script: request.py

---

## 🌐 Hosting Options

- Localhost + LAN
- Ngrok for temporary external access
- Cloud hosting (Render, Railway, AWS, etc.)

---

## 📬 API Endpoint

- **POST** `/predict`
  - **Input**: CSV file (matching training structure)
  - **Output**: JSON predictions

---

## 👨‍💻 Author

- Tejas Patange
- Contact: [tejpatange@gmail.com](mailto\:tejpatange@gmail.com)

---

## 📈 Conclusion

The stacking model delivered the best generalization performance and was selected as the prediction model. This API can be extended to support more formats or models.

