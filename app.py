from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
model = joblib.load('stacking_model.pkl')
scaler = joblib.load('scaler.pkl')
app = Flask(__name__)
# Define expected columns (same as during training)
expected_columns = [
    'Account length', 'International plan',
       'Voice mail plan', 'Number vmail messages', 'Total day minutes',
       'Total day calls', 'Total day charge', 'Total eve minutes',
       'Total eve calls', 'Total eve charge', 'Total night minutes',
       'Total night calls', 'Total night charge', 'Total intl minutes',
       'Total intl calls', 'Total intl charge', 'Customer service calls',
]

def feature_engineering(df):
    # Convert categorical values to numeric
    df = df.replace({'Yes': 1, 'No': 0, 'True': 1, 'False': 0})
    
    # Drop unnecessary columns
    df = df.drop(columns=['State', 'Area code'], errors='ignore')
    
    # Create new aggregated features
    df['TotalCharges'] = df['Total eve charge'] + df['Total day charge'] + df['Total intl charge'] + df['Total night charge']
    df['TotalCalls'] = df['Total eve calls'] + df['Total day calls'] + df['Total intl calls'] + df['Total night calls']
    df['Totalmins'] = df['Total eve minutes'] + df['Total day minutes'] + df['Total intl minutes'] + df['Total night minutes']
    
    # Create charge per call features
    df["day_charge_per_call"] = df["Total day charge"] / (df["Total day calls"] + 1)
    df["eve_charge_per_call"] = df["Total eve charge"] / (df["Total eve calls"] + 1)
    df["night_charge_per_call"] = df["Total night charge"] / (df["Total night calls"] + 1)
    df["intl_charge_per_call"] = df["Total intl charge"] / (df["Total intl calls"] + 1)
    
    # Create customer service interaction feature
    df["frequent_support_contacts"] = (df["Customer service calls"] > 3).astype(int)
    
    # Subscription plan interactions
    df["both_plans"] = ((df["International plan"] == 1) & (df["Voice mail plan"] == 1)).astype(int)
    df["neither_plan"] = ((df["International plan"] == 0) & (df["Voice mail plan"] == 0)).astype(int)
    
    # Call pattern ratios
    df["day_to_night_call_ratio"] = df["Total day calls"] / (df["Total night calls"] + 1)
    df["intl_call_proportion"] = df["Total intl calls"] / (df["Total day calls"] + df["Total eve calls"] + df["Total night calls"] + 1)
    
    # Usage intensity
    df["day_usage_intensity"] = df["Total day minutes"] / (df["Total day calls"] + 1)
    df["eve_usage_intensity"] = df["Total eve minutes"] / (df["Total eve calls"] + 1)
    df["night_usage_intensity"] = df["Total night minutes"] / (df["Total night calls"] + 1)
    df["intl_usage_intensity"] = df["Total intl minutes"] / (df["Total intl calls"] + 1)
    
    return df

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:  # CSV input
        file = request.files['file']
        df = pd.read_csv(file)
    else:  # JSON input
        data = request.get_json(force=True)
        df = pd.DataFrame(data)

    # Ensure column names match
    missing_cols = [col for col in expected_columns if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in expected_columns]

    if missing_cols:
        return jsonify({"error": f"Missing columns: {missing_cols}"}), 400
    
    if extra_cols:
        df = df[expected_columns]  # Keep only expected columns, ignore extra ones

    # Apply feature engineering
    df = feature_engineering(df)

    # Scale data
    df_scaled = scaler.transform(df)

    # Predict
    predictions = model.predict_proba(df_scaled)[:, 1]

    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
