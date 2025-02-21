from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('Trained Model/cannabis_sales_model.pkl')

# Get expected feature names
expected_features = model.feature_names_in_

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Helper functions for safe conversions
        def safe_float(value, default=0.0):
            try:
                return float(value) if value.strip() else default
            except ValueError:
                return default

        def safe_int(value, default=0):
            try:
                return int(value) if value.strip() else default
            except ValueError:
                return default

        # Extract values safely
        data = {
            'Adult-Use Retail Sales': safe_float(request.form.get('adult_use_sales', '0')),
            'Medical Marijuana Retail Sales': safe_float(request.form.get('medical_sales', '0')),
            'Adult-Use Products Sold': safe_int(request.form.get('adult_use_products', '0')),
            'Medical Products Sold': safe_int(request.form.get('medical_products', '0')),
            'Adult-Use Average Product Price': safe_float(request.form.get('adult_use_avg_price', '0')),
            'Medical Average Product Price': safe_float(request.form.get('medical_avg_price', '0')),
            'Year': safe_int(request.form.get('year', '2024')),
            'Month': safe_int(request.form.get('month', '1')),
            'Day': safe_int(request.form.get('day', '1')),
        }

        # Compute missing features
        data['Total Products Sold'] = data['Adult-Use Products Sold'] + data['Medical Products Sold']
        data['Total Products Sold per Week'] = data['Total Products Sold'] / 7  # Assuming weekly data

        # Convert to DataFrame and reorder columns
        input_data = pd.DataFrame([data])
        input_data = input_data[expected_features]  

        # Make prediction
        prediction = model.predict(input_data)[0]

        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
