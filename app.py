from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

app = Flask(__name__)

# Load the trained model (assuming you have saved it as a file)
model = joblib.load('Trained Model/cannabis_sales_model.pkl')

@app.route('/')
def home():
    return render_template('form.html')  # Render the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = {
        'Adult-Use Retail Sales': float(request.form['adult_use_sales']),
        'Medical Marijuana Retail Sales': float(request.form['medical_sales']),
        'Adult-Use Products Sold': int(request.form['adult_use_products']),
        'Medical Products Sold': int(request.form['medical_products']),
        'Adult-Use Average Product Price': float(request.form['adult_use_avg_price']),
        'Medical Average Product Price': float(request.form['medical_avg_price']),
        'Year': int(request.form['year']),
        'Month': int(request.form['month']),
        'Day': int(request.form['day'])
    }

    # Convert data to DataFrame
    input_data = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(input_data)

    # Return the prediction
    return f"Predicted Total Sales: ${prediction[0]:.2f}"

if __name__ == '__main__':
    app.run(debug=True)