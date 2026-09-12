
# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API
import os # Import os for path manipulation
import traceback # Import traceback for error details for better debugging

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Construct the absolute path to the model file
model_path = os.path.join(os.path.dirname(__file__), "superkart_xgb_model.joblib")

# Load the trained model
model = joblib.load(model_path)

# Define a route for the home page
@superkart_api.get('/')
def home():
    print("Backend: Received GET / request") # Added logging
    return "Welcome to the SuperKart System"

# Define an endpoint to predict sales for a single product
@superkart_api.post('/v1/predict')
def predict_sales():
    print("Backend: Received POST /v1/predict request") # Added logging
    try:
        # Get JSON data from the request
        data = request.get_json()
        print(f"Backend: Single prediction payload received: {data}") # Added logging

        # Extract relevant features from the input data
        sample = {
        'Product_Weight': data['Product_Weight'],
        'Product_Sugar_Content': data['Product_Sugar_Content'],
        'Product_Allocated_Area': data['Product_Allocated_Area'],
        'Product_MRP': data['Product_MRP'],
        'Store_Size': data['Store_Size'],
        'Store_Location_City_Type': data['Store_Location_City_Type'],
        'Store_Type': data['Store_Type'],
        'Product_Id_char': data['Product_Id_char'],
        'Store_Age_Years': data['Store_Age_Years'],
        'Product_Type_Category': data['Product_Type_Category']
    }

        # Convert the extracted data into a DataFrame
        input_data = pd.DataFrame([sample])

        # Make a prediction using the trained model
        prediction = model.predict(input_data).tolist()[0]

        # Return the prediction as a JSON response
        return jsonify({'Sales': prediction})
    except Exception as e:
        print(f"Backend: Error during single prediction: {e}") # Added logging
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

# Define an endpoint to predict sales for a batch of products
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    print("Backend: Received POST /v1/predictbatch request") # Added logging
    try:
        # Log incoming request details for debugging
        # print(f"Headers: {request.headers}") # Uncomment if needed for deeper debugging
        # print(f"Files: {request.files}") # Uncomment if needed for deeper debugging

        # Get the uploaded CSV file from the request
        if 'file' not in request.files:
            print("Backend: No file part in the request for batch prediction") # Added logging
            return jsonify({'error': 'No file part in the request'}), 400
        file = request.files['file']

        if file.filename == '':
            print("Backend: No selected file for batch prediction") # Added logging
            return jsonify({'error': 'No selected file'}), 400

        # Read the file into a DataFrame
        input_data = pd.read_csv(file)
        print(f"Backend: DataFrame head from uploaded batch file:\n{input_data.head()}") # Added logging

        # Make predictions for the batch data
        predictions = model.predict(input_data).tolist()

        # Create an output dictionary mapping row index to predicted sales
        output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}
        print("Backend: Batch prediction successful") # Added logging
        return output_dict
    except Exception as e:
        print(f"Backend: Error during batch prediction: {e}") # Added logging
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_api.run(debug=True)
