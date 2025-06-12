'''
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def load_and_train_model():
    """Load Boston Housing data, train model, and save it"""
    # Since Boston Housing dataset is deprecated, we'll use California Housing
    # and simulate Boston Housing features for educational purposes
    print("Loading and preprocessing housing data...")
    
    # Load California housing data (similar to Boston housing)
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Create feature names similar to Boston housing
    feature_names = [
        'Median Income',      # Median income
        'House Age',    # House age
        'Average Rooms',    # Average rooms
        'Average Bedrooms',   # Average bedrooms
        'Population',  # Population
        'Average Occupancy',    # Average occupancy
        'Latitude',    # Latitude
        'Longitude'    # Longitude
    ]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model trained successfully!")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Save the model and scaler
    joblib.dump(model, 'housing_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    
    return model, scaler, feature_names

def load_model():
    """Load the trained model, scaler, and feature names"""
    try:
        model = joblib.load('housing_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        print("Model loaded successfully!")
        return model, scaler, feature_names
    except FileNotFoundError:
        print("Model not found. Training new model...")
        return load_and_train_model()

# Load or train the model when the app starts
model, scaler, feature_names = load_model()

@app.route('/')
def index():
    """Render the main page with the prediction form"""
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        features = []
        for feature_name in feature_names:
            value = float(request.form.get(feature_name, 0))
            features.append(value)
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale the features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Convert prediction to a more readable format (multiply by 100 for California housing)
        predicted_price = prediction * 100000  # Convert to dollars
        
        return render_template('index.html', 
                             features=feature_names, 
                             prediction=predicted_price,
                             input_values=dict(zip(feature_names, features)))
        
    except Exception as e:
        error_message = f"Error making prediction: {str(e)}"
        return render_template('index.html', 
                             features=feature_names, 
                             error=error_message)

if __name__ == '__main__':
    print("Starting Flask app...")
    print("Navigate to http://127.0.0.1:5000 to use the app")
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Global variables to store model components
model = None
scaler = None
feature_names = None

def load_and_train_model():
    """Load California Housing data, train model, and save it"""
    print("Loading and preprocessing housing data...")
    
    # Load California housing data
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Create feature names
    feature_names = [
        'Median Income',      # Median income in tens of thousands
        'House Age',          # Median house age in years
        'Average Rooms',      # Average number of rooms per household
        'Average Bedrooms',   # Average number of bedrooms per household
        'Population',         # Population of the block group
        'Average Occupancy',  # Average house occupancy
        'Latitude',          # Latitude of the block group
        'Longitude'          # Longitude of the block group
    ]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model trained successfully!")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Save the model and scaler with error handling
    try:
        joblib.dump(model, 'housing_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(feature_names, 'feature_names.pkl')
        print("Model files saved successfully!")
    except Exception as e:
        print(f"Warning: Could not save model files: {e}")
    
    return model, scaler, feature_names

def load_model():
    """Load the trained model, scaler, and feature names"""
    try:
        if os.path.exists('housing_model.pkl') and os.path.exists('scaler.pkl') and os.path.exists('feature_names.pkl'):
            model = joblib.load('housing_model.pkl')
            scaler = joblib.load('scaler.pkl')
            feature_names = joblib.load('feature_names.pkl')
            print("Model loaded successfully from saved files!")
            return model, scaler, feature_names
        else:
            print("Model files not found. Training new model...")
            return load_and_train_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training new model...")
        return load_and_train_model()

def initialize_model():
    """Initialize the model components"""
    global model, scaler, feature_names
    try:
        model, scaler, feature_names = load_model()
        return True
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return False

@app.route('/')
def index():
    """Render the main page with the prediction form"""
    global feature_names
    
    # Initialize model if not already done
    if feature_names is None:
        if not initialize_model():
            return "Error: Could not initialize the prediction model.", 500
    
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    global model, scaler, feature_names
    
    # Initialize model if not already done
    if model is None or scaler is None or feature_names is None:
        if not initialize_model():
            return render_template('index.html', 
                                 features=feature_names or [], 
                                 error="Model initialization failed")
    
    try:
        # Get form data
        features = []
        input_values = {}
        
        for feature_name in feature_names:
            value = float(request.form.get(feature_name, 0))
            features.append(value)
            input_values[feature_name] = value
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale the features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Convert prediction to a more readable format
        # California housing target is median house value in hundreds of thousands of dollars
        predicted_price = prediction * 100000  # Convert to dollars
        
        return render_template('index.html', 
                             features=feature_names, 
                             prediction=predicted_price,
                             input_values=input_values)
        
    except ValueError as ve:
        error_message = f"Invalid input values: {str(ve)}"
        return render_template('index.html', 
                             features=feature_names, 
                             error=error_message)
    except Exception as e:
        error_message = f"Error making prediction: {str(e)}"
        return render_template('index.html', 
                             features=feature_names, 
                             error=error_message)

@app.route('/health')
def health_check():
    """Health check endpoint for deployment platforms"""
    global model, scaler, feature_names
    
    if model is None or scaler is None or feature_names is None:
        return jsonify({"status": "unhealthy", "message": "Model not initialized"}), 500
    
    return jsonify({"status": "healthy", "message": "Model ready"})

if __name__ == '__main__':
    print("Starting Flask app...")
    print("Initializing model...")
    
    # Initialize model on startup when running directly
    if initialize_model():
        print("Model initialized successfully!")
        print("Navigate to http://127.0.0.1:5000 to use the app")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize model. Exiting.")
else:
    # When running with gunicorn, initialize model lazily
    print("Running with WSGI server - model will be initialized on first request")
