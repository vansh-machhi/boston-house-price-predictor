# Housing Price Predictor 🏠

A beautiful machine learning web application that predicts house prices using Linear Regression trained on housing data.

## Features

- ✨ Modern, responsive web interface with custom CSS
- 🤖 Machine Learning model using Linear Regression
- 📊 Trained on California Housing dataset (similar to deprecated Boston Housing)
- 🔄 Real-time price predictions
- 📱 Mobile-friendly responsive design

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

### 3. Access the App

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

## How to Use

1. Fill in all the housing feature fields:
   - **MedInc**: Median income (in tens of thousands)
   - **HouseAge**: Median house age
   - **AveRooms**: Average number of rooms per household
   - **AveBedrms**: Average number of bedrooms per household
   - **Population**: Block group population
   - **AveOccup**: Average house occupancy
   - **Latitude**: House block latitude
   - **Longitude**: House block longitude

2. Click "Predict House Price" to get the estimated price

3. The predicted price will be displayed in a beautiful animated result section

## Technical Details

- **Backend**: Flask web framework
- **ML Model**: Linear Regression from scikit-learn
- **Data Processing**: StandardScaler for feature normalization
- **Model Persistence**: Joblib for saving/loading trained models
- **Frontend**: Custom HTML5 + CSS3 with gradient backgrounds and animations

## Model Performance

The model automatically trains on first run and saves the trained model for future predictions. Performance metrics are displayed in the console during training.

## File Structure

```
├── app.py                 # Flask application
├── templates/
│   └── index.html        # Main web interface
├── requirements.txt      # Python dependencies
├── housing_model.pkl     # Saved ML model (generated on first run)
├── scaler.pkl           # Saved feature scaler (generated on first run)
└── feature_names.pkl    # Saved feature names (generated on first run)
```
