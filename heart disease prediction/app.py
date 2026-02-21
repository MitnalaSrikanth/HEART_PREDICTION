from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model and feature names
try:
    with open('heart_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    with open('feature_names.pkl', 'rb') as file:
        feature_names = pickle.load(file)
    
    print("Model and feature names loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    feature_names = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/test')
def test():
    return jsonify({
        'model_loaded': model is not None,
        'feature_names_loaded': feature_names is not None,
        'model_type': type(model).__name__ if model else None,
        'feature_names': feature_names if feature_names else None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'})
    
    try:
        # Get form data
        data = request.form
        
        # Create feature array in the correct order
        features = []
        feature_mapping = {
            'age': float(data['age']),
            'sex': float(data['sex']),
            'cp': float(data['cp']),
            'trestbps': float(data['trestbps']),
            'chol': float(data['chol']),
            'fbs': float(data['fbs']),
            'restecg': float(data['restecg']),
            'thalach': float(data['thalach']),
            'exang': float(data['exang']),
            'oldpeak': float(data['oldpeak']),
            'slope': float(data['slope']),
            'ca': float(data['ca']),
            'thal': float(data['thal'])
        }
        
        # Ensure features are in the correct order
        for feature in feature_names:
            features.append(feature_mapping[feature])
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        prediction_proba = model.predict_proba(features_array)[0]
        
        # Interpret results
        if prediction == 1:
            result = "High Risk of Heart Disease"
            risk_level = "High"
            confidence = prediction_proba[1] * 100
        else:
            result = "Low Risk of Heart Disease"
            risk_level = "Low"
            confidence = prediction_proba[0] * 100
        
        return jsonify({
            'prediction': int(prediction),
            'result': result,
            'risk_level': risk_level,
            'confidence': round(confidence, 2),
            'probability_high_risk': round(prediction_proba[1] * 100, 2),
            'probability_low_risk': round(prediction_proba[0] * 100, 2)
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'})

@app.route('/model_info')
def model_info():
    if model is None:
        return jsonify({'error': 'Model not loaded'})
    
    try:
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, model.feature_importances_.tolist()))
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        else:
            feature_importance = {}
        
        return jsonify({
            'model_type': type(model)._name_,
            'features': feature_names,
            'feature_importance': feature_importance
        })
    
    except Exception as e:
        return jsonify({'error': f'Error getting model info: {str(e)}'})

if __name__ == '__main__':
    # Check if model exists, if not, suggest training
    if not os.path.exists('heart_model.pkl'):
        print("Warning: Model file not found. Please run 'python train_model.py' first.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)