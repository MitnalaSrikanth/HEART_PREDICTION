# Heart Disease Prediction System

A machine learning-based web application for predicting heart disease risk using patient health parameters.

## Project Structure

```
heart_disease_project/
│
├── heart.csv              ← Dataset (you can replace with your own)
├── train_model.py         ← Code to train model
├── heart_model.pkl        ← Saved model (auto-created)
├── feature_names.pkl      ← Feature names (auto-created)
├── app.py                 ← Main web app
├── requirements.txt       ← Python dependencies
├── README.md             ← This file
│
├── templates/             ← HTML files folder
│     └── index.html
│
└── static/                ← CSS and static files
      └── style.css
```

## Features

- **Machine Learning Model**: Random Forest classifier trained on heart disease data
- **Web Interface**: User-friendly form for inputting health parameters
- **Real-time Prediction**: Instant risk assessment with confidence scores
- **Responsive Design**: Works on desktop and mobile devices
- **Professional UI**: Modern Bootstrap-based interface with custom styling

## Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**:
   ```bash
   python train_model.py
   ```
   This will create `heart_model.pkl` and `feature_names.pkl` files.

4. **Run the web application**:
   ```bash
   python app.py
   ```

5. **Open your browser** and navigate to `http://localhost:5000`

## Dataset Features

The model uses the following health parameters:

- **age**: Age in years
- **sex**: Gender (1 = male, 0 = female)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting ECG results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise
- **slope**: Slope of peak exercise ST segment (0-2)
- **ca**: Number of major vessels (0-3)
- **thal**: Thalassemia (1-3)

## Using Your Own Dataset

If you want to use your own dataset:

1. **Replace `heart.csv`** with your dataset file
2. **Ensure your dataset has the same column names** as listed above
3. **Re-train the model**:
   ```bash
   python train_model.py
   ```

## API Endpoints

- **GET /**: Main web interface
- **POST /predict**: Make predictions with form data
- **GET /model_info**: Get model information and feature importance

## Model Performance

The Random Forest model typically achieves:
- **Accuracy**: ~85-90%
- **Features**: Uses all 13 health parameters
- **Algorithm**: Random Forest Classifier with 100 estimators

## Technical Stack

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **Data Processing**: Pandas, NumPy

## Disclaimer

⚠️ **Important**: This system is for educational purposes only and should not be used as a substitute for professional medical advice. Always consult with a qualified healthcare provider for medical concerns.

## Contributing

Feel free to contribute to this project by:
- Adding new features
- Improving the model
- Enhancing the UI/UX
- Adding more health parameters

## License

This project is open source and available under the MIT License.

## Support

If you encounter any issues:
1. Check that all dependencies are installed
2. Ensure the model files exist (`heart_model.pkl`, `feature_names.pkl`)
3. Verify the dataset format is correct
4. Check the console for error messages

---

**Developed with ❤️ for healthcare awareness**
