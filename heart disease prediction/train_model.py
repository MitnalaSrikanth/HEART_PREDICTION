import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('heart.csv')

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nDataset columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print("\nTarget distribution:")
print(df['target'].value_counts())

# Check if dataset has descriptive names or standard names
descriptive_columns = ['chest_pain_type', 'resting_blood_pressure', 'cholestoral', 'fasting_blood_sugar', 'rest_ecg', 'Max_heart_rate', 'exercise_induced_angina', 'vessels_colored_by_flourosopy', 'thalassemia']
standard_columns = ['cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'ca', 'thal']

if any(col in df.columns for col in descriptive_columns):
    print("\nDetected descriptive column names - converting to standard format...")
    
    # Map descriptive column names to standard names
    column_mapping = {
        'age': 'age',
        'sex': 'sex',
        'chest_pain_type': 'cp',
        'resting_blood_pressure': 'trestbps',
        'cholestoral': 'chol',
        'fasting_blood_sugar': 'fbs',
        'rest_ecg': 'restecg',
        'Max_heart_rate': 'thalach',
        'exercise_induced_angina': 'exang',
        'oldpeak': 'oldpeak',
        'slope': 'slope',
        'vessels_colored_by_flourosopy': 'ca',
        'thalassemia': 'thal',
        'target': 'target'
    }
    
    # Rename columns to standard names
    df = df.rename(columns=column_mapping)
    
    # Convert categorical values to numerical
    print("Converting categorical values to numerical...")
    
    # Convert sex
    if df['sex'].dtype == 'object':
        df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    
    # Convert chest pain type
    if df['cp'].dtype == 'object':
        cp_mapping = {
            'Typical angina': 0,
            'Atypical angina': 1,
            'Non-anginal pain': 2,
            'Asymptomatic': 3
        }
        df['cp'] = df['cp'].map(cp_mapping)
    
    # Convert fasting blood sugar
    if df['fbs'].dtype == 'object':
        fbs_mapping = {
            'Lower than 120 mg/ml': 0,
            'Greater than 120 mg/ml': 1,
            'False': 0,
            'True': 1
        }
        df['fbs'] = df['fbs'].map(fbs_mapping)
    
    # Convert rest ECG
    if df['restecg'].dtype == 'object':
        restecg_mapping = {
            'Normal': 0,
            'ST-T wave abnormality': 1,
            'Left ventricular hypertrophy': 2
        }
        df['restecg'] = df['restecg'].map(restecg_mapping)
    
    # Convert exercise induced angina
    if df['exang'].dtype == 'object':
        df['exang'] = df['exang'].map({'Yes': 1, 'No': 0})
    
    # Convert slope
    if df['slope'].dtype == 'object':
        slope_mapping = {
            'Upsloping': 0,
            'Flat': 1,
            'Downsloping': 2
        }
        df['slope'] = df['slope'].map(slope_mapping)
    
    # Convert vessels colored by fluoroscopy
    if df['ca'].dtype == 'object':
        ca_mapping = {
            'Zero': 0,
            'One': 1,
            'Two': 2,
            'Three': 3,
            'Four': 4
        }
        df['ca'] = df['ca'].map(ca_mapping)
    
    # Convert thalassemia
    if df['thal'].dtype == 'object':
        thal_mapping = {
            'Normal': 1,
            'Fixed Defect': 2,
            'Reversable Defect': 3,
            'No': 0
        }
        df['thal'] = df['thal'].map(thal_mapping)
    
    print("Dataset after preprocessing:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)

elif all(col in df.columns for col in standard_columns):
    print("\nDetected standard Kaggle format - using as-is...")
    # Dataset is already in standard format
    pass
else:
    print("\nWarning: Unrecognized dataset format. Please check column names.")
    print("Expected columns:", standard_columns + ['age', 'sex', 'oldpeak', 'slope', 'target'])
    print("Actual columns:", df.columns.tolist())

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Create and train the Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Save the trained model
print("\nSaving model to heart_model.pkl...")
with open('heart_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model training completed successfully!")
print(f"Model saved as 'heart_model.pkl'")
print(f"Accuracy: {accuracy:.4f}")

# Save feature names for later use
feature_names = X.columns.tolist()
with open('feature_names.pkl', 'wb') as file:
    pickle.dump(feature_names, file)

print("Feature names saved as 'feature_names.pkl'")
