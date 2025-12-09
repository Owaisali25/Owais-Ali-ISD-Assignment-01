"""
ml_model.py
Preprocessing and Random Forest training script for Stroke Prediction.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df.replace(['?', 'NA', 'na', 'None', 'none', ''], np.nan, inplace=True)
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df['bmi'].fillna(df['bmi'].median(), inplace=True)
    for c in df.select_dtypes(include=['object']).columns:
        if df[c].isnull().any():
            df[c].fillna(df[c].mode().iloc[0], inplace=True)
    df.dropna(inplace=True)
    y = df['stroke'].astype(int)
    X = df.drop(columns=['stroke'])
    if 'id' in X.columns:
        X = X.drop(columns=['id'])
    X_encoded = pd.get_dummies(X, drop_first=True)
    return X_encoded, y

def train_rf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    num_idx = [i for i,c in enumerate(X_train.columns) if np.issubdtype(X_train[c].dtype, np.number)]
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled.iloc[:, num_idx] = scaler.fit_transform(X_train.iloc[:, num_idx])
    X_test_scaled.iloc[:, num_idx] = scaler.transform(X_test.iloc[:, num_idx])
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return rf, scaler

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv)>1 else "stroke_data.csv"
    X, y = load_and_preprocess(csv_path)
    rf, scaler = train_rf(X, y)
