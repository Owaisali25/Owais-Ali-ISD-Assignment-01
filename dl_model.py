"""
dl_model.py
Simple Keras feed-forward neural network for Stroke Prediction.
"""
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

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

def build_and_train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    num_idx = [i for i,c in enumerate(X_train.columns) if np.issubdtype(X_train[c].dtype, np.number)]
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled.iloc[:, num_idx] = scaler.fit_transform(X_train.iloc[:, num_idx])
    X_test_scaled.iloc[:, num_idx] = scaler.transform(X_test.iloc[:, num_idx])
    input_dim = X_train_scaled.shape[1]
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
    history = model.fit(X_train_scaled, y_train, validation_split=0.15, epochs=30, batch_size=32, class_weight=class_weight_dict)
    y_pred_prob = model.predict(X_test_scaled).ravel()
    y_pred = (y_pred_prob>=0.5).astype(int)
    from sklearn.metrics import classification_report, accuracy_score
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    model.save("dl_model.h5")
    return model, history

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv)>1 else "stroke_data.csv"
    X, y = load_and_preprocess(csv_path)
    model, history = build_and_train(X, y)
