import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
from src.features import compute_blend_weighted_properties

def train_model(X_train, y_train):
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, target_cols):
    y_pred = model.predict(X_train)
    print("\nMAPE Scores (lower => better):")
    for i, col in enumerate(target_cols):
        score = mean_absolute_percentage_error(y_train.iloc[:, i], y_pred[:, i])
        print(f"{col}: {score:.4f}")
    return