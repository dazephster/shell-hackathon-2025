import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
from src.features import compute_blend_weighted_properties
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from xgboost import XGBRegressor, XGBRFRegressor

def train_model(X_train, y_train):
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)
    return model

def train_xgb(X_train, y_train):
    # XGB RF Regressor?
    model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1))
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, target_cols):
    y_pred = model.predict(X_train)
    print("\nMAPE Scores (lower => better):")
    for i, col in enumerate(target_cols):
        score = mean_absolute_percentage_error(y_train.iloc[:, i], y_pred[:, i])
        print(f"{col}: {score:.4f}")
    return

def cross_validate_xgb(X_train, y_train, target_cols, n_splits=5):
    print(f"\nCross-Validation with {n_splits} folds")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    mape_scores = []
    
    for i, col in enumerate(target_cols):
        model = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1
        )
        
        scores = cross_val_score(
            model,
            X_train,
            y_train[col],
            scoring="neg_mean_absolute_percentage_error",
            cv=kf,
        )
        
        mean_mape = -np.mean(scores)
        mape_scores.append((col, mean_mape))
        print(f"{col}: {mean_mape:.4f}")
        
    overall_mean = np.mean([m[1] for m in mape_scores])
    print(f"\nAverage CV MAPE: {overall_mean:.4f}")
    return mape_scores

def grid_search_xgb(X_train, y_train):
    base_xgb = XGBRegressor(
        n_jobs=-1,
        random_state=42,
        objective='reg:squarederror'
    )
    
    param_grid = {
        'estimator__n_estimators': [100, 300],
        'estimator__max_depth': [3, 5, 7],
        'estimator__learning_rate': [0.01, 0.05, 0.1],
        'estimator__subsample': [0.8, 1.0],
        'estimator__colsample_bytree': [0.8, 1.0]
    }
    
    model = MultiOutputRegressor(base_xgb)
    
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    
    # Can't use MAPE here
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=kf,
        scoring='neg_mean_absolute_error',
        verbose=2,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best Params: {grid_search.best_params_}")
    print(f"Best Score: {-grid_search.best_score_}")
    
    return grid_search.best_estimator_