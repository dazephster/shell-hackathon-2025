import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from src.features import compute_blend_weighted_properties
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from xgboost import XGBRegressor, XGBRFRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def train_model(X_train, y_train):
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)
    return model

# Notes
# Init
# n_est=100, learn_rate=0.1, depth=5,
# More Tuned
# n_est=300, learn_rate=0.05, depth=6/8,
# Overtuned?
# n_est=500, learn_rate=0.01, depth=8


# 
# Single Model Training Function
# 
def train_single_model(X_train, y_train, X_val, y_val, model_type):
    if model_type == "xgb":
        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, eval_metric="mae", early_stopping_rounds=20, verbosity=0)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)]
        )
    elif model_type == "lgb":
        model = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, early_stopping_rounds=20, verbose=-1)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
        )
    elif model_type == "cat":
        model = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=6, random_state=42, verbose=0, early_stopping_rounds=20)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        raise ValueError("Unknown model type")

    preds = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, preds)
    return model, mape

#
# Per-property Comparison
#
def train_per_property_models(X_train, y_train):
    models_dict = {}
    mape_dict = {}

    # 5-fold Validation Split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    for prop in y_train.columns:
        print(f"\nTraining models for: {prop}")
        y_tr_prop = y_tr[prop]
        y_val_prop = y_val[prop]

        best_mape = float('inf')
        best_model = None
        best_type = None

        for model_type in ["xgb", "lgb", "cat"]:
            model, mape = train_single_model(X_tr, y_tr_prop, X_val, y_val_prop, model_type)
            print(f"{model_type.upper()} MAPE: {mape:.4f}")
            if mape < best_mape:
                best_mape = mape
                best_model = model
                best_type = model_type

        models_dict[prop] = (best_model, best_type)
        mape_dict[prop] = best_mape
        print(f"Best model for {prop}: {best_type.upper()} with MAPE {best_mape:.4f}")

    return models_dict, mape_dict

def train_and_pred_bp1(X_train, y_train, X_test):
    X_train_bp1, X_val_bp1, y_train_bp1, y_val_bp1 = train_test_split(
        X_train, y_train["BlendProperty1"], test_size=0.2, random_state=42
    )
    model_bp1 = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.02,
        depth=7,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )

    model_bp1.fit(
        X_train_bp1, 
        y_train_bp1, 
        eval_set=(X_val_bp1, y_val_bp1)
    )
    
    preds_bp1 = model_bp1.predict(X_val_bp1)
    mape_bp1 = mean_absolute_percentage_error(y_val_bp1, preds_bp1)
    print(f"BlendProperty1 Unique MAPE: {mape_bp1:.4f}")
    
    min_bp1 = y_train_bp1.min()
    max_bp1 = y_train_bp1.max()

    preds_bp1_test = model_bp1.predict(X_test)
    preds_bp1_test = preds_bp1_test.clip(min_bp1, max_bp1)
    
    return preds_bp1_test

def train_per_property_kfold(X, y, model_types=["xgb", "lgb", "cat"], k=5):
    models_dict = {}
    mape_dict = {}

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for prop in y.columns:
        print(f"\nTraining models for: {prop}")

        fold_mapes = {mt: [] for mt in model_types}
        fold_models = {mt: [] for mt in model_types}

        y_prop = y[prop].values

        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr_prop, y_val_prop = y_prop[train_idx], y_prop[val_idx]

            for model_type in model_types:
                model, mape = train_single_model(X_tr, y_tr_prop, X_val, y_val_prop, model_type)
                fold_mapes[model_type].append(mape)
                fold_models[model_type].append(model)

        # Calculate average MAPE per model type
        avg_mapes = {mt: np.mean(fold_mapes[mt]) for mt in model_types}
        best_type = min(avg_mapes, key=avg_mapes.get)
        best_mape = avg_mapes[best_type]

        # Choose the model from the fold with best validation MAPE for that model_type
        best_fold_idx = np.argmin(fold_mapes[best_type])
        best_model = fold_models[best_type][best_fold_idx]

        print(f"Average MAPE per model: {avg_mapes}")
        print(f"Best model for {prop}: {best_type.upper()} with average MAPE {best_mape:.4f}")

        models_dict[prop] = (best_model, best_type)
        mape_dict[prop] = best_mape

    return models_dict, mape_dict



# For Later param/model comparison
def compare_models(X_train, y_train):
    estimators = [50, 100, 300]
    depths = [3, 5, 7]
    learn_rates = [0.01, 0.05, 0.1]
    
    for i in range(10):
        curRand = random.randint(1, 27)
        est_idx = curRand % 3
        curRand /= 3
        d_idx = curRand % 3
        curRand /= 3
        learn_idx = curRand % 3
        
        # Run train property model on all models with the given indexes here?
    
    return  

def train_xgb(X_train, y_train):
    # XGB RF Regressor?
    model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1))
    model.fit(X_train, y_train)
    return model

def train_catb(X_train, y_train):
    model = MultiOutputRegressor(CatBoostRegressor(iterations=450, learning_rate=0.09, depth=4, l2_leaf_reg=2, random_state=42, verbose=0))
    model.fit(X_train, y_train)
    return model

def train_lightb(X_train, y_train):
    model = MultiOutputRegressor(LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1))
    model.fit(X_train, y_train)
    return model
    

def evaluate_model(model, X_train, y_train, target_cols, model_type="asdf"):
    print(f"\nEvaluating Model Type: {model_type}")
    y_pred = model.predict(X_train)
    print("\nMAPE Scores (lower => better):")
    for i, col in enumerate(target_cols):
        score = mean_absolute_percentage_error(y_train.iloc[:, i], y_pred[:, i])
        print(f"{col}: {score:.4f}")
    return

def evaluate_reduced_model(model, X_train, y_train, target_cols, model_type="asdf"):
    print(f"\nEvaluating Model Type: {model_type}")
    preds = model.predict(X_train)
    full_preds = np.zeros((preds.shape[0], len(y_train.columns)))
    idx_map = {col: i for i, col in enumerate(y_train.columns)}
    for i, col in enumerate(target_cols):
        full_preds[:, idx_map[col]] = preds[:, i]
    full_preds[:, idx_map['BlendProperty7']] = full_preds[:, idx_map['BlendProperty3']]
    
    print("\nMAPE Scores (lower => better):")
    for col in y_train.columns:
        i = idx_map[col]
        score = mean_absolute_percentage_error(y_train[col], full_preds[:, i])
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
        'estimator__max_depth': [3, 5],
        'estimator__learning_rate': [0.01, 0.05, 0.1],
        'estimator__subsample': [0.67, 1.0],
        'estimator__colsample_bytree': [0.67, 1.0]
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

def train_stacking(X_train, y_train, model="CatBoost"):
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    if model == "CatBoost":
        base_model = MultiOutputRegressor(CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, random_state=42, verbose=0))
    elif model == "XGB":
        base_model = MultiOutputRegressor(XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1))
    else:
        raise ValueError("Invalid model name")
    
    base_model.fit(X_tr, y_tr)
    
    base_preds_meta = base_model.predict(X_val)
    
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(base_preds_meta, y_val)
    
    return base_model, meta_model

def evaluate_meta_model(base_model, meta_model, X_train, y_train, target_cols):
    base_preds = base_model.predict(X_train)
    final_preds = meta_model.predict(base_preds)
    print(f"Base predictions shape: {base_preds.shape}")
    print(f"Final predictions shape: {final_preds.shape}")
    print("\nMAPE Scores (lower => better):")
    for i, col in enumerate(target_cols):
        score = mean_absolute_percentage_error(y_train.iloc[:, i], final_preds[:, i])
        print(f"{col}: {score:.4f}")
    return

def grid_search_cat(X_train, y_train):
    base_cat = CatBoostRegressor(
        random_state=42,
        verbose=0,
        allow_writing_files=False
    )
    
    param_grid = {
        'estimator__n_estimators': [450],
        'estimator__max_depth': [4],
        'estimator__learning_rate': [0.09],
        'estimator__l2_leaf_reg': [2],
        'estimator__random_strength': [0.5, 1, 2],
        'estimator__border_count': [64, 128, 254]
    }
    
    model = MultiOutputRegressor(base_cat)
    
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

def train_pls(X_train, y_train):
    model = PLSRegression(n_components=10)
    model.fit(X_train, y_train)
    return model

def apply_pca(y_train):
    pca = PCA(n_components=0.9) # try 0.9 too?
    y_train_pca = pca.fit_transform(y_train)
    print(f"Original targets: {y_train.shape[1]}, PCA components: {y_train_pca.shape[1]}")
    return pca, y_train_pca

def evaluate_model_pca(model, pca, X_train, y_train, target_cols, model_type="asdf"):
    print(f"\nEvaluating PCA with Model Type: {model_type}")
    y_pred_pca = model.predict(X_train)
    y_pred = pca.inverse_transform(y_pred_pca)
    print("\nMAPE Scores (lower => better):")
    for i, col in enumerate(target_cols):
        score = mean_absolute_percentage_error(y_train.iloc[:, i], y_pred[:, i])
        print(f"{col}: {score:.4f}")
    return