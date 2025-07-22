from sklearn.metrics import mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, KFold
from catboost import CatBoostRegressor


def train_catb(X_train, y_train):
    model = MultiOutputRegressor(CatBoostRegressor(iterations=450, learning_rate=0.09, depth=4, l2_leaf_reg=2, random_state=42, verbose=0))
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

def grid_search_cat(X_train, y_train):
    base_cat = CatBoostRegressor(
        random_state=42,
        verbose=0,
        allow_writing_files=False
    )
    
    # Started with n_estimators, max_depth, and learning_rate.
    # After narrowing those down, experimented with other params
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