import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
from src.features import compute_blend_weighted_properties

def train_model(X_train, y_train):
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=20, random_state=42))
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, target_cols):
    y_pred = model.predict(X_train)
    # mape_scores = []
    print("\nMAPE Scores (lower => better):")
    # for i in range(y_train.shape[1]):
    #     mape = mean_absolute_percentage_error(y_train.iloc[:, i], y_pred[:, i])
    #     mape_scores.append(mape)
    # return mape_scores
    for i, col in enumerate(target_cols):
        score = mean_absolute_percentage_error(y_train.iloc[:, i], y_pred[:, i])
        print(f"{col}: {score:.4f}")
    return

def run_training(train_df):
    target_cols = [col for col in train_df.columns if 'BlendProperty' in col]
    y_train = train_df[target_cols]
    
    # raw
    raw_feature_cols = [col for col in train_df.columns if 'Component' in col and 'Property' in col]
    X_train_raw = train_df[raw_feature_cols]
    
    # weighted
    X_train_weighted = compute_blend_weighted_properties(train_df)
    
    X_train_full = pd.concat([X_train_raw, X_train_weighted], axis=1)
    
    model = train_model(X_train_full, y_train)
    evaluate_model(model, X_train_full, y_train, target_cols)
    
    return model, X_train_full, y_train, target_cols