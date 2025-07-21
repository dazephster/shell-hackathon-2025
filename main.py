import pandas as pd
from src.data_loader import load_data
from src.model_train import grid_search_xgb, train_model, evaluate_model, train_xgb, cross_validate_xgb
from src.predict import save_predictions

def main():
    
    # 1.2
    X_train, y_train, X_test, target_cols = load_data(mode='both', visualize=True)
    
    # 1.3 XGB
    model = train_xgb(X_train, y_train)
    
    # 1.4 CV
    # cross_validate_xgb(X_train, y_train, target_cols, 5)
    
    # 1.5 GridSearch
    # model = grid_search_xgb(X_train, y_train)
    
    evaluate_model(model, X_train, y_train, target_cols)
        
    save_predictions(model, X_test, target_cols)
    return
    
if __name__ == "__main__":
    main()