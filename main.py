import pandas as pd
from src.data_loader import load_data
from src.model_train import train_model, evaluate_model
from src.predict import save_predictions

def main():
    
    # 1.2
    X_train, y_train, X_test, target_cols = load_data(mode='both')
    
    model = train_model(X_train, y_train)
    
    evaluate_model(model, X_train, y_train, target_cols)
        
    save_predictions(model, X_test, target_cols)
    return
    
if __name__ == "__main__":
    main()