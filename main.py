import pandas as pd
from src.data_loader import load_data
from src.model_train import train_model, evaluate_model, run_training
from src.predict import save_predictions
from src.features import compute_blend_weighted_properties

def main():
    
    # 1.0
    # X_train, y_train, X_test, target_cols = load_data()
    
    # model = train_model(X_train, y_train)
    
    # mape_scores = evaluate_model(model, X_train, y_train)
    
    # print("\nMAPE Scores (lower => better):")
    # for i, score in enumerate(mape_scores):
    #     print(f"BlendProperty{i+1}: {score:.4f}")
        
    # save_predictions(model, X_test, target_cols)
    ###
    
    # 1.1
    train_df = pd.read_csv('data/train.csv')
    model, X_train, y_train, target_cols = run_training(train_df)
    
    test_df = pd.read_csv('data/test.csv')
    # X_test = compute_blend_weighted_properties(test_df)
    
    X_test_raw = test_df[[col for col in test_df.columns if 'Component' in col and 'Property' in col]]
    X_test_weighted = compute_blend_weighted_properties(test_df)
    X_test = pd.concat([X_test_raw, X_test_weighted], axis=1)
    
    save_predictions(model, X_test, target_cols)
    ###
    
    return
    
if __name__ == "__main__":
    main()