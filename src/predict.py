import pandas as pd

def save_predictions(model, X_test, target_cols, output_path='outputs/solution.csv'):
    predictions = model.predict(X_test)
    pred_df = pd.DataFrame(predictions, columns=target_cols)
    pred_df = pred_df.astype(float)
    
    pred_df.insert(0, 'ID', range(1, len(pred_df) + 1))
    
    pred_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Predictions saved to {output_path}")
    print(pred_df.shape)
    print(pred_df.columns)
    print(pred_df.head())