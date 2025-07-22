import pandas as pd
import numpy as np

def save_predictions(model, X_test, target_cols, output_path='outputs/solution.csv'):
    predictions = model.predict(X_test)
    pred_df = pd.DataFrame(predictions, columns=target_cols)
    pred_df = pred_df.astype(float)
    
    avg_bp3_bp7 = 0.5 * pred_df['BlendProperty3'] + 0.5 * pred_df['BlendProperty7']
    pred_df['BlendProperty3'] = avg_bp3_bp7
    pred_df['BlendProperty7'] = avg_bp3_bp7
    
    pred_df.insert(0, 'ID', range(1, len(pred_df) + 1))
    
    pred_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Predictions saved to {output_path}")
    print(pred_df.shape)
    print(pred_df.columns)
    print(pred_df.head())
    
def save_predictions_meta(base_model, meta_model, X_test, target_cols, output_path='outputs/solution_meta.csv'):
    base_predictions = base_model.predict(X_test)
    final_predictions = meta_model.predict(base_predictions)
    pred_df = pd.DataFrame(final_predictions, columns=target_cols)
    pred_df = pred_df.astype(float)
    
    pred_df.insert(0, 'ID', range(1, len(pred_df) + 1))
    
    pred_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Predictions saved to {output_path}")
    print(pred_df.shape)
    print(pred_df.columns)
    print(pred_df.head())
    
def save_predictions_per_prop(models_dict, X_test, target_cols, output_path='outputs/solution_per_prop.csv'):
    all_preds = []

    for prop in target_cols:
        model, model_type = models_dict[prop]
        preds = model.predict(X_test)
        all_preds.append(preds.reshape(-1, 1))  # Ensure 2D column shape

    predictions = np.hstack(all_preds)
    pred_df = pd.DataFrame(predictions, columns=target_cols)
    pred_df = pred_df.astype(float)

    pred_df.insert(0, 'ID', range(1, len(pred_df) + 1))

    pred_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Predictions saved to {output_path}")
    print(pred_df.shape)
    print(pred_df.columns)
    print(pred_df.head())
    
def save_predictions_reduced(model, X_test, original_target_cols, reduced_target_cols, output_path='outputs/solution.csv'):
    preds = model.predict(X_test)
    
    # Prepare full prediction array
    full_preds = np.zeros((preds.shape[0], len(original_target_cols)))
    idx_map = {col: i for i, col in enumerate(original_target_cols)}
    
    # Fill in reduced predictions
    for i, col in enumerate(reduced_target_cols):
        full_preds[:, idx_map[col]] = preds[:, i]
    
    # Copy BP3 predictions to BP7
    full_preds[:, idx_map['BlendProperty7']] = full_preds[:, idx_map['BlendProperty3']]
    
    # Create DataFrame with original target columns
    pred_df = pd.DataFrame(full_preds, columns=original_target_cols)
    pred_df = pred_df.astype(float)
    
    # Add ID column starting at 1
    pred_df.insert(0, 'ID', range(1, len(pred_df) + 1))
    
    pred_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Predictions saved to {output_path}")
    print(pred_df.shape)
    print(pred_df.columns)
    print(pred_df.head())
    
def save_predictions_unique_bp1(bp1_preds, model, X_test, target_cols, output_path='outputs/solution.csv'):
    predictions = model.predict(X_test)
    pred_df = pd.DataFrame(predictions, columns=target_cols)
    pred_df["BlendProperty1"] = bp1_preds
    pred_df = pred_df.astype(float)
    
    pred_df.insert(0, 'ID', range(1, len(pred_df) + 1))
    
    pred_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Predictions saved to {output_path}")
    print(pred_df.shape)
    print(pred_df.columns)
    print(pred_df.head())