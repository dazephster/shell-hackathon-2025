import pandas as pd

def load_data(train_path = 'data/train.csv', test_path = 'data/test.csv'):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    fraction_cols = [col for col in train_df.columns if 'fraction' in col] 
    property_cols = [col for col in train_df.columns if 'Property' in col and 'Blend' not in col]
    feature_cols = fraction_cols + property_cols
    target_cols = [col for col in train_df.columns if 'BlendProperty' in col]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]
    X_test = test_df[feature_cols]
    
    return X_train, y_train, X_test, target_cols