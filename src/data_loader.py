import pandas as pd
import matplotlib.pyplot as plt

from src.features import compute_blend_weighted_properties

def load_data(mode = 'both', visualize = True, train_path = 'data/train.csv', test_path = 'data/test.csv'):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    fraction_cols = [col for col in train_df.columns if 'fraction' in col]
    property_cols = [col for col in train_df.columns if 'Property' in col and 'Blend' not in col]
    feature_cols = fraction_cols + property_cols
    target_cols = [col for col in train_df.columns if 'BlendProperty' in col]
    
    X_train_raw = train_df[feature_cols]
    X_train_weighted = compute_blend_weighted_properties(train_df)
    X_test_raw = test_df[feature_cols]
    X_test_weighted = compute_blend_weighted_properties(test_df)

    if mode == 'raw':
        X_train = X_train_raw
        X_test = X_test_raw
        print("Training with RAW features")
    elif mode == 'weighted':
        X_train = pd.concat([X_train_weighted, train_df[property_cols]], axis=1)
        X_test = pd.concat([X_test_weighted, test_df[property_cols]], axis=1)
        print("Training with WEIGHTED + property features")
    else:  # both
        X_train = pd.concat([X_train_weighted, X_train_raw], axis=1)
        X_test = pd.concat([X_test_weighted, X_test_raw], axis=1)
        print("Training with BOTH weighted and raw features")

    y_train = train_df[target_cols]
    
    if visualize:
        # visualize_data(train_df, test_df)
        # visualize_BP1(X_train,y_train)
        print('Visualized')
    
    return X_train, y_train, X_test, target_cols

def visualize_data(train_df, test_df):
    # Example: Visualize Component1 fraction vs. BlendProperty1
    # for i in range(1,11):
    #     plt.scatter(train_df['Component4_fraction'], train_df[f'BlendProperty{i}'])
    #     plt.xlabel('Component4 Fraction')
    #     plt.ylabel(f'BlendProperty{i}')
    #     plt.title(f'Component4 Fraction vs BlendProperty{i}')
    #     plt.show()
    return
    
def visualize_BP1(X_train, y_train):
    for weighted_prop_col in [col for col in X_train.columns if 'BlendWeighted_Property' in col]:
        plt.scatter(X_train[weighted_prop_col], y_train['BlendProperty1'])
        plt.xlabel(weighted_prop_col)
        plt.ylabel(f'BlendProperty1')
        plt.title(f'{weighted_prop_col} vs BlendProperty1')
        plt.show()
        
    return