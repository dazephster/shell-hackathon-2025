import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def investigate():
    train_df = pd.read_csv('data/train.csv')
    
    # Get component fraction columns
    fraction_cols = [col for col in train_df.columns if 'fraction' in col]
        
    blendweight_avg_df = pd.DataFrame()
    
    for i in range(10):
        prop_str = f'Property{i+1}'
        curProp_cols = [col for col in train_df.columns if prop_str in col and 'Blend' not in col]
        
        # Start with 0s for the weighted sum
        # index (rows) same as train_df
        weighted_sum = pd.Series(0, index=train_df.index)
        
        for j in range(5):
            weighted_sum += train_df[fraction_cols[j]] * train_df[curProp_cols[j]]
        
        # Store result in new column
        blendweight_avg_df[f'BlendWeighted_{prop_str}'] = weighted_sum
    
    # merge new weighted_avgs df with train_df
    combined_df = pd.merge(train_df, blendweight_avg_df, left_index=True, right_index=True)
    
    return combined_df
    
def compute_correlation_matrix(df):
    target_cols = [f'BlendProperty{i+1}' for i in range(10)]
    feature_cols = [f'BlendWeighted_Property{i+1}' for i in range(10)]
    
    corr_matrix = pd.DataFrame(index=target_cols, columns=feature_cols)
    
    for target in target_cols:
        for feature in feature_cols:
            corr_matrix.loc[target, feature] = df[target].corr(df[feature])
    
    return corr_matrix.astype(float);

def plot_correlation_heatmap(corr_mx):
    plt.figure(figsize=(10,8))
    
    sns.heatmap(
        corr_mx,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )
    
    plt.title("Correlation: Blend Targets vs. Weighted Component Properties")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    return

df = investigate()
corr_mx = compute_correlation_matrix(df)
plot_correlation_heatmap(corr_mx)