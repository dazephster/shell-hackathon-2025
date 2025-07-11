import pandas as pd

def compute_blend_weighted_properties(df: pd.DataFrame) -> pd.DataFrame:
    blend_weighted = pd.DataFrame(index=df.index)
    fraction_cols = [col for col in df.columns if 'fraction' in col]
    
    for i in range(10):
        prop = f'Property{i+1}'
        prop_cols = [f'Component{j+1}_{prop}' for j in range(5)]
        weighted_sum = sum(df[fraction_cols[j]] * df[prop_cols[j]] for j in range(5))
        blend_weighted[f'BlendWeighted_Property{i+1}'] = weighted_sum
        
    return blend_weighted