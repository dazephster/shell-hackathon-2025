Shell Hackathon 2025 Project (https://github.com/dazephster/shell-hackathon-2025/)

How to run:
1) Install dependencies with pip install -r requirements.txt
2) Run python main.py
3) Optional: Add in commented functions or set visualize to true in load_data function

Pearson Correlations
<img width="992" height="525" alt="BPCorrelations" src="https://github.com/user-attachments/assets/7e22138b-1e7b-4b63-ab7b-660d41f52181" />

Weighted Properties Heatmap
<img width="1260" height="800" alt="WBPHeatmap" src="https://github.com/user-attachments/assets/2e517b94-527d-496f-9613-2a19011b9ce7" />


This project predicts 10 blend property values based on component fractions and component property data.

I was new to Data Analysis and Machine Learning before this project.
My process started with decision trees and random forests. After then I added the blend weighted property average columns. I adjusted to using gradient boosting, and attempted a bunch of other methods like per property training, meta modeling with Ridge Regression, predicting individual BPs, and more.

The best-performing model for me was a CatBoostRegressor wrapped in MultiOutputRegressor. 

Feature engineering includes weighted averages of component properties based on their fractions (compute_blend_weighted_properties in features.py).

I used GridSearchCV to tune the hyperparameters on CatBoost. Optimal hyperparameters found:

    n_estimators: 450

    max_depth: 4

    learning_rate: 0.09

    l2_leaf_reg: 2

I also added a small post-processing step on BlendProperty3 and 7 because they were so highly correlated in the test data. I wasn't sure what else to do for this but it is a situation that I'd like to understand further for future problems.

    avg_bp3_bp7 = 0.5 * pred_df['BlendProperty3'] + 0.5 * pred_df['BlendProperty7']
    pred_df['BlendProperty3'] = avg_bp3_bp7
    pred_df['BlendProperty7'] = avg_bp3_bp7

Final Training Data Results when Training with BOTH weighted and raw features
Evaluating Model Type: CatBoost

MAPE Scores (lower => better):
BlendProperty1: 0.4910
BlendProperty2: 0.2064
BlendProperty3: 0.4981
BlendProperty4: 0.2354
BlendProperty5: 0.1395
BlendProperty6: 0.1419
BlendProperty7: 0.8134
BlendProperty8: 0.3358
BlendProperty9: 0.7789
BlendProperty10: 0.1770

Submission Score: 85.0065
