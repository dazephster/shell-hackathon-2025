import pandas as pd
from src.data_loader import load_data
from src.model_train import train_xgb, train_catb, train_lightb, train_stacking, evaluate_reduced_model, train_and_pred_bp1, grid_search_cat, train_pls, apply_pca, evaluate_model_pca
from src.model_train import grid_search_xgb, train_model, evaluate_model, evaluate_meta_model, cross_validate_xgb, train_per_property_models, train_per_property_kfold
from src.predict import save_predictions, save_predictions_per_prop, save_predictions_meta, save_predictions_reduced, save_predictions_unique_bp1
from investigate import plot_target_correlations, plot_spearman_correlations

def main():
    # both gets raw data plus computed weighted avgs cols
    X_train, y_train, X_test, target_cols = load_data(mode='both')
    
    # plot_target_correlations(y_train)
    # plot_spearman_correlations(y_train)
    
    # 1.6 Light/Cat
    
    # xgb_model = train_xgb(X_train, y_train)
    # evaluate_model(xgb_model, X_train, y_train, target_cols, "XGB")
    # save_predictions(xgb_model, X_test, target_cols, 'outputs/solution_xgb.csv')
    
    # light_model = train_lightb(X_train, y_train)
    # evaluate_model(light_model, X_train, y_train, target_cols, "LightGBM")
    # save_predictions(light_model, X_test, target_cols, 'outputs/solution_light.csv')
    
    cat_model = train_catb(X_train, y_train)
    evaluate_model(cat_model, X_train, y_train, target_cols, "CatBoost")
    save_predictions(cat_model, X_test, target_cols, 'outputs/solution_cat.csv')
    
    # 1.7 Per Property Training
    
    # models_dict, mape_dict = train_per_property_models(X_train, y_train)
    
    # models_dict, mape_dict = train_per_property_kfold(X_train, y_train)
    
    # print("\nFinal Best Models Per Property:")
    # for prop, (model, model_type) in models_dict.items():
    #     print(f"{prop}: {model_type.upper()}, MAPE: {mape_dict[prop]:.4f}")
        
    # save_predictions_per_prop(models_dict, X_test, target_cols)
    
    # 1.8 Ridge Meta Model
    
    # base_model, meta_model = train_stacking(X_train, y_train, "XGB")
    # evaluate_meta_model(base_model, meta_model, X_train, y_train, target_cols)
    # save_predictions_meta(base_model, meta_model, X_test, target_cols)
    
    # 1.9 Testing dropping 1.00 correlation
    
    # targets_to_use = [col for col in y_train.columns if col != 'BlendProperty7']
    # xgb_model = train_xgb(X_train, y_train[targets_to_use])
    # evaluate_reduced_model(xgb_model, X_train, y_train, targets_to_use, "XGB")
    # save_predictions_reduced(xgb_model, X_test, y_train.columns, targets_to_use, 'outputs/solution_xgb_red.csv')
    
    # cat_model = train_catb(X_train, y_train[targets_to_use])
    # evaluate_reduced_model(cat_model, X_train, y_train, targets_to_use, "CatBoost")
    # save_predictions_reduced(cat_model, X_test, y_train.columns, targets_to_use, 'outputs/solution_cat_red.csv')
    
    # 1.10 Testing Diff Params for Some BPs
    
    # bp1_preds = train_and_pred_bp1(X_train, y_train, X_test)
    # save_predictions_unique_bp1(bp1_preds, cat_model, X_test, target_cols, 'outputs/solution_cat_un_bp1.csv')
    
    # 1.11 GridSearchCV Cat
    
    # grid_search_cat(X_train, y_train)
    
    # 1.12 PLS
    
    # pls_model = train_pls(X_train, y_train)
    # evaluate_model(pls_model, X_train, y_train, target_cols, 'PLS')
    # save_predictions(pls_model, X_test, target_cols, 'outputs/solution_pls.csv')
    
    # 1.13 PCA
    
    # pca, y_train_pca = apply_pca(y_train)
    # cat_model = train_catb(X_train, y_train_pca)
    # evaluate_model_pca(cat_model, pca, X_train, y_train, target_cols, "CatBoost")
    # train preds much worse, not going to make test preds function
    
    
    return
    
if __name__ == "__main__":
    main()