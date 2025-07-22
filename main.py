from src.data_loader import load_data
from src.model_train import train_catb, grid_search_cat, evaluate_model
from src.predict import save_predictions
from investigate import plot_target_correlations, plot_spearman_correlations, investigate_heatmap

def main():
    # "both" mode gets raw data plus computed weighted avgs cols
    # Both always performed better than just raw or weighted only
    # Set visualize param to True and uncomment functions in data_loader.py to see more visualizations
    X_train, y_train, X_test, target_cols = load_data(mode='both', visualize=False)
    
    # Plotting functions
    # investigate_heatmap()
    # plot_target_correlations(y_train)
    # plot_spearman_correlations(y_train)
    
    # GridSearchCV for CatBoost Hyperparameters
    # grid_search_cat(X_train, y_train)
    
    # Main training, evaluating, predicting functions
    cat_model = train_catb(X_train, y_train)
    evaluate_model(cat_model, X_train, y_train, target_cols, "CatBoost")
    save_predictions(cat_model, X_test, target_cols, 'outputs/solution_cat.csv')
    
    return
    
if __name__ == "__main__":
    main()