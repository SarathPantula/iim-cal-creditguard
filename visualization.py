import matplotlib.pyplot as plt
import shap
from data_processing import load_data, preprocess_data
from model_analysis import rf_pipeline

# Load and preprocess data for visualization
data = load_data('credit_data.csv')
preprocessor, X, _ = preprocess_data(data)

# Compute SHAP values
explainer = shap.TreeExplainer(rf_pipeline.named_steps['model'])
shap_values = explainer.shap_values(X)

# Summary plot of SHAP values
shap.summary_plot(shap_values, X, plot_type="bar")

# Feature importance plot
shap.summary_plot(shap_values, X)

# Detailed SHAP value plot for a specific instance
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
