from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from data_processing import load_data, preprocess_data, split_data

# Load and preprocess the data
data = load_data('credit_data.csv')
preprocessor, X, y = preprocess_data(data)
X_train, X_test, y_train, y_test = split_data(X, y)

# Initialize models
rf_model = RandomForestClassifier(n_estimators=300, max_depth=20)
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

# Create pipelines
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', rf_model)])
gb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', gb_model)])

# Train the models
rf_pipeline.fit(X_train, y_train)
gb_pipeline.fit(X_train, y_train)

# Evaluate the models
rf_predictions = rf_pipeline.predict(X_test)
gb_predictions = gb_pipeline.predict(X_test)

print("Random Forest Evaluation:")
print(classification_report(y_test, rf_predictions))
print("ROC AUC Score:", roc_auc_score(y_test, rf_predictions))

print("\nGradient Boosting Evaluation:")
print(classification_report(y_test, gb_predictions))
print("ROC AUC Score:", roc_auc_score(y_test, gb_predictions))
