import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

# Load the preprocessed training data
print("Loading the preprocessed training data...")
train_data = pd.read_csv('/data/train-data.csv')
print("Data loaded successfully.")
print()

# Split the data into features and target
print("Splitting the data into features and target...")
X_train = train_data.drop('Species', axis=1)
y_train = train_data['Species']
print(f"Features shape: {X_train.shape}")
print(f"Target shape: {y_train.shape}")
print()

# Set up hyperparameter grids
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}

# Tune RandomForest model
print("Tuning RandomForest model...")
rf_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                              param_grid=rf_param_grid,
                              cv=5,  # 5-fold cross-validation
                              n_jobs=-1,  # Use all available cores
                              verbose=2)
rf_grid_search.fit(X_train, y_train)
print("RandomForest tuning completed.")
print(f"Best RandomForest Parameters: {rf_grid_search.best_params_}")
print(f"Best RandomForest Accuracy: {rf_grid_search.best_score_}")
print()

# Tune SVM model
print("Tuning SVM model...")
svm_grid_search = GridSearchCV(estimator=SVC(random_state=42),
                               param_grid=svm_param_grid,
                               cv=5,  # 5-fold cross-validation
                               n_jobs=-1,  # Use all available cores
                               verbose=2)
svm_grid_search.fit(X_train, y_train)
print("SVM tuning completed.")
print(f"Best SVM Parameters: {svm_grid_search.best_params_}")
print(f"Best SVM Accuracy: {svm_grid_search.best_score_}")
print()

# Save the best models
print("Saving the best models to files...")
joblib.dump(rf_grid_search.best_estimator_, '/data/iris_rf_best_model.pkl')
joblib.dump(svm_grid_search.best_estimator_, '/data/iris_svm_best_model.pkl')
print("Best models saved successfully.")
