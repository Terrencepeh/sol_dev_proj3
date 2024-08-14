import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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

# Train a RandomForest model
print("Training the RandomForest model...")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
print("RandomForest model training completed.")
print()

# Display RandomForest feature importances
print("RandomForest Feature Importances:")
rf_importances = rf_model.feature_importances_
for feature, importance in zip(X_train.columns, rf_importances):
    print(f"{feature}: {importance:.4f}")
print()

# Train an SVM model
print("Training the SVM model...")
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
print("SVM model training completed.")
print()

# Save both trained models
print("Saving the trained models to files...")
joblib.dump(rf_model, '/data/iris_rf_model.pkl')
joblib.dump(svm_model, '/data/iris_svm_model.pkl')
print("Models saved successfully.")
