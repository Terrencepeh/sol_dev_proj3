import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the preprocessed testing data
print("Loading the preprocessed testing data...")
test_data = pd.read_csv('/data/test-data.csv')
print("Data loaded successfully.")
print()

# Split the data into features and target
print("Splitting the test data into features and target...")
X_test = test_data.drop('Species', axis=1)
y_test = test_data['Species']
print(f"Features shape: {X_test.shape}")
print(f"Target shape: {y_test.shape}")
print()

# Function to evaluate a model
def evaluate_model(model_path, model_name):
    # Load the trained model
    print(f"Loading the trained model from {model_path}...")
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    print()

    # Make predictions
    print("Making predictions on the test data...")
    predictions = model.predict(X_test)
    print("Predictions completed.")
    print()

    # Evaluate the model
    print("Evaluating the model performance...")
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    # Print the evaluation metrics
    print(f"{model_name} - Model Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    
    # Combine features, predictions, and actual values for display
    results = X_test.copy()
    results['Predicted'] = predictions
    results['Actual'] = y_test.values
    
    print(f"\n{model_name} - Predictions vs Actual:")
    print(results.head(10))  # Show first 10 rows for brevity
    print("="*60)

# Evaluate the first model
print("Performance of the First Model:")
evaluate_model('/data/iris_rf_model.pkl', "Random Forest Model")

# Evaluate the second model
print("Performance of the Second Model:")
evaluate_model('/data/iris_svm_model.pkl', "Support Vector Machine Model")
