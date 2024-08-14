import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('/data/iris.csv')

print('Viewing data....')
print('Input shape:', data.shape)
print()
print('Columns:', data.columns)
print()
print(data.describe())
print()

# Check for null and duplicates
print('Checking for null and duplicates....')
print("Null count:")
print(data.isnull().sum())
print("Duplicated count:")
print(data.duplicated().sum())
print()

# Deal with null and duplicates
print('Dropping null and duplicates....')
data = data.dropna()
data = data.drop_duplicates()

X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['Species']

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print()

# Split the dataset into training and testing sets
print('Splitting data into train and test....')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine features and target into training and testing datasets
train_data = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
test_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)

# Print the file paths
train_file_path = '/data/train-data.csv'
test_file_path = '/data/test-data.csv'

try:
    # Save the processed datasets
    train_data.to_csv(train_file_path, index=False)
    test_data.to_csv(test_file_path, index=False)
    print(f"Preprocessing completed and data saved.")
    print(f"Training data saved at: {train_file_path}")
    print(f"Testing data saved at: {test_file_path}")
except Exception as e:
    print(f"Error occurred: {e}")

# Additional data exploration on the saved files
print('Summary statistics of the saved files:')
try:
    saved_train_data = pd.read_csv(train_file_path)
    saved_test_data = pd.read_csv(test_file_path)

    print("Training data summary:")
    print(saved_train_data.describe())
    print()

    print("Testing data summary:")
    print(saved_test_data.describe())
except Exception as e:
    print(f"Error occurred while reading saved files: {e}")
