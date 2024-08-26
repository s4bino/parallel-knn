from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import time

print("Sklearn")

# Load the Digits dataset
data = load_digits()
X = data.data
y = data.target

# Split the data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Measure the execution time
start_time = time.time()

# Initialize the sklearn KNN model
knn_sklearn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn_sklearn.fit(X_train, y_train)

# Make predictions
predictions_sklearn = knn_sklearn.predict(X_test)

# Calculate the duration
end_time = time.time()
execution_time_sklearn = end_time - start_time

print(f"Sklearn Execution Time: {execution_time_sklearn} seconds")
