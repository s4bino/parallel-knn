from parallel_knn import KNearestNeighborsParallel
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time

print("Parallel")

# Load the Digits dataset
data = load_digits()
X = data.data
y = data.target

# Split the data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Measure the execution time
start_time = time.time()

# Initialize the parallel KNN model
knn = KNearestNeighborsParallel(k=3)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)

# Calculate the duration
end_time = time.time()
execution_time = end_time - start_time

print(f"Parallel Execution Time: {execution_time} seconds")
