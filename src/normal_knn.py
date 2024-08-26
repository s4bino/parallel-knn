import numpy as np

class KNearestNeighbors:
    def __init__(self, k=5):
        """
        Initialize the K-Nearest Neighbors classifier.

        Parameters:
        k (int): Number of neighbors to use for classification.
        """
        self.k = k

    def fit(self, X_train, y_train):
        """
        Fit the model using the training data.

        Parameters:
        X_train (np.ndarray): Feature matrix for training data.
        y_train (np.ndarray): Target vector for training data.
        """
        self.X_train = X_train
        self.y_train = y_train
        # Determine unique classes in the training data
        self.classes = np.unique(y_train)

    def predict(self, X):
        """
        Predict the class labels for the given data.

        Parameters:
        X (np.ndarray): Feature matrix for which predictions are to be made.

        Returns:
        list: Predicted class labels.
        """
        # Predict the class for each sample in X
        return [self._classified(self._kneighbors(x)) for x in X]
    
    def _euclidean_distance(self, X_train, x):
        """
        Compute the Euclidean distance between each training sample and a given sample.

        Parameters:
        X_train (np.ndarray): Feature matrix of training data.
        x (np.ndarray): Feature vector of the sample to compare.

        Returns:
        np.ndarray: Array of distances from the sample to each training point.
        """
        return np.sum((X_train - x) ** 2, axis=1)

    def _kneighbors(self, x):
        """
        Find the k-nearest neighbors of a given sample.

        Parameters:
        x (np.ndarray): Feature vector of the sample to find neighbors for.

        Returns:
        np.ndarray: Array of target values for the k-nearest neighbors.
        """
        # Calculate distances from the sample to all training points
        distances = self._euclidean_distance(self.X_train, x)
        # Get indices of the k-nearest neighbors
        nearest_indices = np.argpartition(distances, self.k)[:self.k]
        # Retrieve the target values of the k-nearest neighbors
        kneighbors = self.y_train[nearest_indices]
        return kneighbors
    
    def _classified(self, kneighbors):
        """
        Determine the most frequent class among the k-nearest neighbors.

        Parameters:
        kneighbors (np.ndarray): Array of target values for the k-nearest neighbors.

        Returns:
        int: Predicted class label based on the majority vote among neighbors.
        """
        # Find the class with the highest count
        classified = np.argmax(np.bincount(kneighbors))
        return classified
