from mpi4py import MPI  # Import MPI for parallelism
import numpy as np

class KNearestNeighbors:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.classes_ = np.unique(y_train)

    def predict(self, X):
        # Get the MPI communicator and the rank/size
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Split the data into chunks, one for each process
        chunk_size = len(X) // size
        start = rank * chunk_size
        if rank == size - 1:  # Last process gets the remainder
            end = len(X)
        else:
            end = start + chunk_size

        # Each process computes predictions for its chunk
        local_X = X[start:end]
        local_predictions = []
        for x in local_X:
            kneighbors = self._kneighbors(x)
            classified = self._classified(kneighbors)
            local_predictions.append(classified)

        # Gather all predictions from all processes
        all_predictions = comm.gather(local_predictions, root=0)

        # The root process (rank 0) concatenates all the results
        if rank == 0:
            return [pred for sublist in all_predictions for pred in sublist]
        else:
            return None

    def _euclidean_distance(self, p, q):
        return np.linalg.norm(p - q, axis=1)

    def _kneighbors(self, x):
        distances = self._euclidean_distance(self.X_train, x)
        nearest_indices = np.argsort(distances)
        kneighbors = self.y_train[nearest_indices[:self.k]]
        return kneighbors

    def _classified(self, kneighbors):
        classified = np.argmax(np.bincount(kneighbors.astype(int)))
        return classified
