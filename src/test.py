from parallel_knn import KNearestNeighbors
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import time

# Carregar o dataset Iris
data = load_iris()
X = data.data
y = data.target

# Dividir os dados em 70% treino e 30% teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

# Inicialize o modelo KNN com paralelização
knn = KNearestNeighbors(k=3)

# Medir o tempo de execução
start_time = time.time()

# Treinar o modelo
knn.fit(X_train, y_train)

# Fazer previsões
predictions = knn.predict(X_test)

# Calcular o tempo de duração
end_time = time.time()
execution_time = end_time - start_time

# Somente o rank 0 terá previsões
if predictions is not None:
    print("Predicted labels:", predictions)

print(f"Tempo de execução: {execution_time} segundos")

