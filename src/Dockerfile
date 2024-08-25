# Use uma imagem base oficial do Python
FROM python:3.9-slim

# Instale dependências necessárias para MPI
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenmpi-dev \
    openmpi-bin \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Crie um diretório de trabalho
WORKDIR /usr/src/app

# Caso não tenha o requirements.txt e queira instalar mpi4py diretamente:
RUN pip install --no-cache-dir mpi4py numpy scikit-learn

# Copie o código fonte para o contêiner
COPY . .

# Comando para rodar o script com MPI
CMD ["mpirun", "--allow-run-as-root", "-n", "2", "python", "test.py"]


# Isso executará seu script com 4 processos MPI.

