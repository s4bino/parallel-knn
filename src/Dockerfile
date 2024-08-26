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

# Copie o código fonte e o script de inicialização para o contêiner
COPY . .
COPY start.sh /usr/src/app/start.sh

# Dê permissão de execução para o script de inicialização
RUN chmod +x /usr/src/app/start.sh

# Defina o script de inicialização como ponto de entrada
ENTRYPOINT ["/usr/src/app/start.sh"]