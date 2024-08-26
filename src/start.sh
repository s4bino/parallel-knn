#!/bin/bash

# Execute o primeiro comando em background
python normal_test.py

# Execute o primeiro comando em background
python sklearn_test.py

# Execute o segundo comando
mpirun --allow-run-as-root -n 2 python parallel_test.py

# Aguarde até que todos os processos em background sejam concluídos
wait