# Diabetes MLflow Experiments

Experimentos de predição de diabetes integrados com o MLflow para rastreamento de experimentos, registro de métricas e salvamento de modelos.

## Como Executar

Este projeto utiliza o [uv](https://github.com/astral-sh/uv) para gerenciamento rápido e eficiente de dependências.

### 1. Executar os Experimentos
Para treinar os modelos e registrar os dados no MLflow, execute:
```bash
uv run python diabetes_mlflow_experiments.py
```

### 2. Iniciar o Servidor do MLflow
Para visualizar e comparar os experimentos na interface web do MLflow, inicie o servidor:
```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abra [http://127.0.0.1:5000](http://127.0.0.1:5000) no seu navegador.
