import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# Configuração inicial do MLflow em ambiente local
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("aula_01_diabetes_experimento")

# Carregamento dos dados
dados = pd.read_csv("./data/diabetes.csv")

feature_cols = ["pregnant", "insulin", "bmi", "age", "glucose", "bp", "pedigree"]

X = dados[feature_cols]
y = dados["label"]

# Separação treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=1,
    stratify=y,
)

# Configurações diferentes para gerar múltiplos experimentos
experimentos = [
    {
        "run_name": "svc_linear_c_1",
        "kernel": "linear",
        "C": 1.0,
        "gamma": "scale",
    },
    {
        "run_name": "svc_rbf_c_1",
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
    },
    {
        "run_name": "svc_rbf_c_2",
        "kernel": "rbf",
        "C": 2.0,
        "gamma": "scale",
    },
    {
        "run_name": "svc_rbf_c_5",
        "kernel": "rbf",
        "C": 5.0,
        "gamma": "scale",
    },
]

melhor_f1 = 0
melhor_run = None

for config in experimentos:
    with mlflow.start_run(run_name=config["run_name"]) as run:
        # Pipeline com normalização + modelo
        modelo = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel=config["kernel"],
                        C=config["C"],
                        gamma=config["gamma"],
                        probability=True,
                        random_state=1,
                    ),
                ),
            ]
        )

        # Treinamento
        modelo.fit(X_train, y_train)

        # Predição
        y_pred = modelo.predict(X_test)

        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Registro dos parâmetros no MLflow
        mlflow.log_param("model", "SVC")
        mlflow.log_param("kernel", config["kernel"])
        mlflow.log_param("C", config["C"])
        mlflow.log_param("gamma", config["gamma"])
        mlflow.log_param("test_size", 0.30)
        mlflow.log_param("random_state", 1)
        mlflow.log_param("features", ",".join(feature_cols))

        # Registro das métricas no MLflow
        mlflow.log_metric("accuracy_score", accuracy)
        mlflow.log_metric("precision_score", precision)
        mlflow.log_metric("recall_score", recall)
        mlflow.log_metric("f1_score", f1)

        # Artefato: relatório de classificação
        report = classification_report(y_test, y_pred, zero_division=0)

        with open("classification_report.txt", "w", encoding="utf-8") as file:
            file.write(report)

        mlflow.log_artifact("classification_report.txt", artifact_path="reports")

        # Artefato: matriz de confusão
        matriz = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=matriz)
        disp.plot()
        plt.title(f"Matriz de Confusão - {config['run_name']}")
        plt.savefig("confusion_matrix.png")
        plt.close()

        mlflow.log_artifact("confusion_matrix.png", artifact_path="plots")

        # Registro do modelo treinado
        signature = infer_signature(X_test, modelo.predict(X_test))

        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="model",
            signature=signature,
            input_example=X_test.head(5),
        )

        print("Run:", config["run_name"])
        print("Run ID:", run.info.run_id)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("-" * 50)

        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_run = run.info.run_id

print("Melhor execução considerando F1 Score:")
print("Run ID:", melhor_run)
print("F1 Score:", melhor_f1)