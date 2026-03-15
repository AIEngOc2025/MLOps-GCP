# MLOps-GCP : API de Scoring de Crédit

Ce projet met en œuvre un pipeline d'opérations de machine learning (MLOps) pour un modèle de scoring de crédit. Le projet est construit avec FastAPI et est conçu pour être déployé sur Google Cloud Platform (GCP).

## Architecture

Le projet est structuré comme suit :

*   **`src/api.py`** : Une application FastAPI qui expose le modèle de scoring de crédit via une API RESTful.
*   **`models/`** : Ce répertoire contient le modèle de machine learning entraîné (`model.joblib`), un préprocesseur (`preprocessor.pkl`) et une liste de fonctionnalités sélectionnées (`selected_features.pkl`).
*   **`utilitaires/`** : Ce répertoire contient plusieurs scripts Jupyter pour des tâches telles que le prétraitement des données, l'entraînement du modèle et les tests.
*   **`Dockerfile`** : Un Dockerfile pour construire une image de conteneur de l'application FastAPI.
*   **`requirements.txt`** : Une liste de dépendances Python pour le projet.

## Points de Terminaison de l'API

L'application FastAPI dans `src/api.py` expose le point de terminaison suivant :

*   **`POST /predict`** : Ce point de terminaison accepte une charge utile JSON avec les données client et renvoie une prédiction de score de crédit.

## Comment Utiliser

1.  **Construire l'image Docker :**

    ```bash
    docker build -t credit-scoring-api .
    ```

2.  **Exécuter le conteneur Docker :**

    ```bash
    docker run -p 8000:8000 credit-scoring-api
    ```

3.  **Envoyer une requête à l'API :**

    Vous pouvez utiliser un outil comme `curl` ou `requests` pour envoyer une requête POST au point de terminaison `/predict` avec les données client. Voici un exemple avec `curl` :

    ```bash
    curl -X 'POST' \
      'https://mlops-gcp-service-610647369028.europe-west9.run.app/predict' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "EXT_SOURCE_1": 0.5,
      "EXT_SOURCE_3": 0.5,
      "EXT_SOURCE_2": 0.5,
      "AMT_CREDIT": 500000,
      "AMT_ANNUITY": 25000,
      "DAYS_EMPLOYED": -2000,
      "AMT_GOODS_PRICE": 500000,
      "DAYS_BIRTH": -15000,
      "DAYS_LAST_PHONE_CHANGE": -1000,
      "AMT_INCOME_TOTAL": 100000
    }'
    ```

## Scripts

Le répertoire `utilitaires/` contient les scripts suivants :

*   **`preprocessing_baseline_model.ipynb`** : Ce script contient le code pour le prétraitement des données et l'entraînement d'un modèle de base.
*   **`model_testing.ipynb`** : Ce script est utilisé pour tester le modèle entraîné.
*   **`fine_tuning.ipynb`** : Ce script contient le code pour l'optimisation du modèle.
*   **`preprocessingFiles.ipynb`**: Ce script contient du code pour le prétraitement des fichiers.
*   **`credit_scoring.ipynb`**: Un script avec des fonctionnalités de scoring de crédit.
