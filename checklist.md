# Projet MLA — Checklist des Tâches

---

## ✅ Tâche 3 — Expérimentation et comparaison des algorithmes ML avec MLflow

### 1. Algorithmes à implémenter (Classification — Détection de fraude)

- [x] k-Nearest Neighbors (KNN)
- [x] Support Vector Machine (SVM)
- [x] Random Forest
- [x] Logistic Regression
- [x] Tester différentes valeurs d'hyperparamètres
- [x] Comparer les performances des modèles (Accuracy, Precision, Recall, F1-score)

### 2. Réduction de dimension (recommandée)
- [x] Principal Component Analysis (PCA / ACP)
- [x] t-distributed Stochastic Neighbor Embedding (t-SNE)

### 3. Utilisation de MLflow
- [x] Enregistrer toutes les expérimentations avec MLflow
- [x] Logger les paramètres du modèle
- [x] Logger les métriques de performance
- [x] Logger la version du modèle entraîné
- [x] Réaliser plusieurs runs MLflow pour comparer les modèles

### 4. Livrables

#### Code source
- [x] Scripts Python implémentant les différents modèles
- [x] Dépôt Git public avec le code complet
- [x] Intégration de MLflow dans le code

#### Analyse expérimentale
- [x] Document décrivant les algorithmes testés
- [x] Document décrivant les paramètres utilisés
- [x] Document décrivant les résultats obtenus

#### Comparaison des modèles
- [x] Tableau comparatif des performances (Modèle | Paramètres | Métrique principale)

#### Analyse critique
- [x] Répondre : Quel algorithme donne les meilleurs résultats ?
- [x] Répondre : Quels paramètres influencent le plus les performances ?
- [x] Répondre : La réduction de dimension améliore-t-elle les résultats ?

### 5. Structure du projet (React + MLflow)
- [x] `backend/app.py` — serveur Flask ou FastAPI
- [x] `backend/train.py` — entraînement + MLflow
- [x] `backend/evaluate.py` — évaluation des modèles
- [x] `backend/data_loader.py`
- [x] `backend/preprocessing.py`
- [x] `backend/requirements.txt`
- [x] `frontend/package.json`
- [x] `frontend/src/App.js`
- [x] `frontend/src/components/PredictForm.js`
- [x] `frontend/src/services/api.js`
- [x] `data/raw/` — données brutes
- [x] `data/processed/` — données nettoyées
- [x] `models/` — modèles sauvegardés
- [x] `notebooks/` — exploration / EDA
- [x] `mlruns/` — runs MLflow
- [x] `requirements.txt`
- [x] `README.md`

### 6. Bonnes pratiques
- [x] Backend : Python + MLflow pour entraînement et API
- [x] Frontend : React pour l'IHM (formulaire, visualisation)
- [x] Communication via REST API entre React et Python
- [x] Versionnement Git pour le code complet
- [x] MLflow : tracker toutes les expérimentations et sauvegarder le meilleur modèle

### 7. Critères d'évaluation (Tâche 3)
| Critère | Pondération |
|---|---|
| Implémentation correcte des algorithmes | 30% |
| Utilisation de MLflow pour le suivi des expériences | 20% |
| Analyse et comparaison des résultats | 30% |
| Qualité du code et de la documentation | 20% |

---

## ✅ Tâche 4 — Interprétation et analyse Random Forest avec MLflow

> ⚠️ Toutes les expérimentations doivent être suivies avec MLflow.

- [x] **Feature Importance**
  - [x] Extraire `feature_importances_` de l'objet Random Forest entraîné
  - [x] Visualiser les importances à l'aide d'un graphique
  - [x] Identifier les 3 variables les plus importantes
  - [x] Analyser si cela correspond à la compréhension des données

- [x] **Stabilité des prédictions**
  - [x] Tester avec différents `random_state`
  - [x] Observer et documenter la variabilité des résultats
  - [x] Conclure sur la robustesse du modèle

- [x] **Analyse des erreurs**
  - [x] Identifier 2-3 exemples mal classés
  - [x] Analyser pourquoi le modèle a échoué sur ces exemples
  - [x] Identifier des patterns dans les erreurs

- [x] **Biais et Variance**
  - [x] Tester différentes valeurs de `n_estimators`
  - [x] Tester différentes valeurs de `max_depth`
  - [x] Remplir le tableau : `n_estimators | max_depth | Train Accuracy | Test Accuracy | Biais | Variance`
  - [x] Identifier le paramétrage montrant de l'overfitting
  - [x] Identifier le paramétrage montrant de l'underfitting
  - [x] Identifier le paramétrage équilibré

- [x] **Comparaison**
  - [x] Comparer les résultats Random Forest vs Arbre de décision

---

### Compte Rendu (Tâche 4)
- [x] Rédiger un document avec des réponses claires et justifiées pour chaque question

## ✅ Tâche 5 — Pipeline MLOps Local

> ⚠️ Vérifier et compléter uniquement les parties non encore implémentées dans votre projet.

---

### Stack technologique requise
| Outil | Rôle | Version minimale |
|---|---|---|
| MLflow | Tracking, Registry, Serving | >= 2.10 |
| Python | Langage principal | >= 3.9 |
| scikit-learn | Modèles ML | >= 1.3 |
| pandas / numpy | Manipulation de données | latest |
| Git + DVC | Versioning code & données | DVC >= 3.x |
| Docker | Conteneurisation (optionnel) | latest |

- [x] Vérifier que les versions installées respectent les minimums ci-dessus

---

### Partie 1 — Tracking des Expérimentations

#### 1.1 Initialisation du projet MLflow
- [x] Créer la structure de répertoires (`data/`, `src/train.py`, `src/preprocess.py`, `mlruns/`, `requirements.txt`)
- [x] Ajouter `import mlflow` et `mlflow.set_experiment(...)` dans `train.py`
- [x] Initialiser un run avec `mlflow.start_run(run_name='baseline_v1')`

#### 1.2 Logging des paramètres et métriques
- [x] Logger les hyperparamètres avec `mlflow.log_params(params)` avant l'entraînement
- [x] Logger les métriques avec `mlflow.log_metrics(metrics)` après évaluation (accuracy, f1_score, roc_auc)
- [x] Logger le modèle comme artefact avec `mlflow.sklearn.log_model(...)`

#### 1.3 Logger des artefacts supplémentaires
- [x] Générer et logger la matrice de confusion (`confusion_matrix.png`)
- [x] Générer et logger le rapport de classification (`classification_report.txt`)

#### Questions de réflexion (Partie 1)
- [x] Q1 : Quelle est la différence entre `mlflow.log_param()` et `mlflow.log_metric()` ?
- [x] Q2 : Pourquoi est-il important de nommer ses runs (`run_name`) ?
- [x] Q3 : Que se passe-t-il si on exécute deux fois le même script sans changer le `run_name` ?

---

### Partie 2 — Comparaison d'Expérimentations

#### 2.1 Entraînement de plusieurs modèles
- [x] Run 1 — Baseline : `RandomForestClassifier(n_estimators=50, max_depth=3)`
- [x] Run 2 — Modèle plus profond : `RandomForestClassifier(n_estimators=200, max_depth=10)`
- [x] Run 3 — Algorithme alternatif : `GradientBoostingClassifier` ou `LogisticRegression`
- [x] Run 4 (optionnel) : XGBoost ou SVM avec GridSearchCV
- [x] Automatiser les runs avec une boucle de configuration

#### 2.2 Exploration de MLflow UI
- [x] Lancer l'interface MLflow (`mlflow ui --host 0.0.0.0 --port 5000`)
- [x] Capturer : trier les runs par accuracy décroissante
- [x] Capturer : sélectionner 2 runs et cliquer sur "Compare"
- [x] Capturer : visualiser le graphique Parallel Coordinates
- [x] Capturer : télécharger la matrice de confusion depuis l'onglet Artifacts

#### 2.3 Requêtes programmatiques via le client MLflow
- [x] Identifier le meilleur run via `MlflowClient` et `client.search_runs(...)`
- [x] Afficher le `run_id`, l'accuracy et les paramètres du meilleur run

#### Questions de réflexion (Partie 2)
- [x] Q4 : Quel modèle obtient le meilleur compromis accuracy / f1_score ? Justifiez.
- [x] Q5 : Le graphique Parallel Coordinates révèle-t-il une corrélation entre `max_depth` et accuracy ?
- [x] Q6 : Comment MLflow permet-il la reproductibilité par rapport à un simple `print()` des métriques ?

---

### Partie 3 — Model Registry

#### 3.1 Enregistrement du meilleur modèle
- [x] Enregistrer le modèle du meilleur run avec `mlflow.register_model(...)`
- [x] Ajouter une description au modèle avec `client.update_registered_model(...)`
- [x] Ajouter des tags avec `client.set_model_version_tag(...)`

#### 3.2 Gestion du cycle de vie
- [x] Promouvoir le modèle en **Staging** avec `client.transition_model_version_stage(..., stage='Staging')`
- [x] Vérifier que l'accuracy dépasse le seuil (ex: 0.85)
- [x] Promouvoir le modèle en **Production** si le seuil est atteint

#### Questions de réflexion (Partie 3)
- [x] Q7 : Pourquoi séparer les étapes Staging et Production dans un registre de modèles ?
- [x] Q8 : Que se passe-t-il si l'on archive une version en Production ? Quel impact opérationnel ?
- [x] Q9 : Comment le Registry facilite-t-il le rollback vers une version précédente ?

---

### Partie 4 — Serving et API REST

#### 4.1 Serving natif MLflow
- [x] Servir le modèle en Production : `mlflow models serve -m 'models:/mon_modele_production/Production' --port 1234 --no-conda`
- [x] Tester l'endpoint avec `curl` (POST `/invocations`)
- [x] Tester l'endpoint depuis Python avec `requests.post(...)`

#### 4.2 Serving custom avec FastAPI (optionnel avancé)
- [x] Créer une API FastAPI avec `@app.post('/predict')`
- [x] Charger le modèle depuis le Registry avec `mlflow.sklearn.load_model(...)`
- [x] Lancer l'API avec `uvicorn app:app --host 0.0.0.0 --port 8000 --reload`

#### Questions de réflexion (Partie 4)
- [x] Q10 : Quel est l'avantage d'un serving MLflow natif vs FastAPI personnalisé ?
- [x] Q11 : Comment géreriez-vous le rechargement automatique d'un nouveau modèle en Production ?
- [x] Q12 : Quels headers HTTP ajouteriez-vous pour sécuriser l'endpoint en production réelle ?

---

### Partie 5 — Automatisation CI/CD Local (Bonus)

#### 5.1 Makefile comme orchestrateur
- [x] Créer un `Makefile` avec les cibles : `setup`, `train`, `register`, `serve`, `test`, `pipeline`
- [x] Vérifier que `make pipeline` enchaîne toutes les étapes

#### 5.2 Pre-commit hook Git
- [x] Créer le fichier `.git/hooks/pre-commit` (avec `chmod +x`)
- [x] Implémenter la validation automatique de l'accuracy avant chaque commit (seuil > 0.80)
- [x] Vérifier que le commit est refusé si le modèle est invalide

---

### Partie 6 — Détection du Data Drift

#### 6.1 Concepts — Data Drift vs Concept Drift
- [x] Comprendre et documenter les 4 types : Data Drift, Concept Drift, Label Drift, Covariate Shift
- [x] Donner un exemple concret avec vos propres données pour chaque type

#### 6.2 Simulation du drift sur vos données
- [x] Installer Evidently : `pip install evidently`
- [x] Créer le script `src/simulate_drift.py`
- [x] Simuler le drift : décalage de moyenne + bruit sur 2 features numériques
- [x] Vérifier la différence de moyenne entre les données de référence et de production

#### 6.3 Génération du rapport Evidently et logging MLflow
- [x] Créer une expérience MLflow `monitoring_drift`
- [x] Générer le rapport HTML visuel avec `DataDriftPreset` et `DataQualityPreset`
- [x] Logger le rapport HTML comme artefact MLflow
- [x] Extraire et logger les métriques numériques : `drift_share`, `drifted_columns`, `total_columns`, `dataset_drifted`

#### 6.4 Test statistique KS-test par feature
- [x] Appliquer `scipy.stats.ks_2samp` sur chaque feature numérique
- [x] Logger le p-value par feature dans MLflow (`ks_pvalue_{col}`)
- [x] Sauvegarder les résultats dans `ks_drift_results.csv` et le logger comme artefact

#### 6.5 Déclenchement automatique du ré-entraînement
- [x] Définir `SEUIL_DRIFT = 0.30` (ré-entraînement) et `SEUIL_WARN = 0.15` (alerte)
- [x] Implémenter la logique conditionnelle : CRITIQUE / AVERTISSEMENT / OK
- [x] Déclencher automatiquement `src/train.py --retrain` si drift > seuil
- [x] Logger `retrain_triggered` dans MLflow

#### 6.6 Pipeline MLOps complet
- [x] Représenter le pipeline complet en boucle fermée dans le rapport (Données → Prétraitement → Entraînement → Registry → Serving → Monitoring → Ré-entraînement)

#### Questions de réflexion (Partie 6)
- [x] Q13 : Quelle est la différence entre data drift et concept drift ? Donnez un exemple concret avec vos données.
- [x] Q14 : Le KS-test et Evidently identifient-ils les mêmes features comme driftées ? Pourquoi ?
- [x] Q15 : Quel seuil de drift choisiriez-vous pour votre projet ? Justifiez selon le domaine métier.
- [x] Q16 : Sans pipeline MLOps automatisé, comment détecteriez-vous ce drift en pratique ? Quels sont les risques ?

---

### Compte Rendu (Tâche 5)
- [x] Rédiger un document avec les réponses aux 16 questions (Q1 à Q16)
- [x] Inclure les 4 captures d'écran MLflow UI requises :
  - [x] Capture : runs triés par accuracy décroissante
  - [x] Capture : comparaison de 2 runs (bouton "Compare")
  - [x] Capture : graphique Parallel Coordinates (hyperparamètres vs métriques)
  - [x] Capture : téléchargement de la matrice de confusion depuis l'onglet Artifacts
- [x] Documenter les choix techniques (modèle retenu, seuils de drift, seuil de production, etc.)
- [x] Committer après chaque partie fonctionnelle avec des noms de runs explicites (ex: `rf_100trees_depth5`)

### Conseils (depuis le PDF)
- Nommer les runs de façon explicite, ex: `rf_100trees_depth5`, pour faciliter la comparaison dans MLflow UI
- Documenter dans le rapport *pourquoi* ce modèle et *pourquoi* ce seuil de validation
- Consulter la documentation officielle en cas de blocage : https://mlflow.org/docs/latest/

### Grille d'évaluation (Tâche 5)
| Critère | Points | Détail |
|---|---|---|
| Tracking MLflow complet | 4 pts | Params, métriques, artefacts loggés |
| MLflow UI & comparaison | 2 pts | Captures + analyse des runs |
| Model Registry | 3 pts | Enregistrement + cycle Staging/Production |
| Serving & test API | 3 pts | Endpoint fonctionnel + test réussi |
| Détection drift (Evidently) | 3 pts | Rapport HTML + métriques MLflow |
| KS-test par feature | 2 pts | Analyse statistique + CSV loggé |
| Logique ré-entraînement | 2 pts | Seuils définis + déclenchement conditionnel |
| Réflexion & analyse | 4 pts | Réponses aux 16 questions |
| Bonus CI/CD local | 2 pts | Makefile ou hook Git opérationnel |
