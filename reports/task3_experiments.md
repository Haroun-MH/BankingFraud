# Projet MLA - Tache 3: Experimentation et comparaison des algorithmes

## Contexte
Classification fraude (dataset IEEE-CIS). Suivi des experiments avec MLflow.

## Algorithmes testes
- KNN
- SVM (LinearSVC)
- Random Forest
- Logistic Regression

## Metriques
Metrique principale: F1-score. Metriques secondaires: precision, recall, ROC-AUC, accuracy.

## Tableau comparatif (meilleure config par modele)
| model | params_json | val_f1 | val_precision | val_recall | val_roc_auc | val_accuracy | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForest | {"class_weight": "balanced_subsample", "max_depth": null, "n_estimators": 200, "n_jobs": -1, "random_state": 42} | 0.5768 | 0.9480 | 0.4145 | 0.9395 | 0.9787 | 931e906853464bf7bbc35c6e4bc85968 |
| SVM_Linear | {"C": 1.0, "class_weight": "balanced", "dual": false, "max_iter": 3000} | 0.2087 | 0.1222 | 0.7147 | 0.8441 | 0.8104 | d28dd47011994d3489863c1abf1c73d8 |
| KNN | {"n_neighbors": 3, "weights": "distance"} | 0.1279 | 0.2964 | 0.0815 | 0.6075 | 0.9611 | 030db68153fd4b7ca976a29c3bad58a6 |
| LogisticRegression | {"C": 0.5, "class_weight": "balanced", "max_iter": 300, "n_jobs": -1, "solver": "saga"} | 0.0981 | 0.0521 | 0.8534 | 0.7489 | 0.4511 | dd3e673c85d0425ca2854f97a67f86cc |

## Meilleur modele (validation)
- Modele: RandomForest
- Params: {"class_weight": "balanced_subsample", "max_depth": null, "n_estimators": 200, "n_jobs": -1, "random_state": 42}
- F1 validation: 0.5768
- Run ID MLflow: 931e906853464bf7bbc35c6e4bc85968

## Notes
- Les runs MLflow sont stockes dans mlruns/ avec les artefacts de modeles.
- Les resultats detailles sont dans reports/task3_comparison.csv.
