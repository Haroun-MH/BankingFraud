# Projet MLA - Tache 4: Interpretation Random Forest

## Feature importance
Top 3 features: TransactionAmt, TransactionDT, card1

Le graphique est disponible: reports/rf_feature_importance.png

## Stabilite des predictions
Resume F1 validation: mean=0.5769, std=0.0021

Tableau des runs (random_state):
| random_state | val_f1 | val_precision | val_recall | val_roc_auc |
| --- | --- | --- | --- | --- |
| 1.0000 | 0.5794 | 0.9509 | 0.4166 | 0.9393 |
| 7.0000 | 0.5738 | 0.9471 | 0.4116 | 0.9398 |
| 21.0000 | 0.5764 | 0.9526 | 0.4133 | 0.9399 |
| 42.0000 | 0.5768 | 0.9480 | 0.4145 | 0.9395 |
| 99.0000 | 0.5780 | 0.9481 | 0.4157 | 0.9395 |

## Analyse des erreurs
Exemples exportes: reports/rf_misclassified_samples.csv

## Biais et variance (accuracy)
Tableau:
| n_estimators | max_depth | train_accuracy | val_accuracy | bias | variance |
| --- | --- | --- | --- | --- | --- |
| 100 | 5.0000 | 0.8010 | 0.8041 | High | Low |
| 100 | 10.0000 | 0.8734 | 0.8722 | High | Low |
| 100 | 20.0000 | 0.9758 | 0.9636 | Medium | Medium |
| 100 | nan | 1.0000 | 0.9787 | Medium | Medium |
| 300 | 5.0000 | 0.8015 | 0.8045 | High | Low |
| 300 | 10.0000 | 0.8742 | 0.8736 | High | Low |
| 300 | 20.0000 | 0.9763 | 0.9639 | Medium | Medium |
| 300 | nan | 1.0000 | 0.9787 | Medium | Medium |
| 500 | 5.0000 | 0.8018 | 0.8050 | High | Low |
| 500 | 10.0000 | 0.8742 | 0.8736 | High | Low |
| 500 | 20.0000 | 0.9766 | 0.9643 | Medium | Medium |
| 500 | nan | 1.0000 | 0.9787 | Medium | Medium |

## Comparaison Random Forest vs Decision Tree
- RF (best): F1=0.5768, Precision=0.9480, Recall=0.4145
- Decision Tree: F1=0.5342, Precision=0.5307, Recall=0.5379

## Notes
- Les runs MLflow incluent les artefacts (modeles, confusion matrices) dans mlruns/.
- Les fichiers complementaires sont dans reports/.
