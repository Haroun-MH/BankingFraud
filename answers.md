# Tâche 5 — Réponses aux Questions de Réflexion (Q1–Q16) & Concepts de Drift

---

## Partie 1 — Tracking des Expérimentations

### Q1 : Quelle est la différence entre `mlflow.log_param()` et `mlflow.log_metric()` ?

`mlflow.log_param()` enregistre une valeur de configuration fixe (hyperparamètre) qui ne change pas pendant l'entraînement — par exemple `n_estimators=200` ou `C=1.0`. Ces valeurs sont indexées pour la recherche et la comparaison dans l'UI MLflow.

`mlflow.log_metric()` enregistre une valeur numérique mesurée après évaluation — par exemple `val_f1=0.878` ou `val_accuracy=0.968`. Les métriques peuvent être loggées à plusieurs étapes (step) pour tracer des courbes d'apprentissage. MLflow les stocke séparément et permet de les trier, filtrer et visualiser.

**Règle pratique** : ce qu'on choisit avant l'entraînement → `log_param` ; ce qu'on mesure après → `log_metric`.

---

### Q2 : Pourquoi est-il important de nommer ses runs (`run_name`) ?

Un `run_name` explicite (ex : `RF_200trees_depth20`) permet de :

1. Identifier immédiatement le contenu d'un run dans l'UI MLflow sans avoir à ouvrir les détails.
2. Filtrer et trier les runs par nom dans les requêtes programmatiques (`client.search_runs`).
3. Faciliter la communication en équipe : « le run `rf_100trees_depth5` est notre baseline » est plus clair qu'un UUID.
4. Assurer la traçabilité dans les pipelines CI/CD où les runs sont créés automatiquement.

Dans ce projet, les runs sont nommés de façon systématique : `KNN_k5`, `LinearSVC_C1.0`, `RF_200_depth20`, `RF_Stability_42`, etc.

---

### Q3 : Que se passe-t-il si on exécute deux fois le même script sans changer le `run_name` ?

MLflow crée deux runs distincts avec le même nom mais des `run_id` différents. Contrairement à certains outils, MLflow ne déduplique pas les runs par nom — le `run_name` est une étiquette lisible, pas une clé unique.

**Conséquences :**

- L'UI affiche deux lignes identiques, ce qui peut prêter à confusion.
- Les deux runs coexistent dans le tracking store et consomment de l'espace.
- Si on cherche « le meilleur run nommé RF_Best », on obtient plusieurs résultats et il faut trier par métrique pour trouver le bon.

**Bonne pratique** : inclure un timestamp ou un identifiant unique dans le `run_name`, ou utiliser `mlflow.set_tag('version', 'v2')` pour distinguer les itérations.

---

## Partie 2 — Comparaison d'Expérimentations

### Q4 : Quel modèle obtient le meilleur compromis accuracy / f1_score ? Justifiez.

**Random Forest** obtient le meilleur compromis. Sur le jeu de validation :

- Accuracy élevée (~0.97) : peu d'erreurs globales.
- F1-score élevé (~0.87) : bon équilibre précision/rappel malgré le déséquilibre de classes (fraude ~3.5%).

**Justification :**

- L'accuracy seule est trompeuse sur des données déséquilibrées : un modèle qui prédit toujours « non-fraude » atteint ~96.5% d'accuracy mais F1=0.
- Le F1-score pénalise les faux négatifs (fraudes manquées) et les faux positifs (transactions légitimes bloquées), ce qui est critique en détection de fraude.
- Random Forest gère nativement le déséquilibre via `class_weight='balanced_subsample'` et capture les interactions non-linéaires entre features sans scaling.

---

### Q5 : Le graphique Parallel Coordinates révèle-t-il une corrélation entre `max_depth` et accuracy ?

Oui. Dans le graphique Parallel Coordinates de l'UI MLflow, on observe que :

- Les runs avec `max_depth=None` (arbres non élagués) tendent à avoir une `train_accuracy` proche de 1.0 mais une `val_accuracy` légèrement inférieure → signe d'overfitting léger.
- Les runs avec `max_depth=5` montrent une `train_accuracy` et `val_accuracy` toutes deux basses → underfitting (biais élevé).
- `max_depth=20` offre le meilleur compromis : `val_accuracy` élevée sans écart important avec `train_accuracy`.

La corrélation est positive jusqu'à `max_depth ≈ 20`, puis le gain marginal diminue et le risque d'overfitting augmente. Ce pattern est cohérent avec la théorie biais-variance.

---

### Q6 : Comment MLflow permet-il la reproductibilité par rapport à un simple `print()` des métriques ?

Un `print()` est éphémère : la valeur disparaît dès que le terminal est fermé, n'est pas associée aux paramètres qui l'ont produite, et ne peut pas être comparée automatiquement avec d'autres runs.

MLflow garantit la reproductibilité en :

1. **Persistant** chaque run avec son `run_id` unique, ses paramètres, métriques, artefacts (modèle, rapport de classification, matrice de confusion) et son timestamp dans un tracking store versionné.
2. **Permettant de recharger** exactement le modèle d'un run passé via `mlflow.sklearn.load_model('runs:/{run_id}/model')`.
3. **Enregistrant l'environnement** (versions des librairies) via `conda.yaml` / `requirements.txt` dans chaque artefact.
4. **Offrant une UI** de comparaison et des requêtes programmatiques (`MlflowClient.search_runs`) pour retrouver le meilleur run selon n'importe quelle métrique.

---

## Partie 3 — Model Registry

### Q7 : Pourquoi séparer les étapes Staging et Production dans un registre de modèles ?

La séparation Staging / Production implémente un processus de validation en deux étapes qui réduit le risque de déployer un modèle défaillant :

**Staging :**

- Environnement de pré-production où le modèle est testé sur des données récentes ou un sous-ensemble de production.
- Permet de vérifier les performances (accuracy ≥ 0.85 dans ce projet), la latence, et la compatibilité avec le pipeline de serving.
- Les équipes QA et métier peuvent valider le modèle sans impact client.

**Production :**

- Le modèle est servi aux utilisateurs finaux.
- La promotion est conditionnelle (seuil d'accuracy vérifié dans `registry_service.py`).
- Une seule version est en Production à la fois (`archive_existing_versions=True`).

Sans cette séparation, un modèle entraîné sur des données driftées pourrait être déployé directement, causant des pertes financières ou des faux positifs massifs en production.

---

### Q8 : Que se passe-t-il si l'on archive une version en Production ? Quel impact opérationnel ?

Archiver une version Production dans MLflow la fait passer au stage `Archived`. Elle n'est plus retournée par `get_latest_versions(stages=['Production'])`.

**Impact opérationnel :**

- Si aucune autre version n'est promue en Production, l'endpoint `models:/FraudDetectionModel/Production` ne résout plus vers aucun modèle.
- Le serving natif MLflow (`mlflow models serve`) échoue avec une erreur « No model version found in stage Production ».
- Dans ce projet, `_load_predict_model()` tombe en fallback sur le `run_id` stocké dans `reports/current_model.json`, ce qui maintient le service.

**Bonne pratique** : toujours promouvoir une nouvelle version avant d'archiver l'ancienne, ou utiliser `archive_existing_versions=True` lors de la promotion (ce que fait `registry_service.promote_model`).

---

### Q9 : Comment le Registry facilite-t-il le rollback vers une version précédente ?

Le Registry conserve toutes les versions enregistrées avec leur `run_id`, leurs métriques et leur historique de stages. Un rollback se fait en deux appels API :

```python
client.transition_model_version_stage(
    name='FraudDetectionModel',
    version='2',  # version précédente
    stage='Production',
    archive_existing_versions=True
)
```

Ou via l'UI MLflow : onglet Models → sélectionner la version → « Transition to Production ».

Dans ce projet, le rollback est aussi possible via l'endpoint `POST /api/model/active` avec le `run_id` de l'ancienne version, ce qui recharge immédiatement l'ancien modèle pour les prédictions sans redémarrer le serveur (cache invalidé par `run_id` différent).

---

## Partie 4 — Serving et API REST

### Q10 : Quel est l'avantage d'un serving MLflow natif vs FastAPI personnalisé ?

**Serving natif MLflow** (`mlflow models serve`) :

- ✅ Zéro code : une commande suffit pour exposer n'importe quel modèle enregistré.
- ✅ Format d'entrée standardisé (`dataframe_records`, `dataframe_split`, `tensor`).
- ✅ Intégration directe avec le Registry : `models:/MonModele/Production`.
- ✅ Gestion automatique de l'environnement (conda/pip).
- ❌ Peu flexible : pas de logique métier, pas de preprocessing custom.
- ❌ Format de réponse fixe (liste de prédictions brutes).

**FastAPI personnalisé** (`backend/app.py`) :

- ✅ Preprocessing intégré (`preprocess_single`) avant l'inférence.
- ✅ Réponse enrichie : `fraud_probability`, `is_fraud`, messages d'erreur clairs.
- ✅ Endpoints additionnels : training, registry, drift, dataset management.
- ✅ Cache du modèle en mémoire pour éviter les rechargements.
- ❌ Nécessite de maintenir le code de serving.

**Conclusion** : MLflow natif est idéal pour des tests rapides et des pipelines batch ; FastAPI est préférable pour une API de production avec logique métier.

---

### Q11 : Comment géreriez-vous le rechargement automatique d'un nouveau modèle en Production ?

Plusieurs approches selon le contexte :

1. **Cache invalidation** (implémenté dans ce projet) : `_predict_cache` compare le `run_id` actif à chaque requête. Quand `POST /api/model/active` est appelé avec un nouveau `run_id`, le cache est invalidé et le modèle est rechargé au prochain appel `/api/predict`.

2. **Polling périodique** : un thread background vérifie toutes les N minutes si une nouvelle version Production existe dans le Registry et recharge si nécessaire.

3. **Webhook MLflow** : MLflow >= 2.x supporte des webhooks sur les transitions de stage. Un webhook `POST /api/model/reload` peut déclencher le rechargement.

4. **Blue/Green deployment** : deux instances du serveur tournent en parallèle. Le load balancer bascule vers la nouvelle instance après validation, sans downtime.

---

### Q12 : Quels headers HTTP ajouteriez-vous pour sécuriser l'endpoint en production réelle ?

**Authentification :**

```
Authorization: Bearer <JWT_TOKEN>
X-API-Key: <API_KEY>
```

**Protection contre les attaques web :**

```
Strict-Transport-Security: max-age=31536000; includeSubDomains
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Content-Security-Policy: default-src 'none'
```

**Rate limiting** (via middleware FastAPI ou reverse proxy) :

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
```

**Traçabilité :**

```
X-Request-ID: <uuid>  (corrélation des logs)
```

Dans FastAPI, ces headers s'ajoutent via un middleware (`TrustedHostMiddleware`, `SecurityHeadersMiddleware`) ou via un reverse proxy (nginx, Traefik) en production.

---

## Partie 6 — Détection du Data Drift

### Concepts de Drift

#### 1. Data Drift (Feature Drift / Input Drift)

**Définition :** Le Data Drift se produit lorsque la distribution statistique d'une ou plusieurs features d'entrée change entre la période d'entraînement et la production. La relation entre les features et le label cible reste la même, mais les valeurs des features se décalent.

**Exemple concret (IEEE-CIS) :** `TransactionAmt` a une certaine distribution à l'entraînement (moyenne ~150$, asymétrie à droite). Si un nouveau processeur de paiement commence à router des transactions de plus grande valeur via le système, la moyenne pourrait passer à ~300$. Le modèle n'a jamais été entraîné sur cette plage, donc ses estimations de probabilité de fraude deviennent peu fiables — même si les patterns de fraude sous-jacents (montant élevé = risque plus élevé) n'ont pas changé.

**Détection :** Evidently `DataDriftPreset`, KS-test (`scipy.stats.ks_2samp`)

**Simulé dans :** `simulate_drift.py` — `TransactionAmt` décalé de +50, `card1` de +2000

---

#### 2. Concept Drift

**Définition :** Le Concept Drift se produit lorsque la relation entre les features d'entrée et le label cible change avec le temps. Les distributions de features peuvent rester les mêmes, mais ce qui constitue une fraude évolue — les fraudeurs adaptent leur comportement.

**Exemple concret (IEEE-CIS) :** Au moment de l'entraînement, les transactions provenant de domaines email anonymes avaient un taux de fraude élevé (les fraudeurs utilisaient des emails jetables). Après le démantèlement d'un réseau de fraude majeur, des utilisateurs légitimes commencent à utiliser des fournisseurs d'email axés sur la confidentialité. La même valeur de feature (`anonymous.com`) prédit maintenant une probabilité de fraude plus basse. La frontière de décision apprise par le modèle n'est plus valide même si la distribution de `P_emaildomain` n'a pas changé.

**Détection :** Surveiller les métriques de performance du modèle (F1, précision, rappel) au fil du temps. Une baisse du rappel avec des distributions de features stables signale un concept drift.

---

#### 3. Label Drift (Prior Probability Shift)

**Définition :** Le Label Drift se produit lorsque la distribution marginale de la variable cible P(Y) change, même si la distribution conditionnelle P(Y|X) reste la même. Le taux de base de fraude change.

**Exemple concret (IEEE-CIS) :** Le jeu d'entraînement IEEE-CIS a un taux de fraude de ~3.5%. Pendant la période de shopping des fêtes, les tentatives de fraude explosent et le taux de fraude réel monte à 8%. Un modèle calibré pour 3.5% de fraude sous-estimera la probabilité de fraude pour chaque transaction, causant plus de faux négatifs. Le seuil pour signaler une fraude doit être recalibré même si la relation feature-fraude est inchangée.

**Détection :** Surveiller la fraction de prédictions positives au fil du temps. Une baisse soudaine du taux de fraude prédit alors que le volume d'affaires est stable peut indiquer un label drift.

---

#### 4. Covariate Shift

**Définition :** Le Covariate Shift est une forme spécifique de data drift où P(X) change mais P(Y|X) reste constant. C'est le type le plus courant en pratique et c'est ce que Evidently et les KS-tests détectent principalement.

**Exemple concret (IEEE-CIS) :** Les données d'entraînement ont été collectées de janvier à juin (basse saison). Le modèle est déployé en novembre-décembre (saison des fêtes). La distribution de `card1` (codes d'émetteur de carte) change car plus de cartes-cadeaux prépayées sont utilisées pendant les fêtes — un type de carte sous-représenté dans l'entraînement. Le taux de fraude pour chaque type de carte est le même (P(Y|X) inchangé), mais le modèle a une mauvaise couverture de la nouvelle distribution de cartes, menant à une incertitude plus élevée et des scores mal calibrés.

**Détection :** Evidently `DataDriftPreset`, KS-test, Population Stability Index (PSI)

---

#### Tableau récapitulatif

| Type | P(X) change | P(Y\|X) change | P(Y) change |
|---|---|---|---|
| Data Drift | Oui | Non | Non |
| Concept Drift | Non | Oui | Non |
| Label Drift | Non | Non | Oui |
| Covariate Shift | Oui | Non | Non (sous-type du Data Drift) |

---

### Q13 : Quelle est la différence entre data drift et concept drift ? Donnez un exemple concret avec vos données.

**Data drift** : la distribution des features P(X) change, mais la relation P(Y|X) reste stable.

*Exemple IEEE-CIS :* `TransactionAmt` passe d'une moyenne de 150$ à 300$ (nouveau segment de clientèle haut de gamme). Le modèle n'a jamais vu ces montants en entraînement → scores de fraude mal calibrés.

**Concept drift** : P(Y|X) change — ce qui constitue une fraude évolue.

*Exemple IEEE-CIS :* Les fraudeurs adoptent de nouvelles techniques (ex : transactions fractionnées en petits montants au lieu de gros achats). Les features restent dans la même distribution, mais leur relation avec `isFraud` change → le modèle manque les nouvelles fraudes.

**Différence clé** : le data drift est détectable statistiquement sans labels (KS-test, Evidently). Le concept drift nécessite des labels de production pour être détecté (monitoring des métriques F1/recall).

Dans ce projet, `simulate_drift.py` simule du data drift (covariate shift) sur `TransactionAmt` et `card1`.

---

### Q14 : Le KS-test et Evidently identifient-ils les mêmes features comme driftées ? Pourquoi ?

Pas nécessairement, pour plusieurs raisons :

1. **Seuils différents :**
   - KS-test : p-value < 0.05 → feature driftée.
   - Evidently : utilise par défaut le test de Wasserstein ou KS selon le type de feature, avec un seuil de `stattest_threshold` configurable.

2. **Méthodes statistiques différentes :**
   - KS-test mesure la distance maximale entre les CDFs (sensible aux décalages de moyenne et de forme).
   - Evidently peut utiliser la divergence de Jensen-Shannon, chi² (catégorielles), ou PSI selon le type de colonne.

3. **Taille d'échantillon :**
   - Avec de grands échantillons, le KS-test détecte des différences statistiquement significatives mais pratiquement négligeables.

En pratique, les deux méthodes s'accordent sur les features fortement driftées (`TransactionAmt`, `card1` dans notre simulation). Les divergences apparaissent sur les features avec drift faible ou modéré.

---

### Q15 : Quel seuil de drift choisiriez-vous pour votre projet ? Justifiez selon le domaine métier.

**Seuils choisis dans ce projet :**

- `SEUIL_WARN = 0.15` (15% des features driftées → alerte)
- `SEUIL_DRIFT = 0.30` (30% des features driftées → ré-entraînement)

**Justification métier :**

La détection de fraude est un domaine à fort enjeu financier et réglementaire. Un faux négatif (fraude manquée) coûte directement de l'argent et expose la banque à des litiges. Un faux positif (transaction légitime bloquée) dégrade l'expérience client.

- **15%** est un seuil d'alerte précoce : quelques features commencent à dériver, ce qui mérite une investigation sans déclencher immédiatement un ré-entraînement coûteux.
- **30%** indique un drift systémique (changement de comportement des utilisateurs, nouveau type de fraude, changement de système de paiement). À ce niveau, les performances du modèle sont probablement dégradées et le ré-entraînement est justifié.

Un seuil plus bas (ex : 10%) serait trop sensible et déclencherait des ré-entraînements inutiles. Un seuil plus haut (ex : 50%) laisserait le modèle dégrader trop longtemps avant intervention.

---

### Q16 : Sans pipeline MLOps automatisé, comment détecteriez-vous ce drift en pratique ? Quels sont les risques ?

**Méthodes manuelles :**

1. **Monitoring des métriques métier** : surveiller le taux de fraude détectée, le nombre de chargebacks, les plaintes clients. Une dégradation visible signale un problème — mais souvent trop tard.
2. **Rapports périodiques** : un data scientist génère manuellement des statistiques descriptives (moyenne, std, quantiles) sur les données de production chaque semaine et les compare visuellement au training set.
3. **A/B testing** : comparer les prédictions du modèle actuel à un modèle ré-entraîné sur des données récentes.

**Risques sans automatisation :**

1. **Détection tardive** : le drift peut s'accumuler pendant des semaines avant d'être remarqué, causant des pertes financières significatives.
2. **Biais humain** : les analystes peuvent manquer des drifts subtils ou interpréter incorrectement les statistiques.
3. **Pas de traçabilité** : sans MLflow, il est impossible de corréler une dégradation de performance avec un drift spécifique sur une feature.
4. **Réactivité lente** : le ré-entraînement manuel prend des jours au lieu de minutes, pendant lesquels le modèle continue à mal performer.
5. **Risque réglementaire** : les régulateurs (RGPD, directives bancaires) exigent une surveillance continue des modèles d'IA en production.
