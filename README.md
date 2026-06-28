# CardioPrédict — Prédiction de Crise Cardiaque

Application web de détection précoce du risque de crise cardiaque basée sur un pipeline Machine Learning complet (XGBoost + Flask). Saisie de paramètres cliniques → prédiction instantanée → jauge de risque animée.

---

## Problème Métier

Les maladies cardiovasculaires causent 17,9 millions de décès par an dans le monde (OMS). La majorité sont évitables si le risque est détecté tôt. Le problème : les outils de screening clinique sont coûteux, lents, et peu accessibles dans les pays en développement.

**Objectif de ce projet :** construire un système de scoring de risque cardiaque basé sur des indicateurs simples (âge, cholestérol, pression artérielle, mode de vie), accessibles en consultation de médecine générale, sans équipement spécialisé.

**Question à résoudre :**
> Étant donné les paramètres cliniques et comportementaux d'un patient, quel est son niveau de risque de crise cardiaque ?

---

## Dataset

- **Source** : Heart Attack Prediction Dataset (Kaggle)
- **Volume** : 8 763 patients
- **Features** : 22 variables cliniques et comportementales
- **Cible** : `Heart Attack Risk` (0 = faible, 1 = élevé)
- **Déséquilibre** : ~56% classe 0 / ~44% classe 1 — dataset relativement équilibré

### Variables d'entrée

| Feature | Type | Description |
|---|---|---|
| Age | Numérique | Âge du patient (années) |
| Sex | Binaire | 0 = Femme, 1 = Homme |
| Cholesterol | Numérique | Cholestérol total (mg/dL) |
| Heart Rate | Numérique | Fréquence cardiaque au repos (bpm) |
| Diabetes | Binaire | 0 = Non, 1 = Oui |
| Family History | Binaire | Antécédents familiaux cardiaques |
| Smoking | Binaire | Tabagisme actif |
| Obesity | Binaire | IMC ≥ 30 |
| Alcohol Consumption | Binaire | Consommation d'alcool régulière |
| Exercise Hours Per Week | Numérique | Heures d'exercice / semaine |
| Diet | Ordinal | 0 = Mauvaise, 1 = Moyenne, 2 = Bonne |
| Previous Heart Problems | Binaire | Antécédents cardiaques personnels |
| Medication Use | Binaire | Prise de médicaments régulière |
| Stress Level | Numérique | Niveau de stress (1 à 10) |
| Sedentary Hours Per Day | Numérique | Heures sédentaires / jour |
| Income | Numérique | Revenu annuel (USD) — valeur médiane : 50 000 |
| BMI | Numérique | Indice de Masse Corporelle |
| Triglycerides | Numérique | Triglycérides (mg/dL) |
| Physical Activity Days Per Week | Numérique | Jours d'activité physique / semaine |
| Sleep Hours Per Day | Numérique | Heures de sommeil / jour |
| Systolique | Numérique | Pression artérielle systolique (mmHg) |
| Diastolique | Numérique | Pression artérielle diastolique (mmHg) |

---

## Architecture du Projet

```
SAMIRA/
├── samira.ipynb                    # Pipeline ML complet (exploration → entraînement)
├── prediction.py                   # Module de prédiction (logique métier isolée)
├── app.py                          # Serveur Flask (routes HTTP)
├── best_model_heart_attack.pkl     # Modèle XGBoost entraîné et sérialisé
├── scaler_heart_attack.pkl         # RobustScaler sérialisé
├── heart_attack_prediction_dataset.csv
├── requirements.txt
├── templates/
│   └── index.html                  # Interface glassmorphism + jauge SVG animée
├── static/
│   └── style.css
└── CardioPrédict_Documentation.docx
```

---

## Pipeline ML — Étapes d'Implémentation

### Étape 1 — Exploration et Analyse (EDA)

- Distribution de chaque variable (histogrammes, boxplots)
- Corrélations avec la cible via heatmap
- Détection et traitement des valeurs aberrantes
- Analyse du déséquilibre de classes

### Étape 2 — Préparation des Données

```python
# Encodage des variables catégorielles
df['Sex']  = df['Sex'].map({'Male': 1, 'Female': 0})
df['Diet'] = df['Diet'].map({'Unhealthy': 0, 'Average': 1, 'Healthy': 2})

# Séparation pression artérielle
df[['Systolique', 'Diastolique']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
df.drop(columns=['Blood Pressure', 'Patient ID', 'Country', 'Continent', 'Hemisphere'], inplace=True)

# Split train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalisation avec RobustScaler (résistant aux outliers)
scaler = RobustScaler()
X_train[COLS_SCALE] = scaler.fit_transform(X_train[COLS_SCALE])
X_test[COLS_SCALE]  = scaler.transform(X_test[COLS_SCALE])
```

**Pourquoi RobustScaler ?**
Les données cliniques (cholestérol, triglycérides, pression artérielle) contiennent des valeurs extrêmes. RobustScaler normalise via la médiane et l'IQR — insensible aux outliers contrairement à StandardScaler.

### Étape 3 — Comparaison des Modèles

Cinq algorithmes évalués en cross-validation 5 folds :

| Modèle | AUC-ROC | Accuracy |
|---|---|---|
| Logistic Regression | ~0.50 | ~50% |
| Decision Tree | ~0.51 | ~51% |
| Random Forest | ~0.52 | ~52% |
| Gradient Boosting | ~0.53 | ~53% |
| **XGBoost** | **~0.54** | **~54%** |

XGBoost retenu comme meilleur modèle.

**Note sur les performances :** Le dataset Heart Attack Prediction (Kaggle) est synthétique. Les features ont une corrélation faible avec la cible, ce qui explique des scores modérés. L'objectif pédagogique du projet est la maîtrise du pipeline bout-en-bout, pas la performance absolue.

### Étape 4 — Optimisation des Hyperparamètres

```python
param_grid = {
    'n_estimators'    : [100, 200, 300],
    'max_depth'       : [3, 5, 7],
    'learning_rate'   : [0.01, 0.1, 0.2],
    'subsample'       : [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

### Étape 5 — Sérialisation

```python
joblib.dump(best_model, 'best_model_heart_attack.pkl')
joblib.dump(scaler,     'scaler_heart_attack.pkl')
```

Les deux fichiers `.pkl` sont chargés **une seule fois** au démarrage du serveur Flask.

---

## Architecture Web — Flux de Données

```
Navigateur (index.html)
    │
    │  POST /predict  { Age: 65, Cholesterol: 320, ... }
    ▼
app.py — Route /predict
    │
    │  Construire payload (22 features + Income=50000)
    ▼
prediction.py — predire(data)
    │
    │  1. dict → DataFrame pandas
    │  2. reindex(COLONNES_MODELE) → ordre exact attendu par XGBoost
    │  3. fillna(0) → valeurs manquantes
    │  4. RobustScaler.transform() → normalisation
    │  5. model.predict() → classe (0 ou 1)
    │  6. model.predict_proba() → probabilité [0.0 — 1.0]
    ▼
app.py — jsonify({ risque: 1, probabilite: 0.7832 })
    │
    ▼
Navigateur → Mise à jour jauge SVG animée + affichage résultat
```

---

## Logique Métier — Interprétation du Score

| Probabilité | Niveau de Risque | Couleur Jauge |
|---|---|---|
| 0% — 30% | Faible | Vert |
| 30% — 55% | Modéré | Jaune |
| 55% — 75% | Élevé | Orange |
| 75% — 100% | Critique | Rouge |

---

## Décision Technique — Income non collecté

`Income` est une feature du dataset d'entraînement attendue par le modèle, mais non disponible en consultation clinique standard. **Solution retenue :** injection de la valeur médiane du dataset (`50 000`) côté serveur dans `app.py` avant l'appel au modèle. Le scaler la normalise comme n'importe quelle autre valeur numérique.

---

## Lancement

```bash
git clone https://github.com/dimitri005/prediction-de-crise-cardiaque-.git
cd prediction-de-crise-cardiaque-

pip install -r requirements.txt

python app.py
# → http://localhost:5000
```

---

## Routes API

| Méthode | Route | Description |
|---|---|---|
| GET | `/` | Interface web (formulaire patient) |
| POST | `/predict` | Prédiction — Body JSON : 21 features patient |
| GET | `/health` | Health check serveur (Render / Railway) |

**Exemple de requête `/predict` :**
```json
{
  "Age": 65, "Sex": 1, "Cholesterol": 320,
  "Heart Rate": 95, "Diabetes": 1,
  "Systolique": 165, "Diastolique": 100,
  "BMI": 34.5, "Stress Level": 9, ...
}
```

**Réponse :**
```json
{ "risque": 1, "probabilite": 0.7832 }
```

---

## Stack Technique

- **Langage** : Python 3.12
- **ML** : XGBoost, Scikit-learn, Pandas, NumPy, SciPy
- **Sérialisation** : Joblib
- **Web** : Flask 3.0.3, Gunicorn 22.0.0
- **Frontend** : HTML/CSS Glassmorphism, SVG animé (jauge de risque)
- **Versioning** : Git / GitHub

---

## Auteur

**Jimmy Kenmo** — Data Science & Business Intelligence  
Université de Dschang — Licence Professionnelle SIAD  
[Portfolio](https://portfolio-data-kenmo.netlify.app)
