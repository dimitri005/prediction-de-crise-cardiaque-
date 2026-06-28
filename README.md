# Prédiction de Crise Cardiaque — SAMIRA

Application web de prédiction du risque de crise cardiaque basée sur un pipeline Machine Learning complet (XGBoost), déployée via Flask avec une interface glassmorphism et une jauge de risque animée.

## Contexte

Les maladies cardiovasculaires représentent la première cause de mortalité mondiale. Ce projet construit un système de détection précoce du risque cardiaque à partir d'indicateurs cliniques et comportementaux, permettant une évaluation instantanée du niveau de risque d'un patient.

## Dataset

- **Source** : Heart Attack Prediction Dataset (Kaggle)
- **Volume** : plusieurs milliers de patients
- **Features** : âge, sexe, cholestérol, pression artérielle, fréquence cardiaque, diabète, tabagisme, obésité, IMC, antécédents familiaux, activité physique, alimentation, stress...
- **Variable cible** : risque de crise cardiaque (binaire : 0 / 1)

## Pipeline ML

### Preprocessing
- Nettoyage et traitement des valeurs manquantes
- Encodage des variables catégorielles
- Standardisation des features numériques (`StandardScaler`)
- Gestion du déséquilibre de classes

### Modélisation
- **Algorithme retenu** : XGBoost (gradient boosting)
- **Validation** : cross-validation stratifiée
- **Comparaison** : plusieurs algorithmes évalués (Logistic Regression, Random Forest, XGBoost)
- **Optimisation** : GridSearchCV sur les hyperparamètres

### Métriques
- AUC-ROC (métrique principale)
- Précision / Rappel / F1-score
- Matrice de confusion

## Application Web

Interface Flask avec design glassmorphism et jauge SVG animée affichant le niveau de risque en temps réel.

**Fonctionnalités :**
- Formulaire de saisie des paramètres cliniques du patient
- Prédiction instantanée avec score de probabilité
- Jauge animée : risque Faible / Modéré / Élevé / Critique
- Design responsive adapté aux écrans médicaux

## Structure du projet

```
.
├── app.py                          # Application Flask
├── prediction.py                   # Module de prédiction
├── best_model_heart_attack.pkl     # Modèle XGBoost entraîné
├── scaler_heart_attack.pkl         # Scaler sérialisé
├── samira.ipynb                    # Notebook — pipeline ML complet
├── heart_attack_prediction_dataset.csv
├── requirements.txt
├── templates/
│   └── index.html                  # Interface glassmorphism
├── static/
│   └── style.css
└── CardioPrédict_Documentation.docx
```

## Lancement

```bash
# Cloner le projet
git clone https://github.com/dimitri005/prediction-de-crise-cardiaque-.git
cd prediction-de-crise-cardiaque-

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
python app.py
```

L'application sera disponible sur `http://localhost:5000`.

## Stack technique

- **Langage** : Python 3.12
- **ML** : XGBoost, Scikit-learn, Pandas, NumPy, SciPy
- **Web** : Flask, Gunicorn
- **Design** : CSS Glassmorphism, SVG animé (jauge de risque)
- **Versioning** : Git / GitHub

## Auteur

**Jimmy Kenmo** — Data Science & Business Intelligence  
Université de Dschang — Licence Professionnelle SIAD  
[Portfolio](https://portfolio-data-kenmo.netlify.app)
