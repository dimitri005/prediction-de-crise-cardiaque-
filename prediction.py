# ═══════════════════════════════════════════════════════════════
# prediction.py — Logique de prédiction isolée
#
# Responsabilité unique : charger le modèle + scaler,
# recevoir un dictionnaire de données patient,
# retourner le risque et la probabilité
# ═══════════════════════════════════════════════════════════════

import joblib       # charger les fichiers .pkl (modèle + scaler)
import pandas as pd # créer un DataFrame à partir du dict patient
import numpy as np  # opérations numériques


# ─────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE ET DU SCALER
#
# On charge UNE SEULE FOIS au démarrage du serveur
# pas à chaque requête — crucial pour les performances
# ─────────────────────────────────────────────────────────────

try:
    model  = joblib.load('best_model_heart_attack.pkl')
    scaler = joblib.load('scaler_heart_attack.pkl')
    print("✓ Modèle et scaler chargés avec succès")
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"Fichier manquant : {e}\n"
        "Vérifiez que best_model_heart_attack.pkl et "
        "scaler_heart_attack.pkl sont dans le même dossier que app.py"
    )


# ─────────────────────────────────────────────────────────────
# COLONNES CONTINUES À NORMALISER
#
# IMPORTANT : liste identique à celle du notebook
# Income est inclus — colonne attendue par le modèle
# ─────────────────────────────────────────────────────────────

COLS_SCALE = [
    'Age',
    'Cholesterol',
    'Heart Rate',
    'Exercise Hours Per Week',
    'Stress Level',
    'Sedentary Hours Per Day',
    'Income',
    'BMI',
    'Triglycerides',
    'Physical Activity Days Per Week',
    'Sleep Hours Per Day',
    'Systolique',
    'Diastolique'
]


# ─────────────────────────────────────────────────────────────
# ORDRE DES COLONNES ATTENDU PAR LE MODÈLE
#
# Ordre exact récupéré via model.get_booster().feature_names
# Ne pas modifier cet ordre — le modèle est sensible à l'ordre
# ─────────────────────────────────────────────────────────────

COLONNES_MODELE = [
    'Age',
    'Sex',
    'Cholesterol',
    'Heart Rate',
    'Diabetes',
    'Family History',
    'Smoking',
    'Obesity',
    'Alcohol Consumption',
    'Exercise Hours Per Week',
    'Diet',
    'Previous Heart Problems',
    'Medication Use',
    'Stress Level',
    'Sedentary Hours Per Day',
    'Income',
    'BMI',
    'Triglycerides',
    'Physical Activity Days Per Week',
    'Sleep Hours Per Day',
    'Systolique',
    'Diastolique'
]


# ─────────────────────────────────────────────────────────────
# FONCTION PRINCIPALE : predire()
#
# Paramètre :
#   data (dict) — données du patient issues du formulaire HTML
#
# Retourne :
#   dict avec deux clés :
#   - 'risque'      : 0 (faible) ou 1 (élevé)
#   - 'probabilite' : float entre 0.0 et 1.0
# ─────────────────────────────────────────────────────────────

def predire(data: dict) -> dict:

    # ── Étape 1 : Convertir le dict en DataFrame pandas ──────
    # [data] crée une liste d'un seul élément → 1 ligne
    df = pd.DataFrame([data])

    # ── Étape 2 : Réordonner les colonnes dans le bon ordre ──
    # reindex() place les colonnes dans l'ordre de COLONNES_MODELE
    # et met NaN pour les colonnes absentes
    df = df.reindex(columns=COLONNES_MODELE)

    # ── Étape 3 : Remplacer les valeurs manquantes par 0 ─────
    # Si une colonne est absente du dict reçu → NaN → 0
    if df.isnull().any().any():
        cols_manquantes = df.columns[df.isnull().any()].tolist()
        print(f"⚠ Colonnes manquantes remplacées par 0 : {cols_manquantes}")
        df = df.fillna(0)

    # ── Étape 4 : Appliquer le RobustScaler ──────────────────
    # transform() applique la normalisation apprise à l'entraînement
    # formule : (valeur - médiane) / IQR
    df[COLS_SCALE] = scaler.transform(df[COLS_SCALE])

    # ── Étape 5 : Prédiction ──────────────────────────────────
    # predict()       → classe : 0 ou 1
    # predict_proba() → [P(classe=0), P(classe=1)]
    # [0][1]          → probabilité de risque élevé pour le 1er patient
    risque      = int(model.predict(df)[0])
    probabilite = float(model.predict_proba(df)[0][1])

    # ── Étape 6 : Retourner le résultat ──────────────────────
    return {
        'risque'      : risque,
        'probabilite' : round(probabilite, 4)
    }


# ─────────────────────────────────────────────────────────────
# TEST RAPIDE — exécuté uniquement via : python prediction.py
# Ne s'exécute PAS quand importé par app.py
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    patient_test = {
        'Age'                            : 65,
        'Sex'                            : 1,
        'Cholesterol'                    : 320,
        'Heart Rate'                     : 95,
        'Diabetes'                       : 1,
        'Family History'                 : 1,
        'Smoking'                        : 1,
        'Obesity'                        : 1,
        'Alcohol Consumption'            : 0,
        'Exercise Hours Per Week'        : 1.0,
        'Diet'                           : 2,
        'Previous Heart Problems'        : 1,
        'Medication Use'                 : 0,
        'Stress Level'                   : 9,
        'Sedentary Hours Per Day'        : 10.0,
        'Income'                         : 30000,
        'BMI'                            : 34.5,
        'Triglycerides'                  : 380,
        'Physical Activity Days Per Week': 1,
        'Sleep Hours Per Day'            : 5.0,
        'Systolique'                     : 165,
        'Diastolique'                    : 100
    }

    resultat = predire(patient_test)
    print(f"\nRésultat du test :")
    print(f"  Risque      : {'ÉLEVÉ' if resultat['risque'] == 1 else 'FAIBLE'}")
    print(f"  Probabilité : {resultat['probabilite'] * 100:.1f}%")
