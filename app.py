# ═══════════════════════════════════════════════════════════════
# app.py — Serveur Flask : point d'entrée de l'application
#
# Responsabilités :
# 1. Créer l'application Flask
# 2. Définir les routes (URLs accessibles)
# 3. Recevoir les données JSON du formulaire HTML
# 4. Appeler prediction.py pour obtenir la prédiction
# 5. Retourner la réponse JSON au navigateur
# ═══════════════════════════════════════════════════════════════

from flask import Flask, render_template, request, jsonify
# Flask          : crée l'application web
# render_template: charge index.html depuis /templates
# request        : lit les données envoyées par le navigateur
# jsonify        : convertit un dict Python en réponse JSON

from prediction import predire
# Importe la fonction de prédiction depuis prediction.py
# Au démarrage, cela charge aussi le modèle et le scaler

import logging
# Logs pour suivre les requêtes et détecter les erreurs


# ─────────────────────────────────────────────────────────────
# CONFIGURATION DES LOGS
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# CRÉATION DE L'APPLICATION FLASK
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)


# ─────────────────────────────────────────────────────────────
# ROUTE 1 : Page principale
# URL    : GET http://localhost:5000/
# Action : retourner le formulaire HTML (index.html)
# ─────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    logger.info("Page principale chargée")
    return render_template('index.html')


# ─────────────────────────────────────────────────────────────
# ROUTE 2 : Prédiction
# URL    : POST http://localhost:5000/predict
# Action : recevoir les données patient en JSON,
#          construire le payload complet avec Income,
#          appeler predire(), retourner le résultat JSON
#
# Appelée automatiquement par le JavaScript de index.html
# via fetch('/predict', { method: 'POST', body: JSON })
# ─────────────────────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():

    # ── Lire les données JSON envoyées par le navigateur ─────
    # get_json() désérialise le corps de la requête POST
    # silent=True : retourne None si le JSON est malformé
    data = request.get_json(force=True, silent=True)

    # ── Vérifier que des données ont bien été reçues ─────────
    if data is None:
        logger.warning("Requête reçue sans données JSON valides")
        return jsonify({'erreur': 'Aucune donnée reçue'}), 400

    logger.info(
        f"Données reçues — Age={data.get('Age')}, "
        f"Cholesterol={data.get('Cholesterol')}, "
        f"Systolique={data.get('Systolique')}"
    )

    # ── Construire le payload complet pour prediction.py ─────
    # On mappe chaque clé reçue du formulaire vers le nom
    # attendu par le modèle
    # Income n'est pas dans le formulaire → valeur médiane 50000
    # (valeur représentative du dataset d'entraînement)
    payload = {
        'Age'                            : data.get('Age'),
        'Sex'                            : data.get('Sex'),
        'Cholesterol'                    : data.get('Cholesterol'),
        'Heart Rate'                     : data.get('Heart Rate'),
        'Diabetes'                       : data.get('Diabetes'),
        'Family History'                 : data.get('Family History'),
        'Smoking'                        : data.get('Smoking'),
        'Obesity'                        : data.get('Obesity'),
        'Alcohol Consumption'            : data.get('Alcohol Consumption'),
        'Exercise Hours Per Week'        : data.get('Exercise Hours Per Week'),
        'Diet'                           : data.get('Diet'),
        'Previous Heart Problems'        : data.get('Previous Heart Problems'),
        'Medication Use'                 : data.get('Medication Use'),
        'Stress Level'                   : data.get('Stress Level'),
        'Sedentary Hours Per Day'        : data.get('Sedentary Hours Per Day'),
        'Income'                         : 50000,
        # Income absent du formulaire HTML
        # 50000 = valeur médiane du dataset d'entraînement
        # Le scaler la normalisera comme toute autre valeur
        'BMI'                            : data.get('BMI'),
        'Triglycerides'                  : data.get('Triglycerides'),
        'Physical Activity Days Per Week': data.get('Physical Activity Days Per Week'),
        'Sleep Hours Per Day'            : data.get('Sleep Hours Per Day'),
        'Systolique'                     : data.get('Systolique'),
        'Diastolique'                    : data.get('Diastolique'),
    }

    # ── Appeler la fonction de prédiction ────────────────────
    try:
        resultat = predire(payload)
        # resultat = {'risque': 0|1, 'probabilite': 0.0 à 1.0}

        logger.info(
            f"Prédiction effectuée — "
            f"Risque : {'ÉLEVÉ' if resultat['risque'] == 1 else 'FAIBLE'} "
            f"({resultat['probabilite']*100:.1f}%)"
        )

        # Retourner le résultat JSON au navigateur
        # Le JavaScript de index.html le reçoit et met à jour l'interface
        return jsonify(resultat)

    except ValueError as e:
        # Données invalides (valeur hors plage, type incorrect)
        logger.error(f"Erreur de valeur : {e}")
        return jsonify({'erreur': f'Données invalides : {str(e)}'}), 422

    except Exception as e:
        # Toute autre erreur inattendue
        logger.error(f"Erreur inattendue : {e}")
        return jsonify({'erreur': 'Erreur interne du serveur'}), 500


# ─────────────────────────────────────────────────────────────
# ROUTE 3 : Health check
# URL    : GET http://localhost:5000/health
# Utilisée par les plateformes de déploiement (Render, Railway)
# pour vérifier que le serveur est bien démarré
# ─────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'statut' : 'ok',
        'modele' : 'best_model_heart_attack.pkl',
        'version': '1.0.0'
    }), 200


# ─────────────────────────────────────────────────────────────
# GESTIONNAIRES D'ERREURS HTTP
# ─────────────────────────────────────────────────────────────

@app.errorhandler(404)
def page_non_trouvee(e):
    # URL inexistante
    logger.warning(f"Page non trouvée : {request.url}")
    return jsonify({'erreur': 'Page non trouvée'}), 404

@app.errorhandler(500)
def erreur_serveur(e):
    # Erreur interne non gérée
    logger.error(f"Erreur serveur : {e}")
    return jsonify({'erreur': 'Erreur interne du serveur'}), 500


# ─────────────────────────────────────────────────────────────
# DÉMARRAGE DU SERVEUR
#
# Exécuté uniquement via : python app.py
# Ne s'exécute PAS avec gunicorn (production)
#
# use_reloader=False : désactive le rechargement automatique
# qui causait l'erreur "Unable to create process"
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logger.info("Démarrage du serveur Flask — CardioPrédict")
    logger.info("Interface disponible : http://localhost:5000")
    app.run(
        debug       = True,
        port        = 5000,
        host        = '0.0.0.0',
        use_reloader= False   # évite le conflit avec le Python système
    )
