# ═══════════════════════════════════════════════════════════════
# Dockerfile — CardioPrédict
#
# Construction : docker build -t cardiopredict .
# Lancement    : docker run -p 5000:5000 cardiopredict
# ═══════════════════════════════════════════════════════════════

# ── Image de base ────────────────────────────────────────────
# python:3.12-slim = Python 3.12 sur Debian minimal
# "slim" : pas d'outils de compilation ni de docs
# Résultat : image légère (~130 Mo vs ~900 Mo pour python:3.12)
FROM python:3.12-slim

# ── Métadonnées de l'image ───────────────────────────────────
LABEL maintainer="NOVA-TECH-SOLUTION"
LABEL description="CardioPrédict — Outil IA de prédiction du risque cardiaque"
LABEL version="1.0.0"

# ── Variables d'environnement ────────────────────────────────
# PYTHONDONTWRITEBYTECODE : ne pas créer les fichiers .pyc
# PYTHONUNBUFFERED       : afficher les logs Flask en temps réel
#                          sans mise en tampon (essentiel en prod)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    PORT=5000

# ── Répertoire de travail dans le container ──────────────────
# Tous les fichiers copiés iront dans /app
# Toutes les commandes s'exécutent depuis /app
WORKDIR /app

# ── Copier uniquement requirements.txt en premier ────────────
# Docker met en cache chaque couche (layer)
# Si requirements.txt n'a pas changé, pip install est sauté
# → rebuilds beaucoup plus rapides lors des modifications du code
COPY requirements.txt .

# ── Installer les dépendances Python ─────────────────────────
# --no-cache-dir : ne pas garder le cache pip dans l'image
#                  réduit la taille finale de l'image
# --upgrade pip  : s'assurer d'avoir la dernière version de pip
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt

# ── Copier tout le code source dans l'image ──────────────────
# Le .dockerignore exclut les fichiers inutiles (venv, notebooks, etc.)
# Cette étape est après pip install pour profiter du cache Docker
COPY . .

# ── Port exposé ───────────────────────────────────────────────
# Déclare que le container écoute sur le port 5000
# N'ouvre pas réellement le port — c'est docker run -p qui le fait
EXPOSE 5000

# ── Vérification de santé du container ───────────────────────
# Docker vérifie toutes les 30s que l'app répond sur /health
# Après 3 échecs → container marqué "unhealthy"
# Utile pour les orchestrateurs (Kubernetes, Docker Swarm)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" \
  || exit 1

# ── Commande de démarrage ────────────────────────────────────
# gunicorn remplace le serveur Flask intégré (développement uniquement)
# Gunicorn est un serveur WSGI de production :
#   - gère plusieurs requêtes simultanées (workers)
#   - stable et sécurisé
#
# Paramètres :
#   -w 2        : 2 workers (processus) — adapté à un petit VPS
#                 Règle générale : 2 * CPU + 1
#   -b 0.0.0.0:5000 : écouter sur toutes les interfaces, port 5000
#   --timeout 120   : timeout de 120 secondes par requête
#                     XGBoost peut être lent au premier chargement
#   app:app     : fichier app.py, objet Flask nommé "app"
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]
