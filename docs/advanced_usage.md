# Guide d'Utilisation Avancée - Agent IA Poker

## 🎯 **1. CALIBRATION DES RÉGIONS**

### Utilisation de l'outil de calibration
```bash
python tools/calibration_tool.py
```

**Instructions :**
1. Ouvrez votre client poker
2. Lancez l'outil de calibration
3. Cliquez sur une région pour la sélectionner (devient verte)
4. Glissez pour déplacer la région
5. Utilisez les touches pour redimensionner :
   - `w`/`s` : Hauteur
   - `a`/`d` : Largeur
   - `r` : Réinitialiser
   - `s` : Sauvegarder
   - `l` : Charger
   - `q` : Quitter

### Régions à calibrer :
- **Cartes du joueur** : Vos cartes privées
- **Cartes communes** : Flop, turn, river
- **Zone du pot** : Montant du pot
- **Boutons d'action** : Fold, Call, Raise, etc.
- **Infos joueur** : Stack, position
- **Zone chat** : Messages système

## 🎯 **2. CONFIGURATION AVANCÉE**

### Paramètres de performance
```ini
[Display]
capture_fps=15          # FPS de capture (10-30)
debug_mode=true         # Mode debug pour logs détaillés

[AI]
aggression_level=0.7    # Niveau d'agressivité (0.1-1.0)
bluff_frequency=0.15    # Fréquence de bluff (0.0-0.3)
risk_tolerance=0.8      # Tolérance au risque (0.1-1.0)

[Automation]
click_randomization=5    # Randomisation des clics (pixels)
move_speed_min=0.1      # Vitesse min de mouvement (sec)
move_speed_max=0.3      # Vitesse max de mouvement (sec)
human_delays=true       # Délais humains

[Safety]
max_hands_per_hour=180  # Limite de mains/heure
emergency_fold_key=F12  # Touche d'urgence
auto_pause_on_detection=true
```

### Profils de jeu
```bash
# Profil conservateur
python main.py --config=conservative.ini

# Profil agressif
python main.py --config=aggressive.ini

# Profil équilibré (défaut)
python main.py --config=balanced.ini
```

## 🎯 **3. MODES DE FONCTIONNEMENT**

### Mode Simulation (Recommandé pour débuter)
```bash
python main.py --mode=simulation --debug
```
- ✅ **Sans risque** : Aucune action réelle
- ✅ **Test complet** : Toutes les fonctionnalités
- ✅ **Debug** : Logs détaillés
- ✅ **Calibration** : Ajustement des paramètres

### Mode Live (Production)
```bash
python main.py --mode=live --config=production.ini
```
- ⚠️ **Actions réelles** : Clics et mises effectifs
- ⚠️ **Responsabilité** : Vérifiez les paramètres
- ⚠️ **Surveillance** : Surveillez les logs

### Mode Debug
```bash
python main.py --debug --verbose
```
- 📊 **Logs détaillés** : Toutes les décisions
- 📊 **Métriques** : Performance et précision
- 📊 **Analyse** : Comportement de l'agent

## 🎯 **4. OPTIMISATION DES PERFORMANCES**

### Optimisation CPU
```python
# Dans config.ini
[Display]
capture_fps=10          # Réduire pour économiser CPU
debug_mode=false        # Désactiver en production

[AI]
simulation_count=500    # Réduire les simulations Monte Carlo
```

### Optimisation Mémoire
```python
# Nettoyage automatique
import gc
gc.collect()  # Forcer le garbage collection
```

### Optimisation Réseau
```python
# Délais entre actions
[Automation]
min_delay_between_actions=0.5
max_delay_between_actions=2.0
```

## 🎯 **5. SÉCURITÉ ET ANTI-DÉTECTION**

### Mesures de sécurité
- ✅ **Randomisation** : Tous les timings et mouvements
- ✅ **Profils humains** : Simulation de styles de jeu
- ✅ **Pauses automatiques** : Pauses réalistes
- ✅ **Erreurs simulées** : Erreurs humaines occasionnelles

### Surveillance
```python
# Détection d'anomalies
[Safety]
detect_captcha=true
detect_anti_bot=true
detect_suspicious_patterns=true
```

### Raccourcis d'urgence
- **F12** : Fold d'urgence
- **Ctrl+C** : Arrêt immédiat
- **Alt+F4** : Fermeture forcée

## 🎯 **6. ANALYSE ET STATISTIQUES**

### Métriques de performance
```python
# Statistiques automatiques
- Mains jouées par heure
- Taux de victoire
- Profit/perte
- Précision de reconnaissance
- Latence de décision
```

### Logs d'analyse
```bash
# Logs détaillés
tail -f poker_ai.log | grep "DECISION"
tail -f poker_ai.log | grep "ERROR"
tail -f poker_ai.log | grep "STATISTICS"
```

### Export des données
```python
# Export CSV des mains
python tools/export_hands.py --format=csv --output=hands.csv
```

## 🎯 **7. DÉPANNAGE AVANCÉ**

### Problèmes courants

#### Erreur "Fenêtre poker non trouvée"
```bash
# Solution 1: Vérifier le titre de la fenêtre
python tools/window_finder.py

# Solution 2: Mode écran complet
python main.py --mode=simulation --fullscreen
```

#### Erreur "Tesseract non disponible"
```bash
# Solution 1: Installer Tesseract
# Suivre docs/installation_guide.md

# Solution 2: Utiliser l'estimation par couleur
# L'agent fonctionne déjà avec fallback
```

#### Performance lente
```bash
# Solution 1: Réduire FPS
[Display]
capture_fps=5

# Solution 2: Désactiver debug
[Display]
debug_mode=false
```

#### Reconnaissance imprécise
```bash
# Solution 1: Recalibrer
python tools/calibration_tool.py

# Solution 2: Ajuster les seuils
[AI]
confidence_threshold=0.8
```

## 🎯 **8. INTÉGRATION AVANCÉE**

### API REST (Futur)
```python
# Endpoint pour contrôler l'agent
POST /api/agent/start
POST /api/agent/stop
GET /api/agent/status
GET /api/agent/statistics
```

### Interface Web (Futur)
```python
# Dashboard web
python tools/web_dashboard.py
# Accès: http://localhost:8080
```

### Multi-tabling (Futur)
```python
# Support plusieurs tables
python main.py --tables=2 --stakes=nl10
```

## 🎯 **9. DÉVELOPPEMENT ET EXTENSION**

### Ajouter de nouveaux sites
```python
# Créer un nouveau module
modules/sites/pokerstars.py
modules/sites/partypoker.py
```

### Personnaliser les stratégies
```python
# Modifier les ranges
modules/strategy_engine.py
# Lignes 40-80: Ranges pré-flop
```

### Ajouter de nouvelles fonctionnalités
```python
# Nouveau module
modules/advanced_analytics.py
modules/machine_learning.py
```

## 🎯 **10. RESSOURCES ET SUPPORT**

### Documentation
- `docs/installation_guide.md` : Installation
- `docs/user_guide.md` : Guide utilisateur
- `docs/advanced_usage.md` : Ce guide
- `README.md` : Vue d'ensemble

### Outils
- `tools/calibration_tool.py` : Calibration
- `tools/window_finder.py` : Trouver fenêtres
- `tools/export_hands.py` : Export données

### Logs
- `poker_ai.log` : Logs principaux
- `debug.log` : Logs debug (si activé)

### Configuration
- `config.ini` : Configuration principale
- `calibrated_regions.json` : Régions calibrées

---

**⚠️ AVERTISSEMENT LÉGAL :**
L'utilisation de cet agent doit respecter les conditions d'utilisation de votre site de poker. L'utilisateur est entièrement responsable de l'utilisation de cet outil. 