# Guide Utilisateur - Agent IA Poker

## Installation

### Prérequis
- Windows 11
- Python 3.9+
- Tesseract OCR (pour la reconnaissance de texte)

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Installation de Tesseract OCR
1. Téléchargez Tesseract depuis : https://github.com/UB-Mannheim/tesseract/wiki
2. Installez-le dans `C:\Program Files\Tesseract-OCR`
3. Ajoutez le chemin aux variables d'environnement PATH

## Configuration

### Fichier config.ini
Le fichier de configuration permet de personnaliser le comportement de l'agent :

```ini
[Display]
target_window_title=PokerStars
capture_fps=10
debug_mode=false

[AI]
aggression_level=0.7
bluff_frequency=0.15
risk_tolerance=0.8
bankroll_management=true

[Automation]
click_randomization=5
move_speed_min=0.1
move_speed_max=0.3
human_delays=true

[Safety]
max_hands_per_hour=180
emergency_fold_key=F12
auto_pause_on_detection=true
```

### Paramètres principaux

#### Display
- `target_window_title` : Titre de la fenêtre de poker à cibler
- `capture_fps` : Fréquence de capture d'écran (1-60)
- `debug_mode` : Mode debug pour plus de logs

#### AI
- `aggression_level` : Niveau d'agressivité (0-1)
- `bluff_frequency` : Fréquence de bluff (0-1)
- `risk_tolerance` : Tolérance au risque (0-1)
- `bankroll_management` : Gestion de bankroll activée

#### Automation
- `click_randomization` : Randomisation des clics en pixels
- `move_speed_min/max` : Vitesse des mouvements de souris
- `human_delays` : Délais humains simulés

#### Safety
- `max_hands_per_hour` : Limite de mains par heure
- `emergency_fold_key` : Touche de fold d'urgence
- `auto_pause_on_detection` : Pause automatique si détection

## Utilisation

### Démarrage basique
```bash
python main.py
```

### Mode simulation (recommandé pour les tests)
```bash
python main.py --mode=simulation
```

### Mode debug
```bash
python main.py --debug
```

### Configuration personnalisée
```bash
python main.py --config=ma_config.ini
```

## Fonctionnalités

### 1. Reconnaissance visuelle
L'agent capture et analyse l'écran pour :
- Détecter les cartes (votre main + cartes communes)
- Lire les montants (pot, mises, stack)
- Identifier les boutons disponibles
- Reconnaître les informations des joueurs

### 2. Prise de décision IA
L'agent utilise plusieurs stratégies :
- **GTO (Game Theory Optimal)** : Stratégies équilibrées
- **Exploitative play** : Adaptation aux adversaires
- **Monte Carlo simulation** : Calcul des probabilités
- **Position-based play** : Ajustements selon la position

### 3. Automatisation furtive
- Mouvements courbes naturels
- Délais variables
- Randomisation des clics
- Anti-détection intégrée

### 4. Sécurité
- Fold d'urgence (F12)
- Limites de mains/heure
- Pause automatique
- Logs détaillés

## Contrôles

### Pendant l'exécution
- `Ctrl+C` : Arrêt propre
- `F12` : Fold d'urgence
- `Pause` : Pause/Reprise (si configuré)

### Logs
Les logs sont sauvegardés dans `poker_ai.log` :
- Actions prises
- Décisions IA
- Erreurs et avertissements
- Statistiques

## Statistiques

L'agent affiche des statistiques en temps réel :
- Mains jouées
- Taux de victoire
- Profit/Perte
- Temps d'exécution

## Dépannage

### Problèmes courants

#### Fenêtre non trouvée
```
Erreur: Fenêtre de poker non trouvée
```
**Solution** : Vérifiez que la fenêtre de poker est ouverte et que le titre correspond dans `config.ini`

#### Erreur de capture d'écran
```
Erreur: Échec de capture d'écran
```
**Solution** : Vérifiez les permissions d'accès à l'écran

#### Reconnaissance incorrecte
```
Erreur: Cartes non reconnues
```
**Solution** : 
1. Ajustez les coordonnées des régions dans `screen_capture.py`
2. Améliorez l'éclairage
3. Vérifiez la résolution d'écran

#### Actions non exécutées
```
Erreur: Boutons non trouvés
```
**Solution** : Calibrez les positions des boutons pour votre interface

### Mode debug
Activez le mode debug pour plus d'informations :
```bash
python main.py --debug
```

## Personnalisation

### Ajustement des coordonnées
Modifiez les régions de capture dans `modules/screen_capture.py` :

```python
self.regions = {
    'hand_area': {'x': 400, 'y': 500, 'width': 200, 'height': 100},
    'community_cards': {'x': 300, 'y': 300, 'width': 400, 'height': 80},
    # ...
}
```

### Stratégie personnalisée
Modifiez les paramètres dans `config.ini` ou directement dans le code :
- Agressivité
- Fréquence de bluff
- Tolérance au risque

### Templates de cartes
Ajoutez vos propres templates de cartes dans `data/card_templates/`

## Sécurité et légalité

### ⚠️ Avertissements importants
1. **Vérifiez la légalité** dans votre juridiction
2. **Respectez les ToS** des plateformes de poker
3. **Usage responsable** uniquement
4. **Risque de détection** toujours présent

### Recommandations
- Utilisez uniquement sur sites autorisant les bots
- Limitez les sessions (max 4h)
- Surveillez les logs pour détecter les anomalies
- Ayez un plan de sortie d'urgence

## Support

### Logs détaillés
Consultez `poker_ai.log` pour :
- Détails des décisions
- Erreurs techniques
- Statistiques de performance

### Tests
Exécutez les tests pour vérifier le bon fonctionnement :
```bash
python -m unittest tests/test_poker_engine.py
```

### Mise à jour
1. Sauvegardez votre configuration
2. Mettez à jour le code
3. Testez en mode simulation
4. Vérifiez la compatibilité

## Performance

### Métriques cibles
- Précision reconnaissance : >95%
- Latence décision : <500ms
- Uptime : >99%
- Win rate : Positif sur 10k+ mains

### Optimisation
- Ajustez `capture_fps` selon votre CPU
- Réduisez les régions de capture
- Utilisez le mode simulation pour les tests

---

**Note** : Ce guide est fourni à titre informatif. L'utilisation de bots de poker peut être illégale dans certaines juridictions. Utilisez à vos propres risques. 