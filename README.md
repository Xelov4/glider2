# 🤖 Agent IA Poker - Betclic Poker

## 📋 Description

Agent IA intelligent pour jouer au poker sur Betclic Poker. Utilise l'OCR, la reconnaissance d'images et des stratégies avancées pour prendre des décisions optimales.

## 🚀 Fonctionnalités

- **Détection automatique** des cartes, boutons et éléments de jeu
- **Stratégie Spin & Rush** ultra-agressive pour les tournois rapides
- **Analyse en temps réel** du pot, des stacks et des actions adverses
- **Réactivité ultra-rapide** (décisions en < 100ms)
- **Calibration automatique** des régions d'écran
- **Logging détaillé** pour le debugging

## 📁 Structure du Projet

```
pok/
├── main.py                 # Point d'entrée principal
├── modules/               # Modules principaux
│   ├── image_analysis.py  # Détection de cartes et OCR
│   ├── button_detector.py # Détection des boutons d'action
│   ├── screen_capture.py  # Capture d'écran et régions
│   ├── automation.py      # Contrôle souris/clavier
│   ├── game_state.py      # État du jeu
│   ├── spin_rush_strategy.py # Stratégie Spin & Rush
│   └── constants.py       # Constantes et configurations
├── templates/             # Templates d'images
│   ├── buttons/          # Boutons d'action
│   └── cards/            # Cartes de poker
├── tools/                # Outils de développement
│   └── calibration_tool.py # Calibration des régions
├── docs/                 # Documentation
├── tests/                # Tests unitaires
└── archive/              # Fichiers obsolètes
```

## 🛠️ Installation

1. **Cloner le repository**
   ```bash
   git clone <repository-url>
   cd pok
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurer Tesseract OCR**
   - Installer Tesseract OCR
   - Configurer le chemin dans `config.ini`

## 🎮 Utilisation

### Lancement rapide
```bash
python main.py
```

### Calibration des régions
```bash
python tools/calibration_tool.py
```

## ⚙️ Configuration

### Fichier config.ini
```ini
[Screen]
target_window = Betclic Poker
capture_interval = 0.01

[Tesseract]
tesseract_path = C:\Program Files\Tesseract-OCR\tesseract.exe

[Strategy]
default_strategy = spin_rush
aggression_level = high
```

### Régions calibrées (calibrated_regions.json)
```json
{
  "hand_area": {"x": 100, "y": 200, "width": 300, "height": 150},
  "fold_button": {"x": 800, "y": 1000, "width": 120, "height": 40},
  "call_button": {"x": 950, "y": 1000, "width": 120, "height": 40}
}
```

## 🧠 Stratégies

### Spin & Rush (Par défaut)
- **Ultra-agressive** pour tournois rapides
- **All-in** sur mains fortes
- **Bluff** fréquent
- **Timer urgent** < 15s = action immédiate

### Générale
- **Équilibre** entre agression et prudence
- **Calcul des pot odds**
- **Analyse des ranges** adverses

## 📊 Logging

L'agent génère des logs détaillés dans `poker_ai.log` :
- Détection de cartes
- Décisions prises
- Actions exécutées
- Erreurs et warnings

## 🔧 Développement

### Tests
```bash
python -m pytest tests/
```

### Debugging
```bash
python tools/calibration_tool.py  # Calibration
python test_fixes.py              # Tests de base
```

## 📈 Performance

- **Temps de réaction** : < 100ms
- **Précision détection** : > 90%
- **Stabilité** : 24/7
- **Mémoire** : < 100MB

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature
3. Commit les changements
4. Push vers la branche
5. Ouvrir une Pull Request

## 📄 Licence

MIT License - Voir LICENSE pour plus de détails

## 🆘 Support

Pour toute question ou problème :
1. Vérifier les logs dans `poker_ai.log`
2. Consulter la documentation dans `docs/`
3. Ouvrir une issue sur GitHub

---

**⚠️ Avertissement** : Cet outil est destiné à des fins éducatives uniquement. L'utilisation en jeu réel peut violer les conditions d'utilisation de Betclic Poker. 