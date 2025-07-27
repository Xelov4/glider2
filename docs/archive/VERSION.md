# Version 0.1 - Agent IA Poker

## 🎯 Fonctionnalités

### ✅ Fonctionnalités implémentées
- **Capture d'écran** : Capture automatique des régions de poker
- **Analyse d'images** : Détection de cartes, boutons, stacks
- **Détection d'état** : Analyse de l'état du jeu en temps réel
- **Prise de décision** : IA pour décider des actions (fold, call, raise, etc.)
- **Automatisation** : Clics automatiques sur les boutons
- **Stratégies** : 
  - Stratégie Spin & Rush (agressive)
  - Stratégie générale (équilibrée)
- **Sécurité** : Limites de mains/heure, pause d'urgence
- **Logging** : Logs détaillés des actions

### 🔧 Corrections appliquées
- ✅ Erreur `'bool' object is not callable` dans main.py
- ✅ Erreurs OpenCV dans button_detector.py et image_analysis.py
- ✅ Régions d'écran pour ultra-wide 5120x1440
- ✅ Validation automatique des coordonnées
- ✅ Compatibilité Windows (commande `py`)

## 📋 Prérequis

### Système
- Windows 10/11
- Python 3.8+
- Écran ultra-wide supporté (5120x1440)

### Dépendances
```bash
pip install -r requirements.txt
```

### Configuration
- Tesseract OCR installé
- Régions calibrées dans `calibrated_regions.json`
- Configuration dans `config.ini`

## 🚀 Utilisation

```bash
# Lancer l'agent
py main.py

# Tester les corrections
py test_fixes.py

# Outil de calibration
py tools/calibration_tool.py
```

## 📁 Structure

```
pok/
├── main.py                 # Point d'entrée principal
├── config.ini             # Configuration
├── requirements.txt       # Dépendances
├── calibrated_regions.json # Régions calibrées
├── modules/              # Modules principaux
│   ├── screen_capture.py
│   ├── image_analysis.py
│   ├── game_state.py
│   ├── button_detector.py
│   ├── ai_decision.py
│   ├── automation.py
│   ├── strategy_engine.py
│   ├── spin_rush_strategy.py
│   └── constants.py
├── tools/                # Outils utilitaires
│   └── calibration_tool.py
├── tests/               # Tests unitaires
└── docs/               # Documentation
```

## 🎮 Configuration

### Stratégie Spin & Rush (par défaut)
- Agression élevée (0.9)
- Bluff fréquent (25%)
- Tolérance au risque élevée (95%)
- Pas de gestion de bankroll
- Pression temporelle (1.5x)

### Sécurité
- Max 180 mains/heure
- Touche d'urgence F12
- Pause automatique sur détection

## 📊 Monitoring

- Logs détaillés dans `poker_ai.log`
- Statistiques de session
- Métriques de performance
- Historique des mains

## 🔄 Prochaines versions

### Version 0.2 (planifiée)
- [ ] Amélioration de la détection de cartes
- [ ] Plus de stratégies
- [ ] Interface graphique
- [ ] Analyse de performance avancée

### Version 0.3 (planifiée)
- [ ] Machine Learning pour décisions
- [ ] Support multi-tables
- [ ] API REST
- [ ] Dashboard web

## 📝 Notes de développement

- Code modulaire et extensible
- Logging complet pour debug
- Gestion d'erreurs robuste
- Compatibilité multi-résolutions
- Validation automatique des données

## 🐛 Bugs connus

- Templates de boutons simulés (nécessite vraies images)
- Détection de cartes basique (amélioration prévue)
- Calibration manuelle requise

## 📞 Support

Pour les problèmes ou questions :
1. Vérifier les logs dans `poker_ai.log`
2. Tester avec `py test_fixes.py`
3. Recalibrer si nécessaire

---
**Version 0.1** - Agent IA Poker fonctionnel avec corrections appliquées 