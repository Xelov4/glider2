# Historique de Conversation - Agent IA Poker

## Résumé de la Session
**Date :** 27 juillet 2025  
**Projet :** Agent IA Poker pour Betclic Poker  
**Objectif :** Déboguer et faire fonctionner un agent poker automatique utilisant OCR et template matching

## Problèmes Identifiés et Résolus

### 1. **Erreurs de Base (Résolues)**
- **Erreur PowerShell :** `&&` non reconnu → Utilisation de `py main.py`
- **Python introuvable :** → Utilisation de `py` au lieu de `python`
- **Erreur bool :** `bool object is not callable` → Correction `is_my_turn()` → `is_my_turn`

### 2. **Erreurs OpenCV (Résolues)**
- **Assertion failed :** Templates trop grands pour les images capturées
- **Solution :** Réduction taille templates simulés + vérifications de taille
- **Fichiers modifiés :** `modules/button_detector.py`, `modules/image_analysis.py`

### 3. **Configuration Poker Client (Résolue)**
- **Problème :** Agent configuré pour "PokerStars" mais client réel = "Betclic Poker"
- **Solution :** Remplacement de toutes les références "PokerStars" → "Betclic Poker"
- **Fichiers modifiés :** `config.ini`, `modules/constants.py`, `modules/screen_capture.py`

### 4. **Problème de Régions d'Écran (Résolu)**
- **Contexte :** Écran ultra-wide 5120x1440
- **Problème :** Coordonnées calibrées dépassaient la largeur d'écran
- **Solution :** Ajustement manuel des coordonnées dans `calibrated_regions.json`
- **Résultat :** Recalibration complète avec coordonnées "normales" (gauche de l'écran)

### 5. **Agent "Ne Fait Rien" (Résolu)**
- **Problème :** Agent ne détectait pas de jeu actif
- **Solution :** Ajout de logique proactive `_try_start_new_hand()`
- **Fonctionnalités ajoutées :**
  - Détection automatique si aucune partie en cours
  - Clic automatique sur "New Hand" avec 3 stratégies de fallback
  - Logging détaillé pour debug

### 6. **Templates Simulés (Résolu)**
- **Problème principal :** Templates générés aléatoirement au lieu d'images réelles
- **Solution :** Modification du code pour charger les vrais templates
- **Templates fournis par l'utilisateur :**
  ```
  templates/
  ├── buttons/
  │   ├── fold_button.png
  │   ├── call_button.png (cann_button.png)
  │   ├── check_button.png
  │   ├── raise_button.png
  │   ├── all_in_button.png
  │   ├── bet_button.png
  │   └── new_hand_button.png
  └── cards/
      ├── ranks/
      │   ├── card_2.png
      │   ├── card_9.png
      │   ├── card_10.png
      │   ├── card_K.png
      │   └── card_A.png
      └── suits/
          ├── suit_spades.png
          ├── suit_hearts.png
          ├── suit_diamonds.png
          └── suit_clubs.png
  ```

## État Actuel du Projet

### ✅ **Fonctionnel :**
- Chargement des templates réels (6 boutons + 5 rangs + 4 couleurs)
- Détection de la fenêtre Betclic Poker
- Capture des régions calibrées
- Logique proactive pour démarrer de nouvelles parties
- Structure de code robuste avec logging détaillé

### ⚠️ **Limitations actuelles :**
- Templates de cartes incomplets (seulement 5 rangs sur 13)
- Fallbacks pour les cartes manquantes
- Encodage Unicode pour les emojis dans les logs

### 🎯 **Prochaines étapes recommandées :**
1. Ajouter les templates de cartes manquants (3,4,5,6,7,8,J,Q)
2. Tester l'agent en conditions réelles
3. Optimiser les seuils de détection si nécessaire

## Fichiers Créés/Modifiés

### **Fichiers principaux modifiés :**
- `main.py` - Logique proactive et logging
- `modules/button_detector.py` - Chargement vrais templates
- `modules/image_analysis.py` - Chargement vrais templates cartes
- `config.ini` - Configuration Betclic Poker
- `modules/constants.py` - Mise à jour configuration
- `modules/screen_capture.py` - Target window
- `modules/automation.py` - Méthode click_at_position

### **Fichiers de test créés :**
- `test_templates.py` - Test chargement templates
- `test_betclick.py` - Test spécifique Betclic Poker
- `test_game_simulation.py` - Test simulation jeu
- `test_new_hand_detection.py` - Test détection New Hand

### **Documentation créée :**
- `README.md` - Documentation complète
- `CORRECTIONS_APPLIQUEES.md` - Historique des corrections
- `.gitignore` - Configuration Git
- `VERSION.md` - Versioning

## Commandes Utilisées

```bash
# Tests de base
py test_templates.py
py test_betclick.py
py main.py

# Calibration
py tools/calibration_tool.py

# Debug
py debug_windows.py
```

## Points Clés Techniques

### **Template Matching :**
- Technique OpenCV pour détecter des motifs dans les images
- Comparaison pixel par pixel entre template et image capturée
- Seuil de confiance pour valider les détections

### **OCR (Tesseract) :**
- Extraction de texte depuis les images
- Configuration spéciale pour les chiffres
- Utilisé pour détecter les montants et textes

### **Screen Capture :**
- Capture de régions spécifiques de l'écran
- Gestion des écrans multi-résolution
- Validation des coordonnées

### **Automation :**
- Clics avec délais humains
- Mouvements de souris courbes
- Randomisation pour éviter la détection

## Leçons Apprises

1. **Importance des templates réels** - Les templates simulés ne fonctionnent pas
2. **Calibration précise** - Essentielle pour la détection
3. **Logging détaillé** - Crucial pour le debug
4. **Gestion des erreurs** - Vérifications de taille et validité
5. **Configuration adaptée** - Chaque client poker a ses spécificités

## État Final

L'agent est maintenant **fonctionnel** avec :
- ✅ Templates réels chargés
- ✅ Configuration Betclic Poker
- ✅ Régions calibrées
- ✅ Logique proactive
- ✅ Logging détaillé

**Prêt pour les tests en conditions réelles !** 🚀 