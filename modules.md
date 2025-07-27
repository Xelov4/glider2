# Modules Python pour Agent IA Poker

## Installation complète

### Commande d'installation globale
```bash
pip install opencv-python pillow numpy pytesseract scikit-image mss pygetwindow pyautogui tensorflow scikit-learn pandas pynput keyboard mouse colorama tqdm configparser logging psutil win32gui win32api win32con pywin32
```

---

## MODULES ESSENTIELS (Core)

### 1. **opencv-python** `4.8.1.78`
```bash
pip install opencv-python==4.8.1.78
```
**Rôle :** Vision par ordinateur principale
- **Fonctions clés :**
  - Template matching pour détecter les boutons
  - Détection de contours pour les cartes
  - Filtrage et traitement d'images
  - Reconnaissance de formes et couleurs
- **Usage :** `cv2.matchTemplate()`, `cv2.findContours()`, `cv2.cvtColor()`

### 2. **pillow** `10.0.1`
```bash
pip install pillow==10.0.1
```
**Rôle :** Manipulation d'images basique
- **Fonctions clés :**
  - Conversion de formats d'image
  - Redimensionnement et recadrage
  - Opérations basiques sur pixels
- **Usage :** `Image.open()`, `Image.save()`, `Image.crop()`

### 3. **numpy** `1.24.3`
```bash
pip install numpy==1.24.3
```
**Rôle :** Calculs numériques et arrays
- **Fonctions clés :**
  - Manipulation des matrices d'images
  - Calculs de probabilités poker
  - Opérations mathématiques rapides
- **Usage :** `np.array()`, `np.mean()`, `np.argmax()`

### 4. **mss** `9.0.1`
```bash
pip install mss==9.0.1
```
**Rôle :** Capture d'écran ultra-rapide
- **Fonctions clés :**
  - Screenshots haute performance (>30 FPS)
  - Capture de régions spécifiques
  - Minimale latence
- **Usage :** `mss.grab()` pour capturer zones définies

---

## MODULES DE RECONNAISSANCE

### 5. **pytesseract** `0.3.10`
```bash
pip install pytesseract==0.3.10
```
**Rôle :** OCR (Optical Character Recognition)
- **Fonctions clés :**
  - Lire montants de jetons ($1,250)
  - Reconnaître texte sur boutons
  - Extraire noms de joueurs
- **Usage :** `pytesseract.image_to_string()`
- **Prérequis :** Installer Tesseract-OCR séparément

### 6. **scikit-image** `0.21.0`
```bash
pip install scikit-image==0.21.0
```
**Rôle :** Traitement d'images avancé
- **Fonctions clés :**
  - Filtres de débruitage
  - Détection de bords améliorée
  - Segmentation d'images
- **Usage :** `skimage.filters`, `skimage.feature`

---

## MODULES D'AUTOMATISATION

### 7. **pyautogui** `0.9.54`
```bash
pip install pyautogui==0.9.54
```
**Rôle :** Automatisation GUI basique
- **Fonctions clés :**
  - Clics de souris simples
  - Saisie de texte
  - Mouvements de souris
- **Usage :** `pyautogui.click()`, `pyautogui.moveTo()`, `pyautogui.typewrite()`

### 8. **pynput** `1.7.6`
```bash
pip install pynput==1.7.6
```
**Rôle :** Contrôle avancé souris/clavier
- **Fonctions clés :**
  - Hooks système pour détecter activité utilisateur
  - Contrôle précis des mouvements
  - Simulation d'événements naturels
- **Usage :** `pynput.mouse.Listener()`, `pynput.keyboard.Controller()`

### 9. **keyboard** `0.13.5`
```bash
pip install keyboard==0.13.5
```
**Rôle :** Contrôle clavier spécialisé
- **Fonctions clés :**
  - Raccourcis clavier (F12 = pause d'urgence)
  - Hooks globaux
  - Simulation de frappes complexes
- **Usage :** `keyboard.add_hotkey()`, `keyboard.press()`

### 10. **mouse** `0.7.1`
```bash
pip install mouse==0.7.1
```
**Rôle :** Contrôle souris spécialisé
- **Fonctions clés :**
  - Glisser-déposer pour sliders de mise
  - Clics avec timing précis
  - Détection de mouvements
- **Usage :** `mouse.drag()`, `mouse.wheel()`

---

## MODULES IA/CALCULS

### 11. **scikit-learn** `1.3.0`
```bash
pip install scikit-learn==1.3.0
```
**Rôle :** Machine Learning pour adaptation
- **Fonctions clés :**
  - Classification des styles d'adversaires
  - Prédiction des actions
  - Clustering des situations de jeu
- **Usage :** `sklearn.cluster.KMeans`, `sklearn.ensemble.RandomForest`

### 12. **pandas** `2.0.3`
```bash
pip install pandas==2.0.3
```
**Rôle :** Gestion des données de jeu
- **Fonctions clés :**
  - Historique des mains jouées
  - Statistiques et analytics
  - Import/export de données
- **Usage :** `pd.DataFrame()`, `pd.read_csv()`, analyse de données

### 13. **tensorflow** `2.13.0` (Optionnel)
```bash
pip install tensorflow==2.13.0
```
**Rôle :** Deep Learning avancé
- **Fonctions clés :**
  - Réseaux de neurones pour reconnaissance de cartes
  - Apprentissage de stratégies complexes
  - Prédiction d'actions adverses
- **Usage :** Modèles CNN pour vision, RNN pour séquences
- **Note :** Lourd, utiliser seulement si IA avancée nécessaire

---

## MODULES WINDOWS SPÉCIFIQUES

### 14. **pygetwindow** `0.0.9`
```bash
pip install pygetwindow==0.0.9
```
**Rôle :** Gestion des fenêtres Windows
- **Fonctions clés :**
  - Trouver la fenêtre de poker
  - Gérer focus et position
  - Détecter minimisation/maximisation
- **Usage :** `pygetwindow.getWindowsWithTitle()`

### 15. **pywin32** `306`
```bash
pip install pywin32==306
```
**Rôle :** APIs Windows avancées
- **Fonctions clés :**
  - Accès direct aux handles de fenêtres
  - Capture d'écran de fenêtres en arrière-plan
  - Contrôle système bas niveau
- **Usage :** `win32gui.FindWindow()`, `win32api.GetSystemMetrics()`

### 16. **psutil** `5.9.5`
```bash
pip install psutil==5.9.5
```
**Rôle :** Monitoring système
- **Fonctions clés :**
  - Surveiller CPU/RAM de l'agent
  - Détecter si processus poker est actif
  - Optimisation des performances
- **Usage :** `psutil.cpu_percent()`, `psutil.Process()`

---

## MODULES UTILITAIRES

### 17. **colorama** `0.4.6`
```bash
pip install colorama==0.4.6
```
**Rôle :** Interface terminal colorée
- **Fonctions clés :**
  - Logs colorés (rouge=erreur, vert=succès)
  - Debug visuel amélioré
  - Interface CLI attractive
- **Usage :** `colorama.Fore.RED`, `colorama.Style.BRIGHT`

### 18. **tqdm** `4.66.1`
```bash
pip install tqdm==4.66.1
```
**Rôle :** Barres de progression
- **Fonctions clés :**
  - Progression de l'entraînement IA
  - Loading des templates/données
  - Feedback utilisateur
- **Usage :** `tqdm(range(100))` pour boucles avec progression

### 19. **configparser** `5.3.0`
```bash
pip install configparser==5.3.0
```
**Rôle :** Gestion de configuration
- **Fonctions clés :**
  - Fichiers .ini pour paramètres
  - Configuration par environnement
  - Settings modifiables sans code
- **Usage :** Lecture/écriture config.ini

---

## MODULES POKER SPÉCIALISÉS

### 20. **python-poker** (Custom/GitHub)
```bash
pip install git+https://github.com/worldveil/deuces.git
```
**Rôle :** Évaluation de mains poker
- **Fonctions clés :**
  - Ranking des mains (straight, flush, etc.)
  - Comparaison de forces
  - Calculs d'equity
- **Usage :** Évaluation rapide des combinaisons

---

## INSTALLATION PAR CATÉGORIES

### Installation Core (Minimum viable)
```bash
pip install opencv-python pillow numpy mss pyautogui pytesseract
```

### Installation Complète (Recommandée)
```bash
pip install opencv-python pillow numpy pytesseract scikit-image mss pygetwindow pyautogui scikit-learn pandas pynput keyboard mouse colorama tqdm configparser psutil pywin32
```

### Installation IA Avancée (+ Deep Learning)
```bash
pip install tensorflow torch torchvision  # Ajouter aux précédents
```

---

## CONFIGURATION POST-INSTALLATION

### 1. Tesseract OCR (Obligatoire pour pytesseract)
```bash
# Télécharger depuis: https://github.com/UB-Mannheim/tesseract/wiki
# Installer dans C:\Program Files\Tesseract-OCR\
# Ajouter au PATH Windows
```

### 2. Configuration pytesseract
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### 3. Test d'installation
```python
# Fichier test_modules.py
try:
    import cv2, numpy, PIL, pytesseract, mss, pyautogui
    print("✅ Modules essentiels installés")
except ImportError as e:
    print(f"❌ Module manquant: {e}")
```

---

## RÉSUMÉ DES RÔLES

| **Catégorie** | **Modules** | **Rôle Principal** |
|---------------|-------------|-------------------|
| **Vision** | opencv-python, pillow, scikit-image | Reconnaissance cartes/boutons |
| **Capture** | mss, pygetwindow, pywin32 | Screenshots haute performance |
| **OCR** | pytesseract | Lecture montants/texte |
| **Automatisation** | pyautogui, pynput, keyboard, mouse | Contrôle souris/clavier |
| **IA** | scikit-learn, pandas, tensorflow | Décisions et apprentissage |
| **Système** | psutil, pywin32 | Integration Windows |
| **Utils** | colorama, tqdm, configparser | Interface et configuration |
| **Poker** | deuces (custom) | Logique poker spécialisée |

**Total :** ~20 modules pour un agent complet et robuste.