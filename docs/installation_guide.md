# Guide d'Installation - Agent IA Poker

## Prérequis

### 1. Python 3.9+
Assurez-vous d'avoir Python 3.9 ou supérieur installé.

### 2. Tesseract OCR (Recommandé)

L'agent utilise Tesseract OCR pour lire les montants de jetons. Si Tesseract n'est pas installé, l'agent utilisera une estimation basée sur les couleurs.

#### Installation sur Windows :

1. **Téléchargement** :
   - Allez sur https://github.com/UB-Mannheim/tesseract/wiki
   - Téléchargez la version Windows (32-bit ou 64-bit selon votre système)

2. **Installation** :
   - Exécutez le fichier .exe téléchargé
   - **IMPORTANT** : Notez le chemin d'installation (par défaut : `C:\Program Files\Tesseract-OCR\`)

3. **Configuration du PATH** :
   - Ouvrez les Variables d'environnement système
   - Dans "Variables système", trouvez "Path" et cliquez "Modifier"
   - Ajoutez le chemin vers le dossier Tesseract (ex: `C:\Program Files\Tesseract-OCR\`)
   - Redémarrez votre terminal/PowerShell

4. **Vérification** :
   ```bash
   tesseract --version
   ```

#### Installation sur Linux (Ubuntu/Debian) :
```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-fra  # Pour le français
```

#### Installation sur macOS :
```bash
brew install tesseract
brew install tesseract-lang  # Pour les langues supplémentaires
```

### 3. Packages Python

Tous les packages Python nécessaires sont listés dans `requirements.txt` et seront installés automatiquement.

## Installation de l'Agent

### 1. Cloner le projet
```bash
git clone <repository-url>
cd pok
```

### 2. Créer un environnement virtuel (Recommandé)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Configuration
1. Copiez `config.ini.example` vers `config.ini` (si disponible)
2. Modifiez les paramètres selon vos besoins

## Vérification de l'Installation

### Test des modules
```bash
python -c "import cv2; import numpy; import pytesseract; print('Tous les modules sont installés')"
```

### Test de Tesseract
```bash
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

### Test de l'agent en mode simulation
```bash
python main.py --mode=simulation --debug
```

## Dépannage

### Erreur "tesseract is not installed"
1. Vérifiez que Tesseract est installé : `tesseract --version`
2. Vérifiez que le PATH est correctement configuré
3. Redémarrez votre terminal
4. Si le problème persiste, l'agent utilisera l'estimation par couleur

### Erreur OpenCV
1. Vérifiez que OpenCV est installé : `python -c "import cv2; print(cv2.__version__)"`
2. Réinstallez OpenCV si nécessaire : `pip install opencv-python`

### Erreur de permissions
- Sur Windows : Exécutez PowerShell en tant qu'administrateur
- Sur Linux/macOS : Utilisez `sudo` si nécessaire

## Support

Pour toute question ou problème :
1. Vérifiez les logs dans `poker_ai.log`
2. Consultez la documentation dans `docs/`
3. Testez en mode simulation d'abord 