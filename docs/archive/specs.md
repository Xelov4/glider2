# Spécifications complètes - Agent IA Poker

## Description du projet

### Vue d'ensemble
Développer un agent IA autonome capable de jouer au poker en ligne en analysant l'écran Windows 11 en temps réel et en automatisant les interactions via souris et clavier. L'agent doit être capable de reconnaître l'état du jeu, calculer les probabilités, prendre des décisions stratégiques optimales et exécuter les actions correspondantes.

### Objectifs principaux
- **Reconnaissance visuelle** : Identifier cartes, jetons, boutons et état du jeu
- **Intelligence artificielle** : Calculer les probabilités et stratégies optimales
- **Automatisation** : Contrôler souris/clavier pour interagir avec l'interface
- **Furtivité** : Éviter la détection par les systèmes anti-bot
- **Performance** : Fonctionner en temps réel avec latence minimale

## Architecture système

### Composants principaux

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Screen Capture │───▶│ Image Analysis  │───▶│  Game State     │
│     Module      │    │     Module      │    │   Detector      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Automation    │◀───│   AI Decision   │◀───│   Poker Engine  │
│     Module      │    │     Module      │    │     Module      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Flux de données
1. **Capture** : Screenshot de la fenêtre de poker
2. **Analyse** : Extraction des éléments visuels (cartes, jetons, boutons)
3. **État** : Construction de l'état actuel du jeu
4. **Décision** : Calcul de l'action optimale via l'IA
5. **Exécution** : Automatisation de l'action (clic, frappe)

## Spécifications techniques

### Environnement de développement
- **OS** : Windows 11
- **Langage principal** : Python 3.9+
- **IDE recommandé** : VS Code ou PyCharm
- **Gestionnaire de paquets** : pip + requirements.txt

### Stack technologique

#### Vision par ordinateur
```python
# Bibliothèques principales
opencv-python==4.8.1.78         # Traitement d'images
pillow==10.0.1                   # Manipulation d'images
numpy==1.24.3                    # Calculs numériques
pytesseract==0.3.10             # OCR
scikit-image==0.21.0            # Traitement d'images avancé
```

#### Capture d'écran
```python
mss==9.0.1                      # Capture ultra-rapide
pygetwindow==0.0.9              # Gestion des fenêtres
pyautogui==0.9.54               # Automatisation GUI
```

#### Intelligence artificielle
```python
tensorflow==2.13.0              # Deep learning (optionnel)
scikit-learn==1.3.0            # Machine learning classique
pandas==2.0.3                   # Manipulation de données
```

#### Automatisation et contrôle
```python
pynput==1.7.6                   # Contrôle souris/clavier avancé
keyboard==0.13.5                # Hooks clavier
mouse==0.7.1                    # Contrôle souris précis
```

#### Utilitaires
```python
colorama==0.4.6                 # Couleurs terminal
tqdm==4.66.1                    # Barres de progression
configparser==5.3.0            # Fichiers de configuration
logging==0.4.9.6               # Système de logs
```

## Modules détaillés

### 1. Module de capture d'écran (`screen_capture.py`)

#### Fonctionnalités
- Détection automatique de la fenêtre de poker
- Capture ciblée de zones spécifiques
- Optimisation des performances (ROI uniquement)
- Gestion multi-moniteurs

#### Spécifications techniques
```python
class ScreenCapture:
    def __init__(self):
        self.target_window = None
        self.capture_regions = {}
        
    def find_poker_window(self) -> bool:
        """Trouve la fenêtre de poker active"""
        
    def capture_region(self, region_name: str) -> np.ndarray:
        """Capture une région spécifique (cartes, boutons, etc.)"""
        
    def capture_full_table(self) -> np.ndarray:
        """Capture complète de la table de poker"""
        
    def get_capture_fps(self) -> float:
        """Retourne le FPS de capture actuel"""
```

#### Régions de capture définies
- **Hand area** : Zone des cartes du joueur
- **Community cards** : Cartes communes
- **Pot area** : Zone des jetons au centre
- **Action buttons** : Boutons fold/call/raise
- **Player info** : Stack des joueurs
- **Chat area** : Zone de chat (pour détecter les timeouts)

### 2. Module d'analyse d'images (`image_analysis.py`)

#### Fonctionnalités
- Reconnaissance de cartes (valeur + couleur)
- Détection des jetons et montants
- Identification des boutons disponibles
- OCR pour lire les montants textuels

#### Spécifications techniques
```python
class ImageAnalyzer:
    def __init__(self):
        self.card_templates = self.load_card_templates()
        self.ocr_config = r'--oem 3 --psm 6 outputbase digits'
        
    def detect_cards(self, image: np.ndarray) -> List[Card]:
        """Détecte les cartes dans l'image"""
        
    def detect_chips(self, image: np.ndarray) -> Dict[str, int]:
        """Détecte les montants de jetons"""
        
    def detect_buttons(self, image: np.ndarray) -> List[str]:
        """Détecte les boutons d'action disponibles"""
        
    def read_text_amount(self, image: np.ndarray) -> int:
        """Lit un montant textuel via OCR"""
```

#### Algorithmes de reconnaissance
- **Template matching** pour les cartes standard
- **Contour detection** pour les jetons
- **Color space analysis** (HSV) pour améliorer la précision
- **Morphological operations** pour nettoyer les images

### 3. Module détecteur d'état (`game_state.py`)

#### Fonctionnalités
- Construction de l'état complet du jeu
- Historique des actions
- Tracking des adversaires
- Calcul de position relative

#### Spécifications techniques
```python
@dataclass
class GameState:
    my_cards: List[Card]
    community_cards: List[Card]
    pot_size: int
    my_stack: int
    players: List[Player]
    current_bet: int
    my_position: str
    available_actions: List[str]
    hand_history: List[Action]
    
class GameStateDetector:
    def __init__(self):
        self.previous_state = None
        self.hand_number = 0
        
    def build_state(self, analyzed_image: Dict) -> GameState:
        """Construit l'état du jeu à partir de l'analyse d'image"""
        
    def detect_state_changes(self, current_state: GameState) -> List[str]:
        """Détecte les changements depuis le dernier état"""
        
    def is_my_turn(self, state: GameState) -> bool:
        """Détermine si c'est notre tour de jouer"""
```

### 4. Module moteur de poker (`poker_engine.py`)

#### Fonctionnalités
- Évaluation des mains
- Calcul des probabilités (outs, odds)
- Simulation Monte Carlo
- Base de données de stratégies pré-calculées

#### Spécifications techniques
```python
class PokerEngine:
    def __init__(self):
        self.hand_evaluator = HandEvaluator()
        self.preflop_charts = self.load_preflop_charts()
        
    def evaluate_hand_strength(self, cards: List[Card], 
                             community: List[Card]) -> float:
        """Évalue la force de la main (0-1)"""
        
    def calculate_pot_odds(self, pot_size: int, bet_size: int) -> float:
        """Calcule les pot odds"""
        
    def simulate_outcomes(self, state: GameState, 
                         num_simulations: int = 1000) -> Dict:
        """Simulation Monte Carlo des résultats possibles"""
        
    def get_preflop_action(self, cards: List[Card], 
                          position: str, num_players: int) -> str:
        """Retourne l'action pré-flop optimale"""
```

#### Tables de stratégie intégrées
- **Preflop charts** : Ranges d'ouverture par position
- **Post-flop decision trees** : Arbres de décision
- **Bet sizing guidelines** : Tailles de mises optimales
- **ICM calculations** : Pour les tournois

### 5. Module de décision IA (`ai_decision.py`)

#### Fonctionnalités
- Prise de décision basée sur la théorie des jeux
- Adaptation dynamique aux adversaires
- Gestion des émotions simulées (tilt prevention)
- Optimisation bankroll

#### Spécifications techniques
```python
class AIDecisionMaker:
    def __init__(self):
        self.poker_engine = PokerEngine()
        self.opponent_models = {}
        self.risk_tolerance = 0.8
        
    def make_decision(self, state: GameState) -> Decision:
        """Prend une décision basée sur l'état du jeu"""
        
    def update_opponent_model(self, opponent_id: str, 
                            action: Action, state: GameState):
        """Met à jour le modèle d'un adversaire"""
        
    def calculate_ev(self, action: str, state: GameState) -> float:
        """Calcule l'espérance de gain d'une action"""
        
    def should_bluff(self, state: GameState) -> bool:
        """Détermine s'il faut bluffer"""
```

#### Algorithmes de décision
- **Game Theory Optimal (GTO)** : Stratégies équilibrées
- **Exploitative play** : Adaptation aux faiblesses adverses
- **Bayesian opponent modeling** : Modélisation probabiliste
- **Regret minimization** : Apprentissage des erreurs

### 6. Module d'automatisation (`automation.py`)

#### Fonctionnalités
- Contrôle précis souris/clavier
- Randomisation des mouvements (anti-détection)
- Gestion des timeouts
- Actions d'urgence (déconnexion rapide)

#### Spécifications techniques
```python
class AutomationController:
    def __init__(self):
        self.click_randomization = 5  # pixels
        self.move_speed_range = (0.1, 0.3)  # secondes
        self.human_delays = True
        
    def click_button(self, button_name: str, coordinates: Tuple[int, int]):
        """Clique sur un bouton avec mouvement naturel"""
        
    def drag_bet_slider(self, start_pos: Tuple[int, int], 
                       end_pos: Tuple[int, int]):
        """Glisse le slider de mise"""
        
    def type_bet_amount(self, amount: int):
        """Tape un montant de mise"""
        
    def emergency_fold(self):
        """Fold d'urgence en cas de problème"""
```

#### Anti-détection
- **Mouvements courbes** : Trajectoires non-linéaires
- **Délais variables** : Timing humain simulé
- **Micro-mouvements** : Petits ajustements naturels  
- **Pattern avoidance** : Éviter les séquences répétitives

## Configuration et paramètres

### Fichier de configuration (`config.ini`)
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

### Structure des fichiers
```
poker_ai_agent/
├── main.py                    # Point d'entrée
├── config.ini                 # Configuration
├── requirements.txt           # Dépendances
├── modules/
│   ├── __init__.py
│   ├── screen_capture.py      # Capture d'écran
│   ├── image_analysis.py      # Analyse d'images
│   ├── game_state.py          # État du jeu
│   ├── poker_engine.py        # Moteur de poker
│   ├── ai_decision.py         # Décisions IA
│   └── automation.py          # Automatisation
├── data/
│   ├── card_templates/        # Templates de cartes
│   ├── strategy_charts/       # Tables de stratégie
│   └── logs/                  # Fichiers de log
├── tests/
│   ├── test_capture.py        # Tests unitaires
│   ├── test_analysis.py
│   └── test_poker_engine.py
└── docs/
    ├── api_reference.md       # Documentation API
    └── user_guide.md          # Guide utilisateur
```

## Interface utilisateur

### GUI principale (optionnel)
- **Status panel** : État actuel (connecté, mains jouées, profit)
- **Debug window** : Affichage des captures et analyses
- **Controls** : Start/Stop/Pause
- **Settings** : Configuration en temps réel
- **Statistics** : Win rate, profit/perte, graphiques

### CLI interface
```bash
python main.py --mode=live --table=6max --stakes=nl10
python main.py --mode=simulation --hands=1000
python main.py --config=aggressive.ini --debug=true
```

## Fonctionnalités avancées

### 1. Multi-tabling
- Support pour plusieurs tables simultanées
- Priorisation des décisions urgentes
- Load balancing automatique

### 2. Apprentissage adaptatif
- Base de données des mains jouées
- Analyse post-session automatique
- Amélioration continue des modèles

### 3. Détection d'anomalies
- Reconnaissance des situations inhabituelles
- Mode sécurisé automatique
- Alertes et notifications

### 4. Backtesting
- Simulation sur historiques réels
- Tests de stratégies alternatives
- Optimisation des paramètres

## Sécurité et anti-détection

### Mesures de protection
- **Randomisation complète** : Tous les timings et mouvements
- **Profils comportementaux** : Simulation de styles de jeu humains
- **Break scheduling** : Pauses automatiques réalistes
- **Error simulation** : Erreurs humaines occasionnelles

### Monitoring
- Détection des captchas
- Reconnaissance des messages anti-bot
- Surveillance des patterns de jeu suspects

## Tests et validation

### Tests unitaires requis
- Test de chaque module individuellement
- Validation des algorithmes de poker
- Vérification des calculs de probabilités
- Tests de performance (FPS, latence)

### Tests d'intégration
- Test complet du workflow
- Validation sur différentes interfaces
- Tests de stress (sessions longues)
- Vérification anti-détection

### Métriques de performance
- **Précision reconnaissance** : >95% pour cartes/jetons
- **Latence décision** : <500ms en moyenne
- **Uptime** : >99% sur sessions 4h+
- **Win rate** : Positif sur 10k+ mains

## Déploiement et maintenance

### Installation
```bash
git clone [repository]
cd poker_ai_agent
pip install -r requirements.txt
python setup.py install
```

### Configuration initiale
1. Calibration écran et fenêtre cible
2. Tests de reconnaissance sur captures statiques  
3. Validation de l'automatisation (mode sandbox)
4. Configuration des paramètres de sécurité

### Monitoring en production
- Logs détaillés de toutes les actions
- Métriques de performance en temps réel
- Alertes automatiques en cas d'anomalie
- Sauvegarde régulière des données

## Considérations légales et éthiques

### Avertissements importants
⚠️ **Vérifier la légalité dans votre juridiction**
⚠️ **Respecter les ToS des plateformes de poker**
⚠️ **Usage uniquement sur sites autorisant les bots**
⚠️ **Responsabilité utilisateur pour conformité légale**

### Usage recommandé
- Tests et développement uniquement
- Plateformes explicitement autorisées
- Jeu responsable et limits appropriées
- Respect des autres joueurs

---

**Note finale** : Ce document constitue une spécification technique complète. L'implémentation doit respecter toutes les lois locales et conditions d'utilisation des plateformes ciblées.