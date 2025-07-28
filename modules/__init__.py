"""
Modules de l'agent IA Poker - Version unifiée
"""

# Imports des modules principaux
from .screen_capture import ScreenCapture, ScreenRegion
from .image_analysis import ImageAnalyzer
from .game_state import GameState
from .ai_decision import AIDecisionMaker
from .automation import AutomationEngine
from .button_detector import ButtonDetector, UIButton, ActionType

# Imports des constantes et énumérations
from .constants import (
    Position, Action, GamePhase, HandRank,
    POSITION_MAPPING, CARD_VALUES, CARD_SUITS, CARD_RANKS,
    DEFAULT_REGIONS, DEFAULT_CONFIG
)

# Version du système
__version__ = "2.0.0"
__author__ = "Poker AI Team"

# Exports publics
__all__ = [
    # Classes principales
    'ScreenCapture',
    'ScreenRegion', 
    'ImageAnalyzer',
    'GameState',
    'AIDecisionMaker',
    'AutomationEngine',
    'ButtonDetector',
    'UIButton',
    'ActionType',
    
    # Énumérations
    'Position',
    'Action', 
    'GamePhase',
    'HandRank',
    
    # Constantes
    'POSITION_MAPPING',
    'CARD_VALUES',
    'CARD_SUITS',
    'CARD_RANKS',
    'DEFAULT_REGIONS',
    'DEFAULT_CONFIG',
    
    # Métadonnées
    '__version__',
    '__author__'
] 