"""
Constantes et énumérations centralisées pour l'agent IA Poker
"""

from enum import Enum
from typing import Dict, List

class Position(Enum):
    """Positions à la table de poker - Unifiées"""
    UTG = "under_the_gun"      # Under the Gun
    MP = "middle_position"      # Middle Position
    CO = "cutoff"              # Cutoff
    BTN = "button"             # Button/Dealer
    SB = "small_blind"         # Small Blind
    BB = "big_blind"           # Big Blind
    EARLY = "early"            # Position précoce (UTG, MP)
    MIDDLE = "middle"          # Position moyenne (CO)
    LATE = "late"              # Position tardive (BTN, SB, BB)
    BLINDS = "blinds"          # Blinds (SB, BB)

class Action(Enum):
    """Actions possibles au poker - Unifiées"""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"

class GamePhase(Enum):
    """Phases du jeu - Unifiées"""
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"

class HandRank(Enum):
    """Rangs des mains de poker"""
    HIGH_CARD = "high_card"
    PAIR = "pair"
    TWO_PAIR = "two_pair"
    THREE_OF_A_KIND = "three_of_a_kind"
    STRAIGHT = "straight"
    FLUSH = "flush"
    FULL_HOUSE = "full_house"
    FOUR_OF_A_KIND = "four_of_a_kind"
    STRAIGHT_FLUSH = "straight_flush"
    ROYAL_FLUSH = "royal_flush"

# Mapping des positions pour compatibilité
POSITION_MAPPING = {
    0: Position.EARLY,
    1: Position.EARLY,
    2: Position.MIDDLE,
    3: Position.MIDDLE,
    4: Position.LATE,
    5: Position.LATE,
    6: Position.LATE
}

# Valeurs des cartes
CARD_VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
}

# Couleurs des cartes
CARD_SUITS = ['♠', '♥', '♦', '♣']

# Rangs des cartes
CARD_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

# Configuration par défaut des régions de capture
DEFAULT_REGIONS = {
    'hand_area': {'x': 512, 'y': 861, 'width': 240, 'height': 130, 'name': 'Cartes du joueur'},
    'community_cards': {'x': 348, 'y': 597, 'width': 590, 'height': 170, 'name': 'Cartes communes'},
    'pot_area': {'x': 465, 'y': 510, 'width': 410, 'height': 70, 'name': 'Zone du pot'},
    'fold_button': {'x': 773, 'y': 1008, 'width': 120, 'height': 40, 'name': 'Bouton Fold'},
    'call_button': {'x': 937, 'y': 1010, 'width': 120, 'height': 40, 'name': 'Bouton Call'},
    'raise_button': {'x': 1105, 'y': 1006, 'width': 120, 'height': 40, 'name': 'Bouton Raise'},
    'check_button': {'x': 936, 'y': 1008, 'width': 120, 'height': 40, 'name': 'Bouton Check'},
    'all_in_button': {'x': 1267, 'y': 907, 'width': 120, 'height': 40, 'name': 'Bouton All-In'},
    'my_stack_area': {'x': 548, 'y': 1015, 'width': 200, 'height': 50, 'name': 'Votre stack'},
    'opponent1_stack_area': {'x': 35, 'y': 657, 'width': 150, 'height': 50, 'name': 'Stack adversaire 1'},
    'opponent2_stack_area': {'x': 1102, 'y': 662, 'width': 150, 'height': 40, 'name': 'Stack adversaire 2'},
    'my_current_bet': {'x': 525, 'y': 824, 'width': 200, 'height': 30, 'name': 'Votre mise actuelle'},
    'opponent1_current_bet': {'x': 210, 'y': 638, 'width': 110, 'height': 80, 'name': 'Mise adversaire 1'},
    'opponent2_current_bet': {'x': 961, 'y': 645, 'width': 110, 'height': 60, 'name': 'Mise adversaire 2'},
    'bet_slider': {'x': 747, 'y': 953, 'width': 360, 'height': 40, 'name': 'Slider de mise'},
    'bet_input': {'x': 1115, 'y': 952, 'width': 100, 'height': 25, 'name': 'Input de mise'},
    'new_hand_button': {'x': 599, 'y': 979, 'width': 100, 'height': 30, 'name': 'Bouton New Hand'},
    'resume_button': {'x': 600, 'y': 400, 'width': 120, 'height': 40, 'name': 'Bouton Reprendre'},
    'blinds_area': {'x': 1166, 'y': 326, 'width': 120, 'height': 30, 'name': 'Zone des blinds'},
    'blinds_timer': {'x': 1168, 'y': 299, 'width': 100, 'height': 20, 'name': 'Timer des blinds'},
    'my_dealer_button': {'x': 297, 'y': 843, 'width': 50, 'height': 50, 'name': 'Bouton dealer (vous)'},
    'opponent1_dealer_button': {'x': 210, 'y': 545, 'width': 50, 'height': 50, 'name': 'Bouton dealer (adv1)'},
    'opponent2_dealer_button': {'x': 1012, 'y': 547, 'width': 80, 'height': 50, 'name': 'Bouton dealer (adv2)'}
}

# Configuration par défaut
DEFAULT_CONFIG = {
    'Display': {
        'target_window_title': 'Betclic Poker',
        'capture_fps': '10',
        'debug_mode': 'false'
    },
    'AI': {
        'aggression_level': '0.7',
        'bluff_frequency': '0.15',
        'risk_tolerance': '0.8',
        'bankroll_management': 'true'
    },
    'Automation': {
        'click_randomization': '5',
        'move_speed_min': '0.1',
        'move_speed_max': '0.3',
        'human_delays': 'true'
    },
    'Safety': {
        'max_hands_per_hour': '180',
        'emergency_fold_key': 'F12',
        'auto_pause_on_detection': 'true'
    },
    'Tesseract': {
        'tesseract_path': 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    }
} 