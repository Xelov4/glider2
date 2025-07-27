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
    'hand_area': {'x': 400, 'y': 500, 'width': 200, 'height': 100, 'name': 'Cartes du joueur'},
    'community_cards': {'x': 300, 'y': 300, 'width': 400, 'height': 80, 'name': 'Cartes communes'},
    'pot_area': {'x': 350, 'y': 250, 'width': 300, 'height': 50, 'name': 'Zone du pot'},
    'action_buttons': {'x': 350, 'y': 600, 'width': 300, 'height': 60, 'name': 'Boutons d\'action'},
    'my_stack_area': {'x': 50, 'y': 100, 'width': 150, 'height': 400, 'name': 'Votre stack'},
    'opponent1_stack_area': {'x': 200, 'y': 100, 'width': 150, 'height': 400, 'name': 'Stack adversaire 1'},
    'opponent2_stack_area': {'x': 350, 'y': 100, 'width': 150, 'height': 400, 'name': 'Stack adversaire 2'},
    'current_bet_to_call': {'x': 400, 'y': 350, 'width': 200, 'height': 30, 'name': 'Mise à payer'},
    'my_current_bet': {'x': 400, 'y': 350, 'width': 200, 'height': 30, 'name': 'Votre mise actuelle'},
    'opponent1_current_bet': {'x': 200, 'y': 350, 'width': 200, 'height': 30, 'name': 'Mise adversaire 1'},
    'opponent2_current_bet': {'x': 600, 'y': 350, 'width': 200, 'height': 30, 'name': 'Mise adversaire 2'},
    'bet_slider': {'x': 400, 'y': 550, 'width': 200, 'height': 20, 'name': 'Slider de mise'},
    'bet_input': {'x': 400, 'y': 580, 'width': 100, 'height': 25, 'name': 'Input de mise'},
    'new_hand_button': {'x': 500, 'y': 700, 'width': 100, 'height': 30, 'name': 'Bouton New Hand'},
    'sit_out_button': {'x': 600, 'y': 700, 'width': 100, 'height': 30, 'name': 'Bouton Sit Out'},
    'leave_table_button': {'x': 800, 'y': 700, 'width': 100, 'height': 30, 'name': 'Leave Table'},
    'blinds_area': {'x': 200, 'y': 200, 'width': 150, 'height': 40, 'name': 'Zone des blinds'},
    'blinds_timer': {'x': 200, 'y': 250, 'width': 100, 'height': 30, 'name': 'Timer des blinds'},
    'my_dealer_button': {'x': 400, 'y': 800, 'width': 50, 'height': 50, 'name': 'Bouton dealer (vous)'},
    'opponent1_dealer_button': {'x': 200, 'y': 300, 'width': 50, 'height': 50, 'name': 'Bouton dealer (adv1)'},
    'opponent2_dealer_button': {'x': 600, 'y': 300, 'width': 50, 'height': 50, 'name': 'Bouton dealer (adv2)'}
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