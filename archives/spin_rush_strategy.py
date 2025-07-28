"""
Stratégie spécifique pour Spin & Rush - Betclic Poker
Format 3 joueurs, stack 500, hyperturbo 60s
"""

from typing import Dict, List, Tuple
from enum import Enum
import logging
from .strategy_engine import Strategy, Position, GamePhase

class SpinRushStrategy(Strategy):
    """
    Stratégie ultra-agressive pour Spin & Rush
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Ranges par position (3 joueurs)
        self.position_ranges = {
            'UTG': {  # Premier
                'pairs': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88'],
                'suited': ['AKs', 'AQs', 'AJs', 'ATs', 'KQs', 'KJs'],
                'offsuit': ['AKo', 'AQo', 'AJo']
            },
            'BTN': {  # Dealer - Ultra agressif
                'pairs': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22'],
                'suited': ['AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
                          'KQs', 'KJs', 'KTs', 'K9s', 'QJs', 'QTs', 'Q9s', 'JTs', 'J9s', 'T9s'],
                'offsuit': ['AKo', 'AQo', 'AJo', 'ATo', 'A9o', 'KQo', 'KJo', 'KTo', 'QJo', 'QTo']
            },
            'BB': {  # Big Blind - Agressif défensif
                'pairs': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66'],
                'suited': ['AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'KQs', 'KJs', 'KTs', 'QJs', 'QTs'],
                'offsuit': ['AKo', 'AQo', 'AJo', 'ATo', 'KQo', 'KJo']
            }
        }
        
        # Bet sizing selon le timer
        self.timer_bet_sizing = {
            'normal': {'preflop': 3, 'postflop': 0.75},  # Timer > 30s
            'pressure': {'preflop': 4, 'postflop': 0.9},  # Timer 15-30s
            'urgent': {'preflop': 5, 'postflop': 1.0}    # Timer < 15s
        }
    
    def get_position(self, game_state) -> str:
        """Détermine la position actuelle"""
        # Utiliser directement la position si disponible
        if hasattr(game_state, 'my_position') and game_state.my_position:
            return game_state.my_position
        
        # Fallback sur la logique dealer
        if hasattr(game_state, 'my_is_dealer') and game_state.my_is_dealer:
            return 'BTN'
        elif hasattr(game_state, 'opponent1_is_dealer') and game_state.opponent1_is_dealer:
            return 'BB'
        else:
            return 'UTG'
    
    def should_play_hand(self, cards: List[str], position: Position, 
                        action_before: str, num_players: int) -> bool:
        """Décide si on doit jouer la main selon la position et le timer"""
        position_str = self.position_to_string(position)
        position_range = self.position_ranges[position_str]
        
        # Timer pressure - plus agressif si timer court
        timer = getattr(game_state, 'blinds_timer', 300)
        if timer < 15:
            # En mode urgent, jouer presque tout
            return True
        elif timer < 30:
            # En mode pressure, élargir les ranges
            return self._hand_in_range(cards, position_range, expand=True)
        else:
            # Mode normal
            return self._hand_in_range(cards, position_range)
    
    def position_to_string(self, position: Position) -> str:
        """Convertit Position enum en string pour Spin & Rush"""
        if position == Position.BTN:
            return 'BTN'
        elif position == Position.BB:
            return 'BB'
        else:
            return 'UTG'
    
    def get_action_decision(self, game_state: 'GameState') -> str:
        """Prend la décision finale d'action"""
        position = self.get_position(game_state)
        timer = getattr(game_state, 'blinds_timer', 300)
        
        # Logique de décision ultra-agressive
        if game_state.my_stack <= game_state.big_blind * 15:
            return 'all_in'
        
        if position == 'BTN' and game_state.street == 'preflop':
            return 'raise'  # Steal blinds BTN
        
        if timer < 15:
            return 'all_in'  # Timer urgent - all-in
        
        # Décision normale basée sur la main
        action_before = getattr(game_state, 'action_before_us', 'fold')
        if self.should_play_hand(game_state.my_cards, game_state.my_position, 
                               action_before, game_state.num_players):
            if self.should_bluff(game_state, timer):
                return 'raise'  # Bluff agressif
            else:
                return 'raise'  # Value bet
        else:
            return 'fold'  # Main faible
    
    def calculate_bet_size(self, action: str, game_state: 'GameState') -> int:
        """Calcule la taille de mise optimale pour Spin & Rush"""
        timer = getattr(game_state, 'blinds_timer', 300)
        timer_mode = self._get_timer_mode(timer)
        sizing = self.timer_bet_sizing[timer_mode]
        
        if action == 'raise':
            if game_state.street == 'preflop':
                return game_state.big_blind * sizing['preflop']
            else:
                return int(game_state.pot_size * sizing['postflop'])
        elif action == 'all_in':
            return game_state.my_stack
        
        return 0
    
    def should_bluff(self, game_state, timer: int) -> bool:
        """Décide si on doit bluffer"""
        base_bluff_freq = 0.25
        
        # Plus de bluffs si timer court
        if timer < 15:
            return True  # Bluff fréquent en mode urgent
        elif timer < 30:
            return base_bluff_freq * 1.5  # Plus de bluffs sous pression
        
        return base_bluff_freq
    
    def should_steal_blinds(self, game_state, position: str) -> bool:
        """Décide si on doit voler les blinds"""
        if position == 'BTN' and game_state.street == 'preflop':
            return True  # Toujours voler en position BTN
        return False
    
    def _hand_in_range(self, cards: List[str], range_dict: Dict, expand: bool = False) -> bool:
        """Vérifie si la main est dans la range"""
        if len(cards) != 2:
            return False
            
        # Convertir les cartes en notation
        card1, card2 = cards[0], cards[1]
        rank1, suit1 = card1[0], card1[1]
        rank2, suit2 = card2[0], card2[1]
        
        # Déterminer si suited ou offsuit
        suited = suit1 == suit2
        hand_str = f"{rank1}{rank2}{'s' if suited else 'o'}"
        
        # Vérifier dans la range
        if hand_str in range_dict['pairs']:
            return True
        if hand_str in range_dict['suited']:
            return True
        if hand_str in range_dict['offsuit']:
            return True
        
        # En mode expand, accepter plus de mains
        if expand:
            return True  # Accepter presque tout en mode urgent
        
        return False
    
    def _get_timer_mode(self, timer: int) -> str:
        """Détermine le mode selon le timer"""
        if timer < 15:
            return 'urgent'
        elif timer < 30:
            return 'pressure'
        else:
            return 'normal' 