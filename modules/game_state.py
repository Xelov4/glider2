"""
Module de détection d'état du jeu pour l'agent IA Poker
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging
from enum import Enum
from .image_analysis import Card

class Position(Enum):
    """Positions à la table de poker"""
    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"
    BLINDS = "blinds"

class Action(Enum):
    """Actions possibles au poker"""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"

@dataclass
class Player:
    """Représente un joueur à la table"""
    id: str
    name: str
    stack: int
    position: Position
    is_active: bool = True
    is_dealer: bool = False
    is_small_blind: bool = False
    is_big_blind: bool = False
    current_bet: int = 0
    last_action: Optional[Action] = None

@dataclass
class GameState:
    """État complet du jeu de poker"""
    my_cards: List[Card] = field(default_factory=list)
    community_cards: List[Card] = field(default_factory=list)
    pot_size: int = 0
    my_stack: int = 0
    players: List[Player] = field(default_factory=list)
    current_bet: int = 0
    my_position: Position = Position.MIDDLE
    available_actions: List[str] = field(default_factory=list)
    hand_history: List[Dict] = field(default_factory=list)
    street: str = "preflop"  # preflop, flop, turn, river
    is_my_turn: bool = False
    hand_number: int = 0
    small_blind: int = 0
    big_blind: int = 0
    min_raise: int = 0
    max_raise: int = 0
    blinds_timer: int = 300  # Temps restant avant augmentation des blinds (en secondes)
    my_current_bet: int = 0
    opponent1_current_bet: int = 0
    opponent2_current_bet: int = 0
    my_is_dealer: bool = False
    opponent1_is_dealer: bool = False
    opponent2_is_dealer: bool = False
    is_my_turn: bool = False  # Détection si c'est notre tour (boutons rouges)
    
    def __str__(self):
        return f"Hand #{self.hand_number} - {self.street} - Pot: {self.pot_size} - My stack: {self.my_stack}"
    
    def is_my_turn(self) -> bool:
        """Détermine si c'est notre tour de jouer"""
        return self.is_my_turn  # Utilise l'attribut existant
    
    def update(self, game_info: Dict):
        """
        Met à jour l'état du jeu avec de nouvelles informations
        """
        try:
            # Mise à jour des cartes
            if 'my_cards' in game_info:
                self.my_cards = game_info['my_cards']
            
            if 'community_cards' in game_info:
                self.community_cards = game_info['community_cards']
            
            # Mise à jour des stacks et mises
            if 'my_stack' in game_info:
                self.my_stack = game_info['my_stack']
            
            if 'pot_size' in game_info:
                self.pot_size = game_info['pot_size']
            
            if 'bet_to_call' in game_info:
                self.current_bet = game_info['bet_to_call']
            
            # Mise à jour des actions disponibles
            if 'available_actions' in game_info:
                self.available_actions = game_info['available_actions']
            
            # Mise à jour de la position
            if 'my_is_dealer' in game_info:
                self.my_is_dealer = game_info['my_is_dealer']
            
            if 'blinds_timer' in game_info:
                self.blinds_timer = game_info['blinds_timer']
            
            # Déterminer la street
            self.street = self.determine_street(self.community_cards)
            
            # Déterminer si c'est notre tour
            self.is_my_turn = len(self.available_actions) > 0
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Erreur mise à jour GameState: {e}")
    
    def determine_street(self, community_cards: List) -> str:
        """
        Détermine la street actuelle basée sur les cartes communes
        """
        num_cards = len(community_cards)
        
        if num_cards == 0:
            return "preflop"
        elif num_cards == 3:
            return "flop"
        elif num_cards == 4:
            return "turn"
        elif num_cards == 5:
            return "river"
        else:
            return "unknown"
    
    def calculate_min_raise(self) -> int:
        """
        Calcule le montant minimum de raise
        """
        try:
            if self.current_bet == 0:
                return self.big_blind
            else:
                return self.current_bet * 2
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Erreur calcul min raise: {e}")
            return 0
    
    def calculate_max_raise(self) -> int:
        """
        Calcule le montant maximum de raise (all-in)
        """
        return self.my_stack
    
    def get_hand_strength(self) -> float:
        """
        Calcule la force de la main actuelle (0-1)
        """
        try:
            if not self.my_cards:
                return 0.0
            
            # Logique simplifiée de calcul de force
            # En production, utiliserait le poker engine
            total_cards = len(self.my_cards) + len(self.community_cards)
            
            if total_cards < 2:
                return 0.0
            
            # Simulation basée sur le nombre de cartes
            strength = min(1.0, total_cards / 7.0)
            
            return strength
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Erreur calcul force main: {e}")
            return 0.0
    
    def get_pot_odds(self, bet_amount: int) -> float:
        """
        Calcule les pot odds
        """
        try:
            if bet_amount == 0:
                return 0.0
            
            total_pot = self.pot_size + bet_amount
            return bet_amount / total_pot
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Erreur calcul pot odds: {e}")
            return 0.0
    
    def is_valid_action(self, action: str, amount: int = 0) -> bool:
        """
        Vérifie si une action est valide dans l'état actuel
        """
        try:
            if action not in self.available_actions:
                return False
            
            if action in ['bet', 'raise']:
                min_raise = self.calculate_min_raise()
                max_raise = self.calculate_max_raise()
                if amount < min_raise or amount > max_raise:
                    return False
            
            if action == 'call':
                if amount != self.current_bet:
                    return False
            
            return True
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Erreur validation action: {e}")
            return False 