"""
Module de stratégie de jeu unifié pour l'agent IA Poker
"""

import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod

class Position(Enum):
    UTG = "under_the_gun"
    MP = "middle_position" 
    CO = "cutoff"
    BTN = "button"
    SB = "small_blind"
    BB = "big_blind"

class GamePhase(Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"

@dataclass
class HandRange:
    """Définit une range de mains (ex: AA-22, AKs-A2s)"""
    pairs: List[str]  # ['AA', 'KK', 'QQ', ...]
    suited: List[str]  # ['AKs', 'AQs', ...]
    offsuit: List[str]  # ['AKo', 'AQo', ...]

class Strategy(ABC):
    """Interface commune pour toutes les stratégies"""
    
    @abstractmethod
    def should_play_hand(self, cards: List[str], position: Position, 
                        action_before: str, num_players: int) -> bool:
        """Détermine si on doit jouer cette main pré-flop"""
        pass
    
    @abstractmethod
    def get_action_decision(self, game_state: 'GameState') -> str:
        """Décision principale basée sur l'état du jeu"""
        pass
    
    @abstractmethod
    def calculate_bet_size(self, action: str, game_state: 'GameState') -> int:
        """Calcule la taille de mise optimale"""
        pass

class GeneralStrategy(Strategy):
    """
    Stratégie générale pour poker standard
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Charger les charts de stratégie pré-calculés
        self.preflop_ranges = self.load_preflop_ranges()
        self.postflop_strategies = self.load_postflop_strategies()
        self.bet_sizing_rules = self.load_bet_sizing_rules()
        
    def load_preflop_ranges(self) -> Dict:
        """Charge les ranges de mains pré-flop par position"""
        return {
            Position.UTG: HandRange(
                pairs=['AA', 'KK', 'QQ', 'JJ', 'TT', '99'],
                suited=['AKs', 'AQs', 'AJs', 'KQs'],
                offsuit=['AKo', 'AQo']
            ),
            Position.MP: HandRange(
                pairs=['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77'],
                suited=['AKs', 'AQs', 'AJs', 'ATs', 'KQs', 'KJs'],
                offsuit=['AKo', 'AQo', 'AJo', 'KQo']
            ),
            Position.CO: HandRange(
                pairs=['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55'],
                suited=['AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'KQs', 'KJs', 'KTs', 'QJs'],
                offsuit=['AKo', 'AQo', 'AJo', 'ATo', 'KQo', 'KJo']
            ),
            Position.BTN: HandRange(
                pairs=['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22'],
                suited=['AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
                       'KQs', 'KJs', 'KTs', 'K9s', 'QJs', 'QTs', 'Q9s', 'JTs', 'J9s', 'T9s', '98s', '87s'],
                offsuit=['AKo', 'AQo', 'AJo', 'ATo', 'A9o', 'KQo', 'KJo', 'KTo', 'QJo', 'QTo', 'JTo']
            ),
            Position.SB: HandRange(
                pairs=['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22'],
                suited=['AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
                       'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'QJs', 'QTs', 'Q9s', 'JTs', 'J9s', 'T9s', '98s', '87s', '76s'],
                offsuit=['AKo', 'AQo', 'AJo', 'ATo', 'A9o', 'A8o', 'KQo', 'KJo', 'KTo', 'QJo', 'QTo', 'JTo']
            ),
            Position.BB: HandRange(
                pairs=['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22'],
                suited=['AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
                       'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'K7s', 'QJs', 'QTs', 'Q9s', 'Q8s', 'JTs', 'J9s', 'J8s', 'T9s', 'T8s', '98s', '87s', '76s', '65s'],
                offsuit=['AKo', 'AQo', 'AJo', 'ATo', 'A9o', 'A8o', 'A7o', 'KQo', 'KJo', 'KTo', 'K9o', 'QJo', 'QTo', 'Q9o', 'JTo', 'J9o', 'T9o']
            )
        }
    
    def load_postflop_strategies(self) -> Dict:
        """Charge les stratégies post-flop"""
        return {
            'value_betting': {
                'strong_hands': ['straight_flush', 'four_of_a_kind', 'full_house', 'flush', 'straight'],
                'medium_hands': ['three_of_a_kind', 'two_pair', 'one_pair'],
                'weak_hands': ['high_card']
            },
            'bluffing': {
                'board_texture': ['dry', 'wet', 'coordinated'],
                'position_advantage': True,
                'stack_depth': 'deep'
            }
        }
    
    def load_bet_sizing_rules(self) -> Dict:
        """Charge les règles de sizing des mises"""
        return {
            'preflop': {
                'open_raise': 2.5,  # BB
                '3bet': 3.0,
                '4bet': 2.5,
                'all_in': 1.0
            },
            'postflop': {
                'value_bet': 0.75,  # % du pot
                'bluff_bet': 0.6,
                'cbet': 0.65,
                'all_in': 1.0
            }
        }
    
    def should_play_hand(self, cards: List[str], position: Position, 
                        action_before: str, num_players: int) -> bool:
        """Détermine si on doit jouer cette main pré-flop"""
        try:
            hand_str = self.cards_to_string(cards)
            range_for_position = self.preflop_ranges[position]
            
            # Vérifier si la main est dans notre range
            if hand_str in range_for_position.pairs:
                return True
            if hand_str in range_for_position.suited:
                return True
            if hand_str in range_for_position.offsuit:
                return True
                
            # Ajustements selon l'action avant nous
            if action_before == "raise":
                # Range plus serrée contre une relance
                return self.is_in_tight_range(hand_str)
            elif action_before == "3bet":
                # Range encore plus serrée contre un 3-bet
                return self.is_in_ultra_tight_range(hand_str)
                
            return False
            
        except Exception as e:
            self.logger.error(f"Erreur évaluation main: {e}")
            return False
    
    def cards_to_string(self, cards: List[str]) -> str:
        """Convertit les cartes en notation poker standard"""
        if len(cards) != 2:
            return ""
            
        card1, card2 = cards
        rank1, suit1 = card1[0], card1[1]
        rank2, suit2 = card2[0], card2[1]
        
        # Déterminer si suited ou offsuit
        if suit1 == suit2:
            return f"{rank1}{rank2}s"  # suited
        else:
            return f"{rank1}{rank2}o"  # offsuit
    
    def is_in_tight_range(self, hand_str: str) -> bool:
        """Vérifie si la main est dans une range serrée"""
        tight_hands = ['AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo']
        return hand_str in tight_hands
    
    def is_in_ultra_tight_range(self, hand_str: str) -> bool:
        """Vérifie si la main est dans une range ultra-serrée"""
        ultra_tight_hands = ['AA', 'KK', 'QQ', 'AKs', 'AKo']
        return hand_str in ultra_tight_hands
    
    def get_action_decision(self, game_state: 'GameState') -> str:
        """Décision principale basée sur l'état du jeu"""
        try:
            if game_state.phase == GamePhase.PREFLOP:
                return self.preflop_decision(game_state)
            else:
                return self.postflop_decision(game_state)
                
        except Exception as e:
            self.logger.error(f"Erreur décision: {e}")
            return "fold"
    
    def preflop_decision(self, game_state: 'GameState') -> str:
        """Logique de décision pré-flop"""
        try:
            should_play = self.should_play_hand(
                game_state.my_cards, 
                game_state.my_position,
                game_state.action_before_us,
                game_state.num_players
            )
            
            if not should_play:
                return "fold"
                
            # Si on joue, déterminer l'action
            if game_state.action_before_us == "check":
                return "raise"  # Open raise
            elif game_state.action_before_us == "call":
                return "raise"  # Iso-raise
            elif game_state.action_before_us == "raise":
                if self.should_3bet(game_state.my_cards, game_state.raiser_position):
                    return "raise"  # 3-bet
                else:
                    return "call"
            else:
                return "call"
                
        except Exception as e:
            self.logger.error(f"Erreur décision pré-flop: {e}")
            return "fold"
    
    def postflop_decision(self, game_state: 'GameState') -> str:
        """Logique de décision post-flop"""
        try:
            hand_strength = self.evaluate_hand_strength(
                game_state.my_cards, 
                game_state.community_cards
            )
            
            # Calculs des equity et odds
            pot_odds = game_state.bet_to_call / (game_state.pot_size + game_state.bet_to_call)
            equity = self.calculate_equity(game_state.my_cards, game_state.community_cards)
            
            # Décision basée sur les maths
            if equity > pot_odds:
                if hand_strength > 0.8:  # Très forte main
                    return "raise"
                else:
                    return "call"
            else:
                # Considérer un bluff
                if self.should_bluff(game_state):
                    return "raise"
                else:
                    return "fold"
                    
        except Exception as e:
            self.logger.error(f"Erreur décision post-flop: {e}")
            return "fold"
    
    def evaluate_hand_strength(self, my_cards: List[str], community_cards: List[str]) -> float:
        """Évalue la force de la main (0-1)"""
        try:
            # Simulation d'évaluation de main
            # En production, utiliser un moteur d'évaluation poker
            all_cards = my_cards + community_cards
            
            # Logique simplifiée d'évaluation
            if len(all_cards) >= 5:
                # Évaluation post-flop
                return random.uniform(0.3, 0.9)
            else:
                # Évaluation pré-flop
                return random.uniform(0.2, 0.8)
                
        except Exception as e:
            self.logger.error(f"Erreur évaluation main: {e}")
            return 0.5
    
    def calculate_equity(self, my_cards: List[str], community_cards: List[str]) -> float:
        """Calcule l'equity de la main"""
        try:
            # Simulation de calcul d'equity
            # En production, utiliser un moteur d'equity
            return random.uniform(0.2, 0.8)
            
        except Exception as e:
            self.logger.error(f"Erreur calcul equity: {e}")
            return 0.5
    
    def should_bluff(self, game_state: 'GameState') -> bool:
        """Détermine si on doit bluffer"""
        try:
            # Facteurs pour le bluff
            position_good = game_state.my_position in [Position.BTN, Position.CO]
            board_dry = len(game_state.community_cards) < 3
            stack_deep = game_state.my_stack > game_state.pot_size * 3
            
            return position_good and board_dry and stack_deep
            
        except Exception as e:
            self.logger.error(f"Erreur évaluation bluff: {e}")
            return False
    
    def should_3bet(self, cards: List[str], raiser_position: Position) -> bool:
        """Détermine si on doit 3-bet"""
        try:
            hand_str = self.cards_to_string(cards)
            strong_hands = ['AA', 'KK', 'QQ', 'AKs', 'AKo']
            return hand_str in strong_hands
            
        except Exception as e:
            self.logger.error(f"Erreur évaluation 3-bet: {e}")
            return False
    
    def calculate_bet_size(self, action: str, game_state: 'GameState') -> int:
        """Calcule la taille de mise optimale"""
        try:
            if action == "all_in":
                return game_state.my_stack
                
            pot_size = game_state.pot_size
            
            if action == "raise" or action == "bet":
                if game_state.phase == GamePhase.PREFLOP:
                    # Raise pré-flop : 2.5-3x la grosse blinde
                    return min(int(pot_size * 0.75), game_state.my_stack)
                else:
                    # Post-flop : 60-75% du pot
                    return min(int(pot_size * 0.65), game_state.my_stack)
            
            return 0  # Pour call/check/fold
            
        except Exception as e:
            self.logger.error(f"Erreur calcul taille mise: {e}")
            return 0 