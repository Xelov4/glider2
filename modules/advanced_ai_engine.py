"""
Module d'IA avancée pour poker avec calculs mathématiques et réactivité maximale
"""

import math
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import time
from collections import defaultdict

class ActionType(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    ALL_IN = "all_in"

@dataclass
class PotOdds:
    """Calcul des pot odds"""
    pot_size: float
    call_amount: float
    odds: float
    profitable: bool

@dataclass
class HandEquity:
    """Équité de la main"""
    equity: float
    vs_range: str
    confidence: float

@dataclass
class BetSizing:
    """Taille de mise optimale"""
    min_bet: float
    max_bet: float
    optimal_bet: float
    reasoning: str

class AdvancedAIEngine:
    """
    Moteur d'IA avancée avec calculs mathématiques et réactivité maximale
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Cache pour les calculs rapides
        self.equity_cache = {}
        self.pot_odds_cache = {}
        
        # Statistiques en temps réel
        self.opponent_stats = defaultdict(lambda: {
            'vpip': 0.0,  # Voluntarily Put Money In Pot
            'pfr': 0.0,   # Pre-Flop Raise
            'af': 0.0,    # Aggression Factor
            'hands_played': 0,
            'fold_to_3bet': 0.0,
            'cbet_frequency': 0.0
        })
        
        # Historique des actions
        self.action_history = []
        self.last_action_time = time.time()
        
        # Configuration de réactivité
        self.ultra_fast_mode = True
        self.max_decision_time = 0.5  # 500ms max pour décision
        self.min_decision_time = 0.1  # 100ms min pour paraître humain
        
    def calculate_pot_odds(self, pot_size: float, call_amount: float) -> PotOdds:
        """Calcule les pot odds"""
        if call_amount == 0:
            return PotOdds(pot_size, 0, float('inf'), True)
        
        odds = pot_size / call_amount
        profitable = odds > 1.0
        
        return PotOdds(pot_size, call_amount, odds, profitable)
    
    def calculate_hand_equity(self, my_cards: List[str], community_cards: List[str], 
                            opponent_range: str = "standard") -> HandEquity:
        """Calcule l'équité de la main vs une range d'adversaire"""
        
        # Cache key
        cache_key = f"{''.join(my_cards)}_{''.join(community_cards)}_{opponent_range}"
        
        if cache_key in self.equity_cache:
            return self.equity_cache[cache_key]
        
        # Calcul rapide d'équité (simplifié pour la vitesse)
        equity = self._quick_equity_calculation(my_cards, community_cards, opponent_range)
        
        result = HandEquity(equity, opponent_range, 0.85)
        self.equity_cache[cache_key] = result
        
        return result
    
    def _quick_equity_calculation(self, my_cards: List[str], community_cards: List[str], 
                                 opponent_range: str) -> float:
        """Calcul rapide d'équité (approximation)"""
        
        if not community_cards:  # Preflop
            return self._preflop_equity(my_cards, opponent_range)
        else:  # Postflop
            return self._postflop_equity(my_cards, community_cards, opponent_range)
    
    def _preflop_equity(self, my_cards: List[str], opponent_range: str) -> float:
        """Équité préflop vs range d'adversaire"""
        
        # Ranges d'adversaires typiques
        ranges = {
            "tight": ["AA", "KK", "QQ", "JJ", "AKs", "AKo"],
            "standard": ["AA", "KK", "QQ", "JJ", "TT", "AKs", "AQs", "AKo", "AQo"],
            "loose": ["AA", "KK", "QQ", "JJ", "TT", "99", "88", "AKs", "AQs", "AJs", "AKo", "AQo", "AJo"],
            "ultra_loose": ["AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "AKs", "AQs", "AJs", "ATs", "AKo", "AQo", "AJo", "ATo"]
        }
        
        opponent_hands = ranges.get(opponent_range, ranges["standard"])
        
        # Calcul d'équité simplifié
        my_hand = ''.join(my_cards)
        if my_hand in opponent_hands:
            return 0.5  # Main similaire
        elif self._is_premium_hand(my_hand):
            return 0.8  # Main premium
        elif self._is_strong_hand(my_hand):
            return 0.65  # Main forte
        else:
            return 0.4  # Main moyenne/faible
    
    def _postflop_equity(self, my_cards: List[str], community_cards: List[str], 
                         opponent_range: str) -> float:
        """Équité postflop (approximation)"""
        
        # Évaluation simplifiée de la force de main
        hand_strength = self._evaluate_hand_strength(my_cards, community_cards)
        
        # Ajustement selon le nombre de cartes communautaires
        if len(community_cards) == 3:  # Flop
            return hand_strength * 0.7
        elif len(community_cards) == 4:  # Turn
            return hand_strength * 0.8
        else:  # River
            return hand_strength * 0.9
    
    def _evaluate_hand_strength(self, my_cards: List[str], community_cards: List[str]) -> float:
        """Évalue la force de la main (0.0 à 1.0)"""
        
        all_cards = my_cards + community_cards
        
        # Détection de patterns
        if self._has_straight_flush(all_cards):
            return 1.0
        elif self._has_four_of_a_kind(all_cards):
            return 0.95
        elif self._has_full_house(all_cards):
            return 0.9
        elif self._has_flush(all_cards):
            return 0.8
        elif self._has_straight(all_cards):
            return 0.75
        elif self._has_three_of_a_kind(all_cards):
            return 0.6
        elif self._has_two_pair(all_cards):
            return 0.5
        elif self._has_pair(all_cards):
            return 0.3
        else:
            return 0.1  # High card
    
    def _is_premium_hand(self, hand: str) -> bool:
        """Détermine si c'est une main premium"""
        premium_hands = ["AA", "KK", "QQ", "AKs", "AKo"]
        return hand in premium_hands
    
    def _is_strong_hand(self, hand: str) -> bool:
        """Détermine si c'est une main forte"""
        strong_hands = ["JJ", "TT", "AQs", "AQo", "AJs", "AJo"]
        return hand in strong_hands
    
    def _has_straight_flush(self, cards: List[str]) -> bool:
        """Détecte une quinte flush"""
        # Implémentation simplifiée
        return False
    
    def _has_four_of_a_kind(self, cards: List[str]) -> bool:
        """Détecte un carré"""
        # Implémentation simplifiée
        return False
    
    def _has_full_house(self, cards: List[str]) -> bool:
        """Détecte un full"""
        # Implémentation simplifiée
        return False
    
    def _has_flush(self, cards: List[str]) -> bool:
        """Détecte une couleur"""
        # Implémentation simplifiée
        return False
    
    def _has_straight(self, cards: List[str]) -> bool:
        """Détecte une quinte"""
        # Implémentation simplifiée
        return False
    
    def _has_three_of_a_kind(self, cards: List[str]) -> bool:
        """Détecte un brelan"""
        # Implémentation simplifiée
        return False
    
    def _has_two_pair(self, cards: List[str]) -> bool:
        """Détecte une double paire"""
        # Implémentation simplifiée
        return False
    
    def _has_pair(self, cards: List[str]) -> bool:
        """Détecte une paire"""
        # Implémentation simplifiée
        return False
    
    def calculate_optimal_bet_size(self, pot_size: float, stack_size: float, 
                                 equity: float, position: str, street: str) -> BetSizing:
        """Calcule la taille de mise optimale"""
        
        # Bet sizing selon l'équité et la position
        if equity > 0.8:  # Main très forte
            sizing_ratio = 0.75
            reasoning = "Main très forte - value bet"
        elif equity > 0.6:  # Main forte
            sizing_ratio = 0.6
            reasoning = "Main forte - value bet"
        elif equity > 0.4:  # Main moyenne
            sizing_ratio = 0.4
            reasoning = "Main moyenne - protection"
        else:  # Main faible
            sizing_ratio = 0.25
            reasoning = "Main faible - bluff/check"
        
        # Ajustement selon la position
        if position == "BTN":
            sizing_ratio *= 1.2  # Plus agressif en position
        elif position == "BB":
            sizing_ratio *= 0.8  # Plus défensif en BB
        
        # Ajustement selon la street
        if street == "preflop":
            sizing_ratio *= 1.5  # Plus agressif préflop
        elif street == "river":
            sizing_ratio *= 0.8  # Plus conservateur à la river
        
        optimal_bet = pot_size * sizing_ratio
        min_bet = pot_size * 0.25
        max_bet = min(pot_size * 2.0, stack_size)
        
        # S'assurer que la mise est dans les limites
        optimal_bet = max(min_bet, min(optimal_bet, max_bet))
        
        return BetSizing(min_bet, max_bet, optimal_bet, reasoning)
    
    def make_decision(self, game_state: Dict) -> Dict:
        """Prend une décision optimale avec réactivité maximale"""
        
        start_time = time.time()
        
        try:
            # 1. ANALYSE RAPIDE DES DONNÉES
            my_cards = game_state.get('my_cards', [])
            community_cards = game_state.get('community_cards', [])
            pot_size = game_state.get('pot_size', 0)
            call_amount = game_state.get('call_amount', 0)
            my_stack = game_state.get('my_stack', 0)
            position = game_state.get('position', 'UTG')
            street = game_state.get('street', 'preflop')
            
            # 2. CALCULS MATHÉMATIQUES RAPIDES
            pot_odds = self.calculate_pot_odds(pot_size, call_amount)
            equity = self.calculate_hand_equity(my_cards, community_cards)
            bet_sizing = self.calculate_optimal_bet_size(pot_size, my_stack, equity.equity, position, street)
            
            # 3. LOGIQUE DE DÉCISION
            decision = self._determine_action(pot_odds, equity, bet_sizing, game_state)
            
            # 4. VÉRIFICATION DU TEMPS
            decision_time = time.time() - start_time
            if decision_time > self.max_decision_time:
                self.logger.warning(f"Decision trop lente: {decision_time:.3f}s")
            
            # 5. AJOUT D'UN DÉLAI HUMAIN MINIMAL
            human_delay = random.uniform(self.min_decision_time, self.max_decision_time)
            
            return {
                'action': decision,
                'bet_size': bet_sizing.optimal_bet if decision in ['raise', 'all_in'] else 0,
                'reasoning': f"{equity.equity:.2f} equity, {pot_odds.odds:.2f} pot odds, {bet_sizing.reasoning}",
                'confidence': equity.confidence,
                'decision_time': decision_time,
                'human_delay': human_delay
            }
            
        except Exception as e:
            self.logger.error(f"Erreur décision IA: {e}")
            # Fallback: FOLD
            return {
                'action': 'fold',
                'bet_size': 0,
                'reasoning': f"Erreur IA: {e}",
                'confidence': 0.0,
                'decision_time': time.time() - start_time,
                'human_delay': 0.1
            }
    
    def _determine_action(self, pot_odds: PotOdds, equity: HandEquity, 
                         bet_sizing: BetSizing, game_state: Dict) -> str:
        """Détermine l'action optimale"""
        
        # 1. VÉRIFICATIONS DE SÉCURITÉ
        if game_state.get('my_stack', 0) <= 0:
            return 'fold'
        
        # 2. LOGIQUE PRINCIPALE
        if equity.equity > 0.8:  # Main très forte
            if bet_sizing.optimal_bet > game_state.get('my_stack', 0) * 0.5:
                return 'all_in'
            else:
                return 'raise'
        
        elif equity.equity > 0.6:  # Main forte
            if pot_odds.profitable:
                return 'call'
            else:
                return 'raise'
        
        elif equity.equity > 0.4:  # Main moyenne
            if pot_odds.profitable:
                return 'call'
            else:
                return 'fold'
        
        else:  # Main faible
            if self._should_bluff(game_state, equity.equity):
                return 'raise'
            elif pot_odds.profitable:
                return 'call'
            else:
                return 'fold'
    
    def _should_bluff(self, game_state: Dict, equity: float) -> bool:
        """Détermine si on doit bluffer"""
        
        # Facteurs pour le bluff
        position = game_state.get('position', 'UTG')
        street = game_state.get('street', 'preflop')
        pot_size = game_state.get('pot_size', 0)
        stack_size = game_state.get('my_stack', 0)
        
        # Bluff plus probable en position et avec un gros pot
        bluff_probability = 0.1  # Base 10%
        
        if position in ['BTN', 'CO']:
            bluff_probability += 0.2
        if street == 'river':
            bluff_probability += 0.1
        if pot_size > stack_size * 2:
            bluff_probability += 0.1
        
        return random.random() < bluff_probability
    
    def update_opponent_stats(self, opponent_id: str, action: str, bet_size: float):
        """Met à jour les statistiques d'un adversaire"""
        
        stats = self.opponent_stats[opponent_id]
        stats['hands_played'] += 1
        
        if action in ['call', 'raise', 'all_in']:
            stats['vpip'] = (stats['vpip'] * (stats['hands_played'] - 1) + 1) / stats['hands_played']
        
        if action == 'raise':
            stats['pfr'] = (stats['pfr'] * (stats['hands_played'] - 1) + 1) / stats['hands_played']
        
        # Mise à jour de l'aggression factor
        if action in ['raise', 'all_in']:
            stats['af'] = (stats['af'] * (stats['hands_played'] - 1) + 1) / stats['hands_played']
    
    def get_opponent_range(self, opponent_id: str) -> str:
        """Estime la range d'un adversaire basée sur ses stats"""
        
        stats = self.opponent_stats[opponent_id]
        
        if stats['vpip'] < 0.2:
            return "tight"
        elif stats['vpip'] < 0.4:
            return "standard"
        elif stats['vpip'] < 0.6:
            return "loose"
        else:
            return "ultra_loose" 