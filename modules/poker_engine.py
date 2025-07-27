"""
Module moteur de poker pour l'agent IA Poker
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from .image_analysis import Card
from .game_state import GameState

class HandEvaluator:
    """
    Évaluateur de mains de poker
    """
    
    def __init__(self):
        self.rank_values = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        self.hand_rankings = {
            'high_card': 1,
            'pair': 2,
            'two_pair': 3,
            'three_of_a_kind': 4,
            'straight': 5,
            'flush': 6,
            'full_house': 7,
            'four_of_a_kind': 8,
            'straight_flush': 9,
            'royal_flush': 10
        }
    
    def evaluate_hand(self, cards: List[Card]) -> Tuple[str, int]:
        """
        Évalue une main de poker et retourne (type_de_main, valeur)
        """
        if len(cards) < 5:
            return 'high_card', 0
        
        # Trier les cartes par valeur
        sorted_cards = sorted(cards, key=lambda c: self.rank_values[c.rank], reverse=True)
        
        # Vérifier les différents types de mains
        if self.is_royal_flush(sorted_cards):
            return 'royal_flush', 9000
        elif self.is_straight_flush(sorted_cards):
            return 'straight_flush', 8000 + self.rank_values[sorted_cards[0].rank]
        elif self.is_four_of_a_kind(sorted_cards):
            return 'four_of_a_kind', 7000 + self.rank_values[sorted_cards[0].rank]
        elif self.is_full_house(sorted_cards):
            return 'full_house', 6000 + self.rank_values[sorted_cards[0].rank]
        elif self.is_flush(sorted_cards):
            return 'flush', 5000 + self.rank_values[sorted_cards[0].rank]
        elif self.is_straight(sorted_cards):
            return 'straight', 4000 + self.rank_values[sorted_cards[0].rank]
        elif self.is_three_of_a_kind(sorted_cards):
            return 'three_of_a_kind', 3000 + self.rank_values[sorted_cards[0].rank]
        elif self.is_two_pair(sorted_cards):
            return 'two_pair', 2000 + self.rank_values[sorted_cards[0].rank]
        elif self.is_pair(sorted_cards):
            return 'pair', 1000 + self.rank_values[sorted_cards[0].rank]
        else:
            return 'high_card', self.rank_values[sorted_cards[0].rank]
    
    def is_royal_flush(self, cards: List[Card]) -> bool:
        """Vérifie si c'est une quinte flush royale"""
        return self.is_straight_flush(cards) and self.rank_values[cards[0].rank] == 14
    
    def is_straight_flush(self, cards: List[Card]) -> bool:
        """Vérifie si c'est une quinte flush"""
        return self.is_straight(cards) and self.is_flush(cards)
    
    def is_four_of_a_kind(self, cards: List[Card]) -> bool:
        """Vérifie si c'est un carré"""
        ranks = [card.rank for card in cards]
        for rank in set(ranks):
            if ranks.count(rank) == 4:
                return True
        return False
    
    def is_full_house(self, cards: List[Card]) -> bool:
        """Vérifie si c'est un full"""
        ranks = [card.rank for card in cards]
        rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
        return 3 in rank_counts.values() and 2 in rank_counts.values()
    
    def is_flush(self, cards: List[Card]) -> bool:
        """Vérifie si c'est une couleur"""
        suits = [card.suit for card in cards]
        return len(set(suits)) == 1
    
    def is_straight(self, cards: List[Card]) -> bool:
        """Vérifie si c'est une quinte"""
        values = [self.rank_values[card.rank] for card in cards]
        values.sort()
        for i in range(len(values) - 4):
            if values[i+4] - values[i] == 4:
                return True
        return False
    
    def is_three_of_a_kind(self, cards: List[Card]) -> bool:
        """Vérifie si c'est un brelan"""
        ranks = [card.rank for card in cards]
        for rank in set(ranks):
            if ranks.count(rank) == 3:
                return True
        return False
    
    def is_two_pair(self, cards: List[Card]) -> bool:
        """Vérifie si c'est une double paire"""
        ranks = [card.rank for card in cards]
        pairs = 0
        for rank in set(ranks):
            if ranks.count(rank) == 2:
                pairs += 1
        return pairs == 2
    
    def is_pair(self, cards: List[Card]) -> bool:
        """Vérifie si c'est une paire"""
        ranks = [card.rank for card in cards]
        for rank in set(ranks):
            if ranks.count(rank) == 2:
                return True
        return False

class PokerEngine:
    """
    Moteur de poker avec évaluation de mains et calculs de probabilités
    """
    
    def __init__(self):
        self.hand_evaluator = HandEvaluator()
        self.preflop_charts = self.load_preflop_charts()
        self.logger = logging.getLogger(__name__)
        
        # Probabilités pré-calculées
        self.probability_cache = {}
        
    def load_preflop_charts(self) -> Dict:
        """
        Charge les charts pré-flop (simulation)
        """
        charts = {}
        
        # Charts simplifiées - en production ce serait plus complexe
        premium_hands = ['AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo', 'AQs']
        strong_hands = ['TT', '99', '88', 'AQo', 'AJs', 'AJo', 'KQs']
        medium_hands = ['77', '66', '55', 'ATs', 'KQo', 'KJs', 'QJs']
        
        for hand in premium_hands:
            charts[hand] = {'action': 'raise', 'frequency': 0.9}
        for hand in strong_hands:
            charts[hand] = {'action': 'raise', 'frequency': 0.7}
        for hand in medium_hands:
            charts[hand] = {'action': 'call', 'frequency': 0.5}
            
        return charts
    
    def evaluate_hand_strength(self, cards: List[Card], community: List[Card]) -> float:
        """
        Évalue la force de la main (0-1)
        """
        try:
            if len(cards) < 2:
                return 0.0
            
            # Évaluation de la main actuelle
            all_cards = cards + community
            hand_type, hand_value = self.hand_evaluator.evaluate_hand(all_cards)
            
            # Normalisation de la valeur (0-1)
            normalized_value = min(1.0, hand_value / 10000.0)
            
            # Bonus pour les mains premium
            if hand_type in ['royal_flush', 'straight_flush', 'four_of_a_kind']:
                normalized_value = 1.0
            elif hand_type in ['full_house', 'flush', 'straight']:
                normalized_value = max(normalized_value, 0.8)
            elif hand_type in ['three_of_a_kind', 'two_pair']:
                normalized_value = max(normalized_value, 0.6)
            elif hand_type == 'pair':
                normalized_value = max(normalized_value, 0.4)
            
            return normalized_value
            
        except Exception as e:
            self.logger.error(f"Erreur évaluation force main: {e}")
            return 0.0
    
    def calculate_pot_odds(self, pot_size: int, bet_size: int) -> float:
        """
        Calcule les pot odds
        """
        try:
            if bet_size == 0:
                return 0.0
            
            total_pot = pot_size + bet_size
            return bet_size / total_pot
            
        except Exception as e:
            self.logger.error(f"Erreur calcul pot odds: {e}")
            return 0.0
    
    def calculate_implied_odds(self, pot_size: int, bet_size: int, 
                             stack_sizes: List[int]) -> float:
        """
        Calcule les implied odds
        """
        try:
            pot_odds = self.calculate_pot_odds(pot_size, bet_size)
            
            # Facteur d'implied odds basé sur les stacks
            avg_stack = np.mean(stack_sizes) if stack_sizes else 0
            implied_factor = min(2.0, avg_stack / max(bet_size, 1))
            
            return pot_odds * implied_factor
            
        except Exception as e:
            self.logger.error(f"Erreur calcul implied odds: {e}")
            return 0.0
    
    def simulate_outcomes(self, state: GameState, num_simulations: int = 1000) -> Dict:
        """
        Simulation Monte Carlo des résultats possibles
        """
        try:
            results = {
                'win_rate': 0.0,
                'ev_call': 0.0,
                'ev_fold': 0.0,
                'ev_raise': 0.0,
                'pot_equity': 0.0
            }
            
            if not state.my_cards:
                return results
            
            # Cartes utilisées
            used_cards = state.my_cards + state.community_cards
            available_cards = self.get_available_cards(used_cards)
            
            wins = 0
            total_ev = 0
            
            for _ in range(num_simulations):
                # Simulation d'une main complète
                simulation_result = self.simulate_hand(state, available_cards)
                
                if simulation_result['won']:
                    wins += 1
                
                total_ev += simulation_result['ev']
            
            # Calcul des résultats
            results['win_rate'] = wins / num_simulations
            results['ev_call'] = total_ev / num_simulations
            results['ev_fold'] = 0.0  # Fold = 0 EV
            results['ev_raise'] = results['ev_call'] * 1.2  # Approximation
            results['pot_equity'] = results['win_rate'] * state.pot_size
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur simulation Monte Carlo: {e}")
            return {'win_rate': 0.0, 'ev_call': 0.0, 'ev_fold': 0.0, 'ev_raise': 0.0, 'pot_equity': 0.0}
    
    def simulate_hand(self, state: GameState, available_cards: List[Card]) -> Dict:
        """
        Simule une main complète
        """
        try:
            # Compléter les cartes communes
            remaining_community = 5 - len(state.community_cards)
            simulated_community = state.community_cards.copy()
            
            # Ajouter des cartes aléatoires
            for _ in range(remaining_community):
                if available_cards:
                    card = random.choice(available_cards)
                    simulated_community.append(card)
                    available_cards.remove(card)
            
            # Évaluer notre main
            our_strength = self.evaluate_hand_strength(state.my_cards, simulated_community)
            
            # Simuler les mains des adversaires
            opponent_strengths = []
            for _ in range(len(state.players) - 1):  # -1 pour nous
                if len(available_cards) >= 2:
                    opponent_cards = [available_cards.pop(), available_cards.pop()]
                    opponent_strength = self.evaluate_hand_strength(opponent_cards, simulated_community)
                    opponent_strengths.append(opponent_strength)
            
            # Déterminer le gagnant
            won = True
            for opp_strength in opponent_strengths:
                if opp_strength > our_strength:
                    won = False
                    break
            
            # Calculer l'EV
            ev = state.pot_size if won else -state.current_bet
            
            return {'won': won, 'ev': ev}
            
        except Exception as e:
            self.logger.error(f"Erreur simulation main: {e}")
            return {'won': False, 'ev': 0}
    
    def get_available_cards(self, used_cards: List[Card]) -> List[Card]:
        """
        Génère la liste des cartes disponibles
        """
        all_cards = []
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['♠', '♥', '♦', '♣']
        
        for rank in ranks:
            for suit in suits:
                card = Card(rank=rank, suit=suit, confidence=1.0)
                all_cards.append(card)
        
        # Retirer les cartes utilisées
        used_card_strings = [f"{card.rank}{card.suit}" for card in used_cards]
        available = [card for card in all_cards if f"{card.rank}{card.suit}" not in used_card_strings]
        
        return available
    
    def get_preflop_action(self, cards: List[Card], position: str, num_players: int) -> str:
        """
        Retourne l'action pré-flop optimale
        """
        try:
            if len(cards) != 2:
                return 'fold'
            
            # Créer la clé de la main
            card1, card2 = cards[0], cards[1]
            rank1, rank2 = card1.rank, card2.rank
            suited = card1.suit == card2.suit
            
            # Trier les rangs (plus haut en premier)
            if self.hand_evaluator.rank_values[rank1] < self.hand_evaluator.rank_values[rank2]:
                rank1, rank2 = rank2, rank1
            
            hand_key = f"{rank1}{rank2}{'s' if suited else 'o'}"
            
            # Chercher dans les charts
            if hand_key in self.preflop_charts:
                chart_entry = self.preflop_charts[hand_key]
                action = chart_entry['action']
                frequency = chart_entry['frequency']
                
                # Ajuster selon la position
                if position == 'late' and action == 'call':
                    action = 'raise'
                elif position == 'early' and action == 'raise':
                    action = 'call'
                
                # Décision probabiliste
                if random.random() < frequency:
                    return action
                else:
                    return 'fold'
            
            # Main non trouvée dans les charts
            return 'fold'
            
        except Exception as e:
            self.logger.error(f"Erreur action pré-flop: {e}")
            return 'fold'
    
    def calculate_bet_sizing(self, state: GameState, action: str) -> int:
        """
        Calcule la taille de mise optimale
        """
        try:
            if action not in ['bet', 'raise']:
                return 0
            
            pot_size = state.pot_size
            stack = state.my_stack
            
            if action == 'bet':
                # Bet sizing basé sur la force de la main
                hand_strength = self.evaluate_hand_strength(state.my_cards, state.community_cards)
                
                if hand_strength > 0.8:  # Main très forte
                    return min(int(pot_size * 0.75), stack)
                elif hand_strength > 0.6:  # Main forte
                    return min(int(pot_size * 0.5), stack)
                else:  # Main faible
                    return min(int(pot_size * 0.25), stack)
            
            elif action == 'raise':
                # Raise sizing
                current_bet = state.current_bet
                min_raise = state.min_raise
                
                if current_bet == 0:
                    return min(int(pot_size * 0.5), stack)
                else:
                    return min(current_bet * 2, stack)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Erreur calcul bet sizing: {e}")
            return 0
    
    def get_outs(self, state: GameState) -> int:
        """
        Calcule le nombre d'outs (cartes qui améliorent la main)
        """
        try:
            if not state.my_cards:
                return 0
            
            # Simulation pour compter les outs
            available_cards = self.get_available_cards(state.my_cards + state.community_cards)
            current_strength = self.evaluate_hand_strength(state.my_cards, state.community_cards)
            
            outs = 0
            for card in available_cards:
                # Tester si cette carte améliore notre main
                test_cards = state.my_cards + state.community_cards + [card]
                new_strength = self.evaluate_hand_strength(state.my_cards, test_cards)
                
                if new_strength > current_strength:
                    outs += 1
            
            return outs
            
        except Exception as e:
            self.logger.error(f"Erreur calcul outs: {e}")
            return 0 