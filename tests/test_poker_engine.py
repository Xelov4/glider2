"""
Tests pour le module poker engine
"""

import unittest
import sys
import os

# Ajouter le répertoire parent au path pour importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.poker_engine import PokerEngine, HandEvaluator
from modules.image_analysis import Card
from modules.game_state import GameState

class TestHandEvaluator(unittest.TestCase):
    """Tests pour l'évaluateur de mains"""
    
    def setUp(self):
        self.evaluator = HandEvaluator()
    
    def test_royal_flush(self):
        """Test d'une quinte flush royale"""
        cards = [
            Card('A', '♠', 1.0),
            Card('K', '♠', 1.0),
            Card('Q', '♠', 1.0),
            Card('J', '♠', 1.0),
            Card('10', '♠', 1.0)
        ]
        hand_type, value = self.evaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, 'royal_flush')
        self.assertGreaterEqual(value, 9000)
    
    def test_straight_flush(self):
        """Test d'une quinte flush"""
        cards = [
            Card('9', '♥', 1.0),
            Card('8', '♥', 1.0),
            Card('7', '♥', 1.0),
            Card('6', '♥', 1.0),
            Card('5', '♥', 1.0)
        ]
        hand_type, value = self.evaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, 'straight_flush')
        self.assertGreater(value, 8000)
    
    def test_four_of_a_kind(self):
        """Test d'un carré"""
        cards = [
            Card('A', '♠', 1.0),
            Card('A', '♥', 1.0),
            Card('A', '♦', 1.0),
            Card('A', '♣', 1.0),
            Card('K', '♠', 1.0)
        ]
        hand_type, value = self.evaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, 'four_of_a_kind')
        self.assertGreater(value, 7000)
    
    def test_full_house(self):
        """Test d'un full"""
        cards = [
            Card('A', '♠', 1.0),
            Card('A', '♥', 1.0),
            Card('A', '♦', 1.0),
            Card('K', '♣', 1.0),
            Card('K', '♠', 1.0)
        ]
        hand_type, value = self.evaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, 'full_house')
        self.assertGreater(value, 6000)
    
    def test_flush(self):
        """Test d'une couleur"""
        cards = [
            Card('A', '♠', 1.0),
            Card('K', '♠', 1.0),
            Card('Q', '♠', 1.0),
            Card('J', '♠', 1.0),
            Card('9', '♠', 1.0)
        ]
        hand_type, value = self.evaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, 'flush')
        self.assertGreater(value, 5000)
    
    def test_straight(self):
        """Test d'une quinte"""
        cards = [
            Card('A', '♠', 1.0),
            Card('K', '♥', 1.0),
            Card('Q', '♦', 1.0),
            Card('J', '♣', 1.0),
            Card('10', '♠', 1.0)
        ]
        hand_type, value = self.evaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, 'straight')
        self.assertGreater(value, 4000)
    
    def test_three_of_a_kind(self):
        """Test d'un brelan"""
        cards = [
            Card('A', '♠', 1.0),
            Card('A', '♥', 1.0),
            Card('A', '♦', 1.0),
            Card('K', '♣', 1.0),
            Card('Q', '♠', 1.0)
        ]
        hand_type, value = self.evaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, 'three_of_a_kind')
        self.assertGreater(value, 3000)
    
    def test_two_pair(self):
        """Test d'une double paire"""
        cards = [
            Card('A', '♠', 1.0),
            Card('A', '♥', 1.0),
            Card('K', '♦', 1.0),
            Card('K', '♣', 1.0),
            Card('Q', '♠', 1.0)
        ]
        hand_type, value = self.evaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, 'two_pair')
        self.assertGreater(value, 2000)
    
    def test_pair(self):
        """Test d'une paire"""
        cards = [
            Card('A', '♠', 1.0),
            Card('A', '♥', 1.0),
            Card('K', '♦', 1.0),
            Card('Q', '♣', 1.0),
            Card('J', '♠', 1.0)
        ]
        hand_type, value = self.evaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, 'pair')
        self.assertGreater(value, 1000)
    
    def test_high_card(self):
        """Test d'une carte haute"""
        cards = [
            Card('A', '♠', 1.0),
            Card('K', '♥', 1.0),
            Card('Q', '♦', 1.0),
            Card('J', '♣', 1.0),
            Card('9', '♠', 1.0)
        ]
        hand_type, value = self.evaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, 'high_card')
        self.assertEqual(value, 14)  # Valeur de l'As

class TestPokerEngine(unittest.TestCase):
    """Tests pour le moteur de poker"""
    
    def setUp(self):
        self.engine = PokerEngine()
    
    def test_evaluate_hand_strength(self):
        """Test de l'évaluation de force de main"""
        # Main forte (quinte flush royale)
        cards = [
            Card('A', '♠', 1.0),
            Card('K', '♠', 1.0)
        ]
        community = [
            Card('Q', '♠', 1.0),
            Card('J', '♠', 1.0),
            Card('10', '♠', 1.0)
        ]
        
        strength = self.engine.evaluate_hand_strength(cards, community)
        self.assertGreater(strength, 0.9)  # Très forte
        
        # Main faible
        cards = [
            Card('2', '♠', 1.0),
            Card('7', '♥', 1.0)
        ]
        community = [
            Card('3', '♦', 1.0),
            Card('8', '♣', 1.0),
            Card('K', '♠', 1.0)
        ]
        
        strength = self.engine.evaluate_hand_strength(cards, community)
        self.assertLess(strength, 0.5)  # Faible
    
    def test_calculate_pot_odds(self):
        """Test du calcul des pot odds"""
        pot_size = 100
        bet_size = 25
        
        odds = self.engine.calculate_pot_odds(pot_size, bet_size)
        expected = 25 / 125  # bet_size / (pot_size + bet_size)
        self.assertAlmostEqual(odds, expected, places=5)
    
    def test_get_preflop_action(self):
        """Test des actions pré-flop"""
        # Main premium
        cards = [Card('A', '♠', 1.0), Card('A', '♥', 1.0)]
        action = self.engine.get_preflop_action(cards, 'late', 6)
        self.assertIn(action, ['raise', 'fold'])
        
        # Main faible
        cards = [Card('2', '♠', 1.0), Card('7', '♥', 1.0)]
        action = self.engine.get_preflop_action(cards, 'early', 6)
        self.assertEqual(action, 'fold')
    
    def test_calculate_bet_sizing(self):
        """Test du calcul de taille de mise"""
        # Créer un état de jeu simulé
        state = GameState()
        state.pot_size = 100
        state.my_stack = 1000
        state.current_bet = 0
        state.my_cards = [Card('A', '♠', 1.0), Card('A', '♥', 1.0)]
        state.community_cards = []
        
        # Test bet sizing
        bet_size = self.engine.calculate_bet_sizing(state, 'bet')
        self.assertGreater(bet_size, 0)
        self.assertLessEqual(bet_size, state.my_stack)
    
    def test_get_outs(self):
        """Test du calcul d'outs"""
        state = GameState()
        state.my_cards = [Card('A', '♠', 1.0), Card('K', '♠', 1.0)]
        state.community_cards = [
            Card('Q', '♠', 1.0),
            Card('J', '♠', 1.0),
            Card('2', '♥', 1.0)
        ]
        
        outs = self.engine.get_outs(state)
        self.assertGreaterEqual(outs, 0)

if __name__ == '__main__':
    unittest.main() 