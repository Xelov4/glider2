"""
Script de test pour la stratégie Spin & Rush
"""

import sys
import logging
from modules.spin_rush_strategy import SpinRushStrategy

def test_spin_rush_strategy():
    """Teste la stratégie Spin & Rush avec différents scénarios"""
    
    # Configuration du logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("=== TEST STRATÉGIE SPIN & RUSH ===\n")
    
    # Créer la stratégie
    strategy = SpinRushStrategy()
    
    # Classe simulée pour GameState
    class MockGameState:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # Tests de différents scénarios
    test_scenarios = [
        {
            'name': 'BTN - Main forte - Timer normal',
            'game_state': MockGameState(
                my_cards=['A♠', 'K♠'],
                community_cards=[],
                my_stack=500,
                pot_size=30,
                big_blind=10,
                street='preflop',
                blinds_timer=60,
                num_players=3,
                my_position='BTN'
            ),
            'expected': 'raise'  # BTN avec main forte = raise
        },
        {
            'name': 'BB - Main faible - Timer urgent',
            'game_state': MockGameState(
                my_cards=['7♣', '2♦'],
                community_cards=[],
                my_stack=150,
                pot_size=45,
                big_blind=15,
                street='preflop',
                blinds_timer=10,  # Timer urgent
                num_players=3,
                my_position='BB'
            ),
            'expected': 'all_in'  # Timer urgent = all-in
        },
        {
            'name': 'UTG - Main moyenne - Stack court',
            'game_state': MockGameState(
                my_cards=['J♠', 'T♠'],
                community_cards=[],
                my_stack=75,  # Stack court
                pot_size=30,
                big_blind=10,
                street='preflop',
                blinds_timer=45,
                num_players=3,
                my_position='UTG'
            ),
            'expected': 'all_in'  # Stack court = all-in
        },
        {
            'name': 'BTN - Main forte - Flop',
            'game_state': MockGameState(
                my_cards=['A♠', 'K♠'],
                community_cards=['A♥', 'K♦', 'Q♠'],
                my_stack=400,
                pot_size=120,
                big_blind=20,
                street='flop',
                blinds_timer=30,
                num_players=3,
                my_position='BTN'
            ),
            'expected': 'raise'  # Main forte postflop = raise
        }
    ]
    
    print("🎯 Tests de décisions Spin & Rush:\n")
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print(f"   Cartes: {scenario['game_state'].my_cards}")
        print(f"   Position: {scenario['game_state'].my_position}")
        print(f"   Stack: {scenario['game_state'].my_stack}")
        print(f"   Timer: {scenario['game_state'].blinds_timer}s")
        
        try:
            # Obtenir la décision
            decision = strategy.get_action_decision(scenario['game_state'])
            
            # Vérifier si la décision est correcte
            if decision == scenario['expected']:
                status = "✅ CORRECT"
            else:
                status = "❌ INCORRECT"
            
            print(f"   Décision: {decision} (attendu: {scenario['expected']})")
            print(f"   Résultat: {status}")
            
            # Afficher les détails de la décision
            if decision == 'all_in':
                print("   💡 Raison: Stack court ou timer urgent")
            elif decision == 'raise':
                print("   💡 Raison: Main forte ou bluff")
            elif decision == 'call':
                print("   💡 Raison: Pot odds favorables")
            elif decision == 'fold':
                print("   💡 Raison: Main faible")
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
        
        print()
    
    # Test des ranges par position
    print("📋 Test des ranges par position:\n")
    
    positions = ['UTG', 'BTN', 'BB']
    test_hands = [
        ('AA', 'Paire d\'As'),
        ('AKs', 'AK suited'),
        ('AKo', 'AK offsuit'),
        ('72o', '72 offsuit'),
        ('T9s', 'T9 suited')
    ]
    
    for position in positions:
        print(f"Position {position}:")
        position_range = strategy.position_ranges[position]
        
        for hand, description in test_hands:
            # Simuler la vérification de range
            if hand in position_range['pairs'] or hand in position_range['suited'] or hand in position_range['offsuit']:
                status = "✅ DANS LA RANGE"
            else:
                status = "❌ HORS RANGE"
            
            print(f"   {hand} ({description}): {status}")
        
        print()
    
    # Test du bet sizing selon le timer
    print("💰 Test du bet sizing selon le timer:\n")
    
    timer_modes = [
        (60, 'normal'),
        (25, 'pressure'),
        (10, 'urgent')
    ]
    
    for timer, mode in timer_modes:
        sizing = strategy.timer_bet_sizing[mode]
        print(f"Timer {timer}s ({mode}):")
        print(f"   Preflop: {sizing['preflop']}x BB")
        print(f"   Postflop: {sizing['postflop']}x pot")
        print()
    
    print("=== TEST TERMINÉ ===")

if __name__ == "__main__":
    test_spin_rush_strategy() 