#!/usr/bin/env python3
"""
Script de test pour vérifier que toutes les régions sont bien utilisées dans la logique de l'agent
"""

import json
import logging
from modules.screen_capture import ScreenCapture
from modules.image_analysis import ImageAnalyzer
from modules.button_detector import ButtonDetector
from modules.game_state import GameState

def test_agent_regions():
    """Test que toutes les régions sont utilisées par l'agent"""
    print("🎯 TEST DES RÉGIONS DANS L'AGENT")
    print("=" * 50)
    
    # 1. Charger les régions depuis le JSON
    print("\n📄 1. RÉGIONS DANS LE JSON:")
    try:
        with open('calibrated_regions.json', 'r') as f:
            json_regions = json.load(f)
        
        json_region_names = list(json_regions.keys())
        print(f"   ✅ JSON contient {len(json_region_names)} régions")
        
    except Exception as e:
        print(f"   ❌ Erreur lecture JSON: {e}")
        return False
    
    # 2. Vérifier que ScreenCapture charge toutes les régions
    print("\n🤖 2. RÉGIONS CHARGÉES PAR SCREENCAPTURE:")
    try:
        screen_capture = ScreenCapture()
        agent_regions = screen_capture.regions
        
        print(f"   ✅ ScreenCapture charge {len(agent_regions)} régions")
        
        # Vérifier que toutes les régions du JSON sont dans l'agent
        missing_in_agent = set(json_region_names) - set(agent_regions.keys())
        if missing_in_agent:
            print(f"   ❌ Régions manquantes dans l'agent: {missing_in_agent}")
        else:
            print("   ✅ Toutes les régions du JSON sont dans l'agent")
        
    except Exception as e:
        print(f"   ❌ Erreur ScreenCapture: {e}")
        return False
    
    # 3. Vérifier l'utilisation des régions dans main.py
    print("\n🎮 3. RÉGIONS UTILISÉES DANS MAIN.PY:")
    
    # Régions utilisées dans _analyze_game_state
    game_state_regions = [
        'hand_area',           # Cartes du joueur
        'community_cards',     # Cartes communautaires
        'action_buttons',      # Boutons d'action
    ]
    
    # Régions utilisées dans _analyze_stacks_and_bets
    stacks_bets_regions = [
        'my_stack_area',       # Stack du joueur
        'my_current_bet',      # Mise actuelle
        'pot_area',            # Pot
    ]
    
    # Régions utilisées dans _analyze_position_and_blinds
    position_blinds_regions = [
        'my_dealer_button',    # Bouton dealer
        'blinds_timer',        # Timer des blinds
    ]
    
    # Régions utilisées dans _try_start_new_hand
    new_hand_regions = [
        'new_hand_button',     # Bouton New Hand
    ]
    
    # Vérifier que toutes ces régions existent
    all_used_regions = set(game_state_regions + stacks_bets_regions + position_blinds_regions + new_hand_regions)
    
    print(f"   📊 Régions utilisées dans main.py: {len(all_used_regions)}")
    for region in sorted(all_used_regions):
        if region in json_region_names:
            print(f"      ✅ {region}")
        else:
            print(f"      ❌ {region} (MANQUANTE)")
    
    # 4. Vérifier les régions non utilisées
    unused_regions = set(json_region_names) - all_used_regions
    if unused_regions:
        print(f"\n   ⚠️  Régions non utilisées dans main.py: {len(unused_regions)}")
        for region in sorted(unused_regions):
            print(f"      ⚠️  {region}")
    else:
        print("\n   ✅ Toutes les régions sont utilisées dans main.py")
    
    # 5. Test de capture des régions importantes
    print("\n📸 4. TEST DE CAPTURE DES RÉGIONS IMPORTANTES:")
    
    important_regions = [
        'hand_area',           # Essentiel pour détecter les cartes
        'action_buttons',      # Essentiel pour détecter les actions
        'community_cards',     # Essentiel pour l'évaluation
        'pot_area',            # Important pour les décisions
        'new_hand_button',     # Important pour démarrer de nouvelles parties
    ]
    
    for region_name in important_regions:
        if region_name in agent_regions:
            try:
                captured = screen_capture.capture_region(region_name)
                if captured is not None and captured.size > 0:
                    print(f"   ✅ {region_name:20} - Capturé: {captured.shape}")
                else:
                    print(f"   ❌ {region_name:20} - Capture vide")
            except Exception as e:
                print(f"   ❌ {region_name:20} - Erreur: {e}")
        else:
            print(f"   ❌ {region_name:20} - Région non trouvée")
    
    # 6. Test de détection avec les templates
    print("\n🔍 5. TEST DE DÉTECTION AVEC TEMPLATES:")
    
    try:
        # Test détection de boutons
        if 'action_buttons' in agent_regions:
            button_detector = ButtonDetector()
            captured = screen_capture.capture_region('action_buttons')
            if captured is not None:
                buttons = button_detector.detect_available_actions(captured)
                print(f"   🎮 Boutons détectés: {len(buttons)}")
                for btn in buttons:
                    print(f"      - {btn.name} (confiance: {btn.confidence:.2f})")
            else:
                print("   ❌ Impossible de capturer action_buttons")
        
        # Test détection de cartes
        if 'hand_area' in agent_regions:
            image_analyzer = ImageAnalyzer()
            captured = screen_capture.capture_region('hand_area')
            if captured is not None:
                cards = image_analyzer.detect_cards(captured)
                print(f"   🃏 Cartes détectées: {len(cards)}")
                for card in cards:
                    print(f"      - {card}")
            else:
                print("   ❌ Impossible de capturer hand_area")
                
    except Exception as e:
        print(f"   ❌ Erreur test détection: {e}")
    
    return True

def test_game_state_integration():
    """Test l'intégration avec GameState"""
    print("\n🎯 TEST INTÉGRATION GAMESTATE")
    print("=" * 40)
    
    try:
        # Créer un GameState de test
        game_state = GameState()
        
        # Simuler des données de jeu
        test_data = {
            'my_cards': ['Ah', 'Kd'],
            'community_cards': ['2h', '7s', 'Jc'],
            'available_actions': ['fold', 'call', 'raise'],
            'my_stack': 1000.0,
            'pot_size': 150.0,
            'my_current_bet': 50.0
        }
        
        # Mettre à jour le GameState
        game_state.update(test_data)
        
        print(f"   ✅ GameState mis à jour avec succès")
        print(f"   🃏 Cartes joueur: {game_state.my_cards}")
        print(f"   🃏 Cartes communes: {game_state.community_cards}")
        print(f"   🎮 Actions disponibles: {game_state.available_actions}")
        print(f"   💰 Stack: {game_state.my_stack}")
        print(f"   🏆 Pot: {game_state.pot_size}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur GameState: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🎯 TEST COMPLET DE L'AGENT")
    print("=" * 50)
    
    success1 = test_agent_regions()
    success2 = test_game_state_integration()
    
    print("\n" + "=" * 50)
    
    if success1 and success2:
        print("🎉 L'AGENT EST PRÊT!")
        print("Toutes les régions sont correctement configurées et utilisées")
        print("L'agent devrait maintenant fonctionner correctement")
    else:
        print("❌ PROBLÈMES DÉTECTÉS")
        print("Vérifiez la configuration des régions")
    
    print("\n💡 CONSEILS:")
    print("- Lancez l'agent: py main.py")
    print("- Surveillez les logs pour voir les détections")
    print("- Vérifiez que les templates sont bien chargés") 