#!/usr/bin/env python3
"""
Script de test pour simuler une partie de poker et voir si l'agent réagit
"""

import time
import logging
from modules.screen_capture import ScreenCapture
from modules.image_analysis import ImageAnalyzer
from modules.game_state import GameState
from modules.button_detector import ButtonDetector

def test_agent_reaction():
    """Test si l'agent réagit à une partie de poker"""
    print("🎰 TEST RÉACTION AGENT POKER")
    print("=" * 50)
    
    try:
        # Initialiser les modules
        screen_capture = ScreenCapture()
        image_analyzer = ImageAnalyzer()
        game_state = GameState()
        button_detector = ButtonDetector()
        
        print("✅ Modules initialisés")
        
        # Capturer les régions
        print("\n📸 Capture des régions...")
        regions = screen_capture.capture_all_regions()
        print(f"Régions capturées: {len(regions)}")
        
        # Analyser les boutons d'action
        if 'action_buttons' in regions:
            print("\n🎮 Analyse des boutons d'action...")
            buttons = button_detector.detect_available_actions(regions['action_buttons'])
            
            if buttons:
                print("✅ Boutons détectés:")
                for btn in buttons:
                    print(f"   - {btn.name} (confiance: {btn.confidence:.2f})")
            else:
                print("❌ Aucun bouton détecté")
                print("💡 Cela peut signifier:")
                print("   - Pas de partie en cours")
                print("   - Pas notre tour")
                print("   - Problème de détection")
        
        # Analyser les cartes
        if 'hand_area' in regions:
            print("\n🃏 Analyse des cartes...")
            cards = image_analyzer.detect_cards(regions['hand_area'])
            
            if cards:
                print("✅ Cartes détectées:")
                for card in cards:
                    print(f"   - {card}")
            else:
                print("❌ Aucune carte détectée")
        
        # Analyser les cartes communes
        if 'community_cards' in regions:
            print("\n🃏 Analyse des cartes communes...")
            community_cards = image_analyzer.detect_cards(regions['community_cards'])
            
            if community_cards:
                print("✅ Cartes communes détectées:")
                for card in community_cards:
                    print(f"   - {card}")
            else:
                print("❌ Aucune carte commune détectée")
        
        # Analyser le stack
        if 'my_stack_area' in regions:
            print("\n💰 Analyse du stack...")
            stack_text = image_analyzer.extract_text(regions['my_stack_area'])
            print(f"Texte extrait: '{stack_text}'")
        
        # Test de l'état du jeu
        print("\n🎯 Test de l'état du jeu...")
        
        # Simuler des informations de jeu
        test_game_info = {
            'my_cards': cards if 'hand_area' in regions else [],
            'community_cards': community_cards if 'community_cards' in regions else [],
            'available_actions': buttons if 'action_buttons' in regions else [],
            'my_stack': 1000,  # Valeur par défaut
            'pot_size': 50,     # Valeur par défaut
            'current_bet': 0    # Valeur par défaut
        }
        
        # Mettre à jour l'état du jeu
        game_state.update(test_game_info)
        
        print(f"État du jeu:")
        print(f"   - Mes cartes: {len(game_state.my_cards)}")
        print(f"   - Cartes communes: {len(game_state.community_cards)}")
        print(f"   - Actions disponibles: {len(game_state.available_actions)}")
        print(f"   - Mon stack: {game_state.my_stack}")
        print(f"   - Pot: {game_state.pot_size}")
        print(f"   - C'est mon tour: {game_state.is_my_turn}")
        
        # Recommandations
        print("\n💡 RECOMMANDATIONS:")
        
        if not buttons:
            print("1. Assurez-vous qu'une partie de poker est en cours")
            print("2. Vérifiez que c'est votre tour de jouer")
            print("3. Les boutons d'action doivent être visibles")
        
        if not cards:
            print("4. Vérifiez que vos cartes sont visibles")
            print("5. Recalibrez les régions si nécessaire")
        
        if game_state.is_my_turn:
            print("6. 🎯 L'agent devrait jouer maintenant !")
        else:
            print("6. Attendez votre tour ou vérifiez la détection")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_continuous_monitoring():
    """Test de monitoring continu"""
    print("\n🔄 TEST MONITORING CONTINU")
    print("=" * 30)
    
    try:
        screen_capture = ScreenCapture()
        button_detector = ButtonDetector()
        
        print("Monitoring en cours... (Ctrl+C pour arrêter)")
        print("Recherche de boutons d'action...")
        
        for i in range(30):  # 30 secondes de test
            regions = screen_capture.capture_all_regions()
            
            if 'action_buttons' in regions:
                buttons = button_detector.detect_available_actions(regions['action_buttons'])
                
                if buttons:
                    button_names = [btn.name for btn in buttons]
                    print(f"🎮 [{i+1:2d}s] Boutons: {button_names}")
                else:
                    if i % 5 == 0:  # Afficher moins souvent
                        print(f"⏳ [{i+1:2d}s] En attente...")
            
            time.sleep(1)
        
        print("✅ Test de monitoring terminé")
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring arrêté par l'utilisateur")
        return True
    except Exception as e:
        print(f"❌ Erreur monitoring: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test de réaction
    success1 = test_agent_reaction()
    
    # Test de monitoring (optionnel)
    print("\nVoulez-vous tester le monitoring continu ? (o/n)")
    response = input().lower()
    
    if response in ['o', 'oui', 'y', 'yes']:
        success2 = test_continuous_monitoring()
    else:
        success2 = True
    
    print("\n" + "=" * 50)
    
    if success1 and success2:
        print("🎉 Tests terminés avec succès!")
        print("L'agent devrait maintenant fonctionner correctement")
    else:
        print("❌ Certains tests ont échoué")
        print("Vérifiez la configuration et recalibrez si nécessaire") 