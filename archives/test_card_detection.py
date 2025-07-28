"""
Script de test pour la détection des cartes
"""

import cv2
import numpy as np
from modules.image_analysis import ImageAnalyzer
from modules.screen_capture import ScreenCapture
import logging

def test_card_detection():
    """Teste la détection des cartes"""
    print("=== TEST DÉTECTION DES CARTES ===")
    
    # Initialiser les modules
    image_analyzer = ImageAnalyzer()
    screen_capture = ScreenCapture()
    
    # Configurer le logging
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        # 1. TEST CAPTURE DES RÉGIONS
        print("\n1. Capture des régions...")
        
        # Capturer hand_area
        hand_image = screen_capture.capture_region('hand_area')
        if hand_image is not None:
            print(f"✅ hand_area capturé: {hand_image.shape}")
            # Sauvegarder pour inspection
            cv2.imwrite('test_hand_area.png', hand_image)
        else:
            print("❌ Impossible de capturer hand_area")
            return
        
        # Capturer community_cards
        community_image = screen_capture.capture_region('community_cards')
        if community_image is not None:
            print(f"✅ community_cards capturé: {community_image.shape}")
            cv2.imwrite('test_community_cards.png', community_image)
        else:
            print("❌ Impossible de capturer community_cards")
        
        # 2. TEST DÉTECTION DES CARTES
        print("\n2. Détection des cartes...")
        
        # Détecter les cartes du joueur
        if hand_image is not None:
            print("🔍 Détection des cartes du joueur...")
            my_cards = image_analyzer.detect_cards(hand_image)
            print(f"Cartes détectées: {my_cards}")
            
            if my_cards:
                print("✅ Cartes du joueur détectées avec succès!")
                for card in my_cards:
                    print(f"   - {card} (confiance: {card.confidence:.2f})")
            else:
                print("❌ Aucune carte du joueur détectée")
        
        # Détecter les cartes communautaires
        if community_image is not None:
            print("\n🔍 Détection des cartes communautaires...")
            community_cards = image_analyzer.detect_cards(community_image)
            print(f"Cartes communautaires détectées: {community_cards}")
            
            if community_cards:
                print("✅ Cartes communautaires détectées avec succès!")
                for card in community_cards:
                    print(f"   - {card} (confiance: {card.confidence:.2f})")
            else:
                print("❌ Aucune carte communautaire détectée")
        
        # 3. TEST OCR
        print("\n3. Test OCR...")
        
        if hand_image is not None:
            text = image_analyzer.extract_text(hand_image)
            print(f"Texte extrait de hand_area: '{text}'")
        
        if community_image is not None:
            text = image_analyzer.extract_text(community_image)
            print(f"Texte extrait de community_cards: '{text}'")
        
        # 4. ANALYSE DES TEMPLATES
        print("\n4. Analyse des templates de cartes...")
        print(f"Templates chargés: {len(image_analyzer.card_templates)}")
        
        for card_key, template in list(image_analyzer.card_templates.items())[:5]:  # Afficher les 5 premiers
            if template is not None:
                print(f"   - {card_key}: {template.shape}")
            else:
                print(f"   - {card_key}: None")
        
        print("\n=== TEST TERMINÉ ===")
        print("Images sauvegardées: test_hand_area.png, test_community_cards.png")
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_card_detection() 