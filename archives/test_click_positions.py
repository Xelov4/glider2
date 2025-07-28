#!/usr/bin/env python3
"""
Test des positions de clic de l'agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.screen_capture import ScreenCapture

def test_click_positions():
    """Test et affiche les positions de clic pour chaque région"""
    
    print("=== TEST POSITIONS DE CLIC ===")
    
    screen_capture = ScreenCapture()
    
    # Régions critiques pour le poker
    critical_regions = [
        'new_hand_button',
        'action_buttons', 
        'resume_button',
        'hand_area',
        'community_cards'
    ]
    
    print("\n🎯 POSITIONS DE CLIC CALCULÉES:")
    for region_name in critical_regions:
        if region_name in screen_capture.regions:
            region = screen_capture.regions[region_name]
            
            # Calculer le centre (position de clic)
            center_x = region.x + region.width // 2
            center_y = region.y + region.height // 2
            
            print(f"📍 {region_name}:")
            print(f"   Région: ({region.x}, {region.y}, {region.width}, {region.height})")
            print(f"   Clic:   ({center_x}, {center_y})")
            print()
        else:
            print(f"❌ {region_name}: MANQUANT")
    
    # Test spécifique pour new_hand_button
    print("=== TEST SPÉCIFIQUE NEW HAND BUTTON ===")
    if 'new_hand_button' in screen_capture.regions:
        region = screen_capture.regions['new_hand_button']
        center_x = region.x + region.width // 2
        center_y = region.y + region.height // 2
        
        print(f"Région new_hand_button: ({region.x}, {region.y}, {region.width}, {region.height})")
        print(f"Position de clic calculée: ({center_x}, {center_y})")
        print(f"Position de clic attendue: (1179, 206)")
        
        if center_x == 1179 and center_y == 206:
            print("✅ Position de clic CORRECTE!")
        else:
            print("❌ Position de clic INCORRECTE!")
            print(f"Différence: ({center_x - 1179}, {center_y - 206})")

def test_region_capture():
    """Test de capture d'une région spécifique"""
    
    print("\n=== TEST CAPTURE RÉGION ===")
    
    screen_capture = ScreenCapture()
    
    # Tester la capture de new_hand_button
    if 'new_hand_button' in screen_capture.regions:
        region = screen_capture.regions['new_hand_button']
        print(f"Région new_hand_button: {region}")
        
        # Capturer la région
        captured = screen_capture.capture_region('new_hand_button')
        if captured is not None:
            print(f"✅ Capture réussie: {captured.shape}")
            
            # Sauvegarder l'image pour inspection
            import cv2
            cv2.imwrite('test_new_hand_button.png', captured)
            print("💾 Image sauvegardée: test_new_hand_button.png")
        else:
            print("❌ Échec de la capture")
    else:
        print("❌ Région new_hand_button non trouvée")

if __name__ == "__main__":
    print("🔍 Test des positions de clic...")
    
    test_click_positions()
    test_region_capture()
    
    print("\n📋 Vérifiez l'image 'test_new_hand_button.png' pour voir ce qui est capturé") 