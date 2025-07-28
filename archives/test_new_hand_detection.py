#!/usr/bin/env python3
"""
Script de test pour la détection du bouton "New Hand"
"""

import time
import logging
from modules.screen_capture import ScreenCapture
from modules.automation import AutomationEngine

def test_new_hand_detection():
    """Test de détection du bouton New Hand"""
    print("🎮 TEST DÉTECTION BOUTON 'NEW HAND'")
    print("=" * 50)
    
    try:
        # Initialiser les modules
        screen_capture = ScreenCapture()
        automation = AutomationEngine()
        
        print("✅ Modules initialisés")
        
        # Capturer les régions
        print("\n📸 Capture des régions...")
        regions = screen_capture.capture_all_regions()
        print(f"Régions capturées: {len(regions)}")
        
        # Vérifier si la région "new_hand_button" existe
        if 'new_hand_button' in regions:
            print("✅ Région 'new_hand_button' trouvée")
            
            # Obtenir les informations de la région
            region_info = screen_capture.get_region_info('new_hand_button')
            if region_info:
                print(f"   Position: ({region_info['x']}, {region_info['y']})")
                print(f"   Taille: {region_info['width']}x{region_info['height']}")
                
                # Calculer le centre
                center_x = region_info['x'] + region_info['width'] // 2
                center_y = region_info['y'] + region_info['height'] // 2
                print(f"   Centre: ({center_x}, {center_y})")
                
                # Test de clic (simulation)
                print(f"\n🖱️ Test de clic sur 'New Hand'...")
                print("   (Simulation - pas de vrai clic)")
                print(f"   Position: ({center_x}, {center_y})")
                
                return True
            else:
                print("❌ Informations de région non disponibles")
                return False
        else:
            print("❌ Région 'new_hand_button' non trouvée")
            print("💡 Vérifiez que la région est configurée dans calibrated_regions.json")
            return False
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_ocr_detection():
    """Test de détection OCR du texte 'New Hand'"""
    print("\n🔍 TEST DÉTECTION OCR 'NEW HAND'")
    print("=" * 40)
    
    try:
        import pyautogui
        import pytesseract
        
        # Capturer l'écran complet
        print("📸 Capture de l'écran complet...")
        screenshot = pyautogui.screenshot()
        
        # Extraire le texte
        print("🔍 Extraction du texte...")
        text = pytesseract.image_to_string(screenshot, config='--psm 6')
        
        # Chercher "New Hand" ou variantes
        search_terms = ['new hand', 'nouvelle main', 'new game', 'nouvelle partie']
        found_terms = []
        
        for term in search_terms:
            if term.lower() in text.lower():
                found_terms.append(term)
        
        if found_terms:
            print(f"✅ Termes trouvés: {found_terms}")
            
            # Obtenir les coordonnées du texte
            import cv2
            import numpy as np
            
            img_np = np.array(screenshot)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            data = pytesseract.image_to_data(img_gray, output_type=pytesseract.Output.DICT)
            
            print("📍 Coordonnées des textes trouvés:")
            for i, text_detected in enumerate(data['text']):
                if any(term in text_detected.lower() for term in search_terms):
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    conf = data['conf'][i]
                    
                    print(f"   - '{text_detected}' à ({x}, {y}) - Confiance: {conf}%")
            
            return True
        else:
            print("❌ Aucun terme 'New Hand' trouvé dans l'écran")
            print("💡 Vérifiez que:")
            print("   - Une partie de poker est ouverte")
            print("   - Le bouton 'New Hand' est visible")
            print("   - Le texte est lisible")
            
            return False
        
    except Exception as e:
        print(f"❌ Erreur OCR: {e}")
        return False

def test_automation_click():
    """Test de clic automatisé"""
    print("\n🖱️ TEST CLIC AUTOMATISÉ")
    print("=" * 30)
    
    try:
        automation = AutomationEngine()
        
        # Position de test (basée sur calibrated_regions.json)
        test_x = 4338 + 290 // 2  # x + width/2
        test_y = 962 + 60 // 2    # y + height/2
        
        print(f"🎯 Position de test: ({test_x}, {test_y})")
        print("⚠️  ATTENTION: Ce test va effectuer un vrai clic!")
        print("   Assurez-vous que Betclic Poker est ouvert et visible")
        
        response = input("Continuer ? (o/n): ").lower()
        
        if response in ['o', 'oui', 'y', 'yes']:
            print("🖱️ Exécution du clic...")
            success = automation.click_at_position(test_x, test_y)
            
            if success:
                print("✅ Clic réussi!")
                return True
            else:
                print("❌ Clic échoué")
                return False
        else:
            print("⏹️ Test annulé")
            return True
        
    except Exception as e:
        print(f"❌ Erreur clic: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🎰 TEST COMPLET DÉTECTION 'NEW HAND'")
    print("=" * 50)
    
    # Tests
    success1 = test_new_hand_detection()
    success2 = test_ocr_detection()
    success3 = test_automation_click()
    
    print("\n" + "=" * 50)
    
    if success1 and success2 and success3:
        print("🎉 TOUS LES TESTS PASSÉS!")
        print("L'agent devrait maintenant pouvoir lancer des nouvelles parties")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("Vérifiez la configuration et recalibrez si nécessaire")
    
    print("\n💡 CONSEILS:")
    print("- Assurez-vous que Betclic Poker est ouvert")
    print("- Vérifiez que le bouton 'New Hand' est visible")
    print("- Recalibrez les régions si nécessaire")
    print("- L'agent cherchera automatiquement le bouton toutes les 20 secondes") 