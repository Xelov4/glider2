#!/usr/bin/env python3
"""
🔍 TEST SIMPLE DE DÉTECTION DE CARTES
=====================================

Test simple pour vérifier la détection de cartes sans problèmes d'encodage.
"""

import sys
import os
import time
from datetime import datetime

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_simple_card_detection():
    """Test simple de la détection de cartes"""
    print("🔍 TEST SIMPLE DE DÉTECTION DE CARTES")
    print("=" * 50)
    print(f"⏰ Début du test: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    try:
        # Import des modules
        print("📦 Import des modules...")
        from modules.screen_capture import ScreenCapture
        print("✅ ScreenCapture importé")
        
        # Test de capture d'écran
        print("\n📸 Test de capture d'écran...")
        screen_capture = ScreenCapture()
        
        # Régions à tester
        regions_to_test = [
            'hand_area',
            'community_cards',
            'pot_area',
            'my_stack_area'
        ]
        
        for region_name in regions_to_test:
            print(f"\n🎯 Test de la région: {region_name}")
            print("-" * 30)
            
            try:
                # Capture de la région
                start_time = time.time()
                region_image = screen_capture.capture_region(region_name)
                capture_time = time.time() - start_time
                
                if region_image is None:
                    print(f"❌ Impossible de capturer {region_name}")
                    continue
                
                # Informations sur l'image
                height, width = region_image.shape[:2]
                channels = region_image.shape[2] if len(region_image.shape) == 3 else 1
                total_pixels = height * width
                
                print(f"✅ Image capturée: {width}x{height} ({channels} canaux)")
                print(f"📊 Pixels totaux: {total_pixels:,}")
                print(f"⏱️  Temps capture: {capture_time:.3f}s")
                
                # Test OCR simple
                print(f"🔍 Test OCR simple...")
                try:
                    import pytesseract
                    
                    # Configuration OCR simple
                    config = '--oem 3 --psm 6'
                    
                    # Conversion en niveaux de gris
                    import cv2
                    if len(region_image.shape) == 3:
                        gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = region_image.copy()
                    
                    # OCR
                    start_time = time.time()
                    text = pytesseract.image_to_string(gray, config=config)
                    ocr_time = time.time() - start_time
                    
                    clean_text = text.strip()
                    print(f"  Texte détecté: '{clean_text}'")
                    print(f"  Longueur: {len(clean_text)} caractères")
                    print(f"  Temps OCR: {ocr_time:.3f}s")
                    
                    # Chercher des cartes dans le texte
                    card_chars = []
                    for char in clean_text:
                        if char in '23456789TJQKA':
                            card_chars.append(char)
                    
                    if card_chars:
                        print(f"  🃏 Rangs détectés: {card_chars}")
                    else:
                        print(f"  ❌ Aucun rang détecté")
                    
                except Exception as e:
                    print(f"  ❌ Erreur OCR: {e}")
                
                # Test d'analyse de couleur simple
                print(f"🎨 Test analyse couleur...")
                try:
                    import cv2
                    import numpy as np
                    
                    # Conversion HSV
                    hsv = cv2.cvtColor(region_image, cv2.COLOR_BGR2HSV)
                    
                    # Masques de couleurs
                    lower_red1 = np.array([0, 50, 50])
                    upper_red1 = np.array([10, 255, 255])
                    lower_red2 = np.array([170, 50, 50])
                    upper_red2 = np.array([180, 255, 255])
                    
                    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
                    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
                    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
                    
                    lower_black = np.array([0, 0, 0])
                    upper_black = np.array([180, 255, 50])
                    mask_black = cv2.inRange(hsv, lower_black, upper_black)
                    
                    # Compter les pixels
                    red_pixels = cv2.countNonZero(mask_red)
                    black_pixels = cv2.countNonZero(mask_black)
                    
                    print(f"  Pixels rouges: {red_pixels:,} ({red_pixels/total_pixels*100:.1f}%)")
                    print(f"  Pixels noirs: {black_pixels:,} ({black_pixels/total_pixels*100:.1f}%)")
                    
                    if red_pixels > total_pixels * 0.05:
                        print(f"  🔴 Rouge détecté (cartes ♥♦)")
                    elif black_pixels > total_pixels * 0.05:
                        print(f"  ⚫ Noir détecté (cartes ♠♣)")
                    else:
                        print(f"  ⚪ Pas de couleur dominante")
                    
                except Exception as e:
                    print(f"  ❌ Erreur analyse couleur: {e}")
                
            except Exception as e:
                print(f"❌ Erreur test {region_name}: {e}")
        
        print(f"\n✅ TEST SIMPLE TERMINÉ!")
        print(f"⏰ Fin du test: {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_card_detection() 