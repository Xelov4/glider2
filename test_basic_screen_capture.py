#!/usr/bin/env python3
"""
📸 TEST BASIQUE DE CAPTURE D'ÉCRAN
===================================

Test très simple pour vérifier la capture d'écran sans dépendances complexes.
"""

import sys
import os
import time
from datetime import datetime

def test_basic_screen_capture():
    """Test basique de capture d'écran"""
    print("📸 TEST BASIQUE DE CAPTURE D'ÉCRAN")
    print("=" * 40)
    print(f"⏰ Début du test: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    try:
        # Test direct de capture d'écran
        print("📦 Test de capture d'écran directe...")
        
        import cv2
        import numpy as np
        from PIL import ImageGrab
        
        # Capture d'écran complète
        start_time = time.time()
        screenshot = ImageGrab.grab()
        capture_time = time.time() - start_time
        
        # Conversion en numpy array
        screenshot_np = np.array(screenshot)
        screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        
        height, width = screenshot_bgr.shape[:2]
        total_pixels = height * width
        
        print(f"✅ Capture d'écran réussie: {width}x{height}")
        print(f"📊 Pixels totaux: {total_pixels:,}")
        print(f"⏱️  Temps capture: {capture_time:.3f}s")
        
        # Test OCR simple sur l'écran complet
        print(f"\n🔍 Test OCR sur l'écran complet...")
        try:
            import pytesseract
            
            # Conversion en niveaux de gris
            gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
            
            # OCR
            start_time = time.time()
            text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
            ocr_time = time.time() - start_time
            
            clean_text = text.strip()
            print(f"  Texte détecté (premiers 200 chars): '{clean_text[:200]}...'")
            print(f"  Longueur totale: {len(clean_text)} caractères")
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
        print(f"\n🎨 Test analyse couleur...")
        try:
            # Conversion HSV
            hsv = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2HSV)
            
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
            
            if red_pixels > total_pixels * 0.01:  # 1% de rouge
                print(f"  🔴 Rouge détecté (cartes ♥♦)")
            elif black_pixels > total_pixels * 0.01:  # 1% de noir
                print(f"  ⚫ Noir détecté (cartes ♠♣)")
            else:
                print(f"  ⚪ Pas de couleur dominante")
            
        except Exception as e:
            print(f"  ❌ Erreur analyse couleur: {e}")
        
        print(f"\n✅ TEST BASIQUE TERMINÉ!")
        print(f"⏰ Fin du test: {datetime.now().strftime('%H:%M:%S')}")
        
        # Recommandations
        print(f"\n💡 RECOMMANDATIONS:")
        print(f"   - Si vous voyez des cartes à l'écran, lancez le jeu")
        print(f"   - Assurez-vous que les cartes sont visibles")
        print(f"   - Puis relancez le test")
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_screen_capture() 