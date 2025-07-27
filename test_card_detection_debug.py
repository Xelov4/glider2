#!/usr/bin/env python3
"""
🔍 Script de Debug - Détection de Cartes
========================================

Ce script teste et analyse la détection de cartes pour identifier les problèmes
et améliorer la reconnaissance.

VERSION: 1.0.0
"""

import cv2
import numpy as np
import logging
import sys
import os

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.image_analysis import ImageAnalyzer
from modules.screen_capture import ScreenCapture

def setup_logging():
    """Configure le logging pour le debug"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_card_detection():
    """Test complet de la détection de cartes"""
    print("🔍 TEST DE DÉTECTION DE CARTES")
    print("=" * 50)
    
    # Initialiser les modules
    image_analyzer = ImageAnalyzer()
    screen_capture = ScreenCapture()
    
    try:
        # Capturer les régions de cartes
        print("\n📸 Capture des régions...")
        
        # Régions à tester
        regions_to_test = [
            'hand_area',
            'community_cards'
        ]
        
        for region_name in regions_to_test:
            print(f"\n🎯 Test de la région: {region_name}")
            print("-" * 30)
            
            try:
                # Capturer la région
                region_image = screen_capture.capture_region(region_name)
                
                if region_image is None:
                    print(f"❌ Impossible de capturer {region_name}")
                    continue
                
                print(f"✅ Image capturée: {region_image.shape}")
                
                # Debug de la détection
                debug_info = image_analyzer.debug_card_detection(region_image, region_name)
                
                # Afficher les résultats
                print(f"📊 Résultats OCR:")
                for result in debug_info['ocr_results']:
                    print(f"  {result['config']}: '{result['text']}' ({result['length']} chars)")
                
                print(f"🃏 Cartes détectées: {debug_info['detected_cards']}")
                
                if 'color_analysis' in debug_info:
                    color_info = debug_info['color_analysis']
                    print(f"🎨 Analyse couleur:")
                    print(f"  Pixels rouges: {color_info['red_pixels']}")
                    print(f"  Pixels noirs: {color_info['black_pixels']}")
                    print(f"  Total pixels: {color_info['total_pixels']}")
                
                if debug_info['errors']:
                    print(f"❌ Erreurs: {debug_info['errors']}")
                
                # Test de détection simple
                cards = image_analyzer.detect_cards(region_image, region_name)
                print(f"🎯 Détection finale: {[f'{c.rank}{c.suit}' for c in cards]}")
                
            except Exception as e:
                print(f"❌ Erreur test {region_name}: {e}")
        
        print("\n✅ Test terminé!")
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")

def test_ocr_configurations():
    """Test des différentes configurations OCR"""
    print("\n🔧 TEST DES CONFIGURATIONS OCR")
    print("=" * 50)
    
    # Import manquant
    import pytesseract
    
    screen_capture = ScreenCapture()
    image_analyzer = ImageAnalyzer()
    
    # Capturer une région pour tester
    test_image = screen_capture.capture_region('hand_area')
    if test_image is None:
        print("❌ Impossible de capturer l'image de test")
        return
    
    # Configurations à tester
    configs = [
        ('Standard', ''),
        ('Dense', '--oem 1 --psm 6'),
        ('Single char', '--oem 1 --psm 8'),
        ('Cards only', '--oem 1 --psm 6 -c tessedit_char_whitelist=23456789TJQKA♠♥♦♣'),
        ('Cards dense', '--oem 1 --psm 8 -c tessedit_char_whitelist=23456789TJQKA♠♥♦♣'),
        ('Cards single', '--oem 1 --psm 10 -c tessedit_char_whitelist=23456789TJQKA♠♥♦♣')
    ]
    
    for name, config in configs:
        try:
            text = pytesseract.image_to_string(test_image, config=config).strip()
            print(f"{name:<15} : ✅ '{text}' ({len(text)} chars)")
        except Exception as e:
            print(f"{name:<15} : ❌ {e}")

if __name__ == "__main__":
    setup_logging()
    
    print("🎯 DÉMARRAGE DU DEBUG DE DÉTECTION DE CARTES")
    print("=" * 60)
    
    # Test principal
    test_card_detection()
    
    # Test des configurations OCR
    test_ocr_configurations()
    
    print("\n🏁 DEBUG TERMINÉ") 