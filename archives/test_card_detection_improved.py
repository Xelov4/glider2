#!/usr/bin/env python3
"""
🔍 TEST DE DÉTECTION DE CARTES AMÉLIORÉ
========================================

Ce script teste la nouvelle méthode de détection de cartes avec template matching.
"""

import sys
import os
import cv2
import numpy as np
import time
from typing import List, Dict

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.image_analysis import ImageAnalyzer
from modules.screen_capture import ScreenCapture

def test_improved_card_detection():
    """Test complet de la détection de cartes améliorée"""
    print("🔍 TEST DE DÉTECTION DE CARTES AMÉLIORÉ")
    print("=" * 60)
    
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
            print("-" * 40)
            
            try:
                # Capturer la région
                region_image = screen_capture.capture_region(region_name)
                
                if region_image is None:
                    print(f"❌ Impossible de capturer {region_name}")
                    continue
                
                print(f"✅ Image capturée: {region_image.shape}")
                
                # Test de la nouvelle détection par templates
                print("\n🔍 Test détection par templates...")
                start_time = time.time()
                
                template_cards = image_analyzer._detect_cards_by_templates(region_image)
                template_time = time.time() - start_time
                
                print(f"⏱️  Temps template matching: {template_time:.3f}s")
                print(f"🃏 Cartes détectées par templates: {[f'{c.rank}{c.suit}' for c in template_cards]}")
                
                # Test de la détection OCR classique
                print("\n🔍 Test détection OCR...")
                start_time = time.time()
                
                ocr_cards = image_analyzer._detect_cards_ocr_optimized(region_image)
                ocr_time = time.time() - start_time
                
                print(f"⏱️  Temps OCR: {ocr_time:.3f}s")
                print(f"🃏 Cartes détectées par OCR: {[f'{c.rank}{c.suit}' for c in ocr_cards]}")
                
                # Test de la détection complète
                print("\n🔍 Test détection complète...")
                start_time = time.time()
                
                all_cards = image_analyzer.detect_cards(region_image, region_name)
                total_time = time.time() - start_time
                
                print(f"⏱️  Temps total: {total_time:.3f}s")
                print(f"🃏 Cartes détectées au total: {[f'{c.rank}{c.suit}' for c in all_cards]}")
                
                # Debug détaillé
                print("\n🔍 Debug détaillé...")
                debug_info = image_analyzer.debug_card_detection(region_image, region_name)
                
                print(f"📊 Résultats OCR:")
                for result in debug_info['ocr_results']:
                    print(f"  {result['config']}: '{result['text']}' ({result['length']} chars)")
                
                if 'color_analysis' in debug_info:
                    color_info = debug_info['color_analysis']
                    print(f"🎨 Analyse couleur:")
                    print(f"  Pixels rouges: {color_info['red_pixels']}")
                    print(f"  Pixels noirs: {color_info['black_pixels']}")
                    print(f"  Total pixels: {color_info['total_pixels']}")
                
                # Test des contours
                print("\n🔍 Test détection contours...")
                if len(region_image.shape) == 3:
                    gray_image = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray_image = region_image.copy()
                
                contours = image_analyzer._find_card_contours(gray_image)
                print(f"📐 Contours de cartes trouvés: {len(contours)}")
                
                for i, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    print(f"  Contour {i+1}: aire={area:.0f}, taille={w}x{h}, ratio={aspect_ratio:.2f}")
                
            except Exception as e:
                print(f"❌ Erreur test {region_name}: {e}")
        
        print("\n✅ Test terminé!")
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")

def test_specific_card_detection():
    """Test de détection de cartes spécifiques"""
    print("\n🎯 TEST DE DÉTECTION DE CARTES SPÉCIFIQUES")
    print("=" * 50)
    
    image_analyzer = ImageAnalyzer()
    screen_capture = ScreenCapture()
    
    try:
        # Capturer hand_area
        hand_image = screen_capture.capture_region('hand_area')
        
        if hand_image is None:
            print("❌ Impossible de capturer hand_area")
            return
        
        print(f"✅ Image hand_area: {hand_image.shape}")
        
        # Test de détection de rangs spécifiques
        ranks_to_test = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7']
        
        for rank in ranks_to_test:
            print(f"\n🔍 Test détection rang: {rank}")
            
            # Détecter le rang dans l'image
            detected_rank = image_analyzer._detect_rank_in_region(hand_image)
            print(f"  Rang détecté: {detected_rank}")
            
            if detected_rank == rank:
                print(f"  ✅ Rang {rank} détecté correctement!")
            else:
                print(f"  ❌ Rang {rank} non détecté")
        
        # Test de détection de couleurs
        suits_to_test = ['♠', '♥', '♦', '♣']
        
        for suit in suits_to_test:
            print(f"\n🔍 Test détection couleur: {suit}")
            
            # Détecter la couleur dans l'image
            detected_suit = image_analyzer._detect_suit_in_region(hand_image)
            print(f"  Couleur détectée: {detected_suit}")
            
            if detected_suit == suit:
                print(f"  ✅ Couleur {suit} détectée correctement!")
            else:
                print(f"  ❌ Couleur {suit} non détectée")
        
    except Exception as e:
        print(f"❌ Erreur test spécifique: {e}")

if __name__ == "__main__":
    print("🚀 LANCEMENT DES TESTS DE DÉTECTION DE CARTES")
    print("=" * 60)
    
    # Test principal
    test_improved_card_detection()
    
    # Test spécifique
    test_specific_card_detection()
    
    print("\n🎉 TOUS LES TESTS TERMINÉS!") 