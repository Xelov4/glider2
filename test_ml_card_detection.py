#!/usr/bin/env python3
"""
🧠 TEST DE DÉTECTION DE CARTES PAR MACHINE LEARNING
===================================================

Test du nouveau système ML pour la reconnaissance de cartes.
"""

import sys
import os
import time
import cv2
import numpy as np
from datetime import datetime

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_ml_card_detection():
    """Test du système ML de détection de cartes"""
    print("🧠 TEST DE DÉTECTION DE CARTES PAR MACHINE LEARNING")
    print("=" * 60)
    print(f"⏰ Début du test: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    try:
        # Import du détecteur ML
        print("📦 Import du détecteur ML...")
        from modules.card_ml_detector import CardMLDetector
        
        # Initialiser le détecteur
        print("🔧 Initialisation du détecteur ML...")
        ml_detector = CardMLDetector()
        
        if not ml_detector.is_trained:
            print("❌ Modèle non entraîné")
            return
        
        print("✅ Modèle ML initialisé et entraîné")
        
        # Import du captureur d'écran
        from modules.screen_capture import ScreenCapture
        screen_capture = ScreenCapture()
        
        # Régions à tester
        regions_to_test = [
            'hand_area',
            'community_cards'
        ]
        
        for region_name in regions_to_test:
            print(f"\n🎯 Test ML de la région: {region_name}")
            print("-" * 40)
            
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
                total_pixels = height * width
                
                print(f"✅ Image capturée: {width}x{height}")
                print(f"📊 Pixels totaux: {total_pixels:,}")
                print(f"⏱️  Temps capture: {capture_time:.3f}s")
                
                # Test de détection ML
                print(f"🧠 Test détection ML...")
                start_time = time.time()
                
                ml_cards = ml_detector.detect_cards_ml(region_image)
                ml_time = time.time() - start_time
                
                print(f"⏱️  Temps ML: {ml_time:.3f}s")
                print(f"🃏 Cartes détectées par ML: {len(ml_cards)}")
                
                for i, card in enumerate(ml_cards):
                    print(f"  Carte {i+1}: {card.rank}{card.suit} (conf: {card.confidence:.2f})")
                
                # Test d'analyse de couleur ML
                print(f"🎨 Test analyse couleur ML...")
                if ml_cards:
                    for card in ml_cards:
                        # Analyser la couleur de la région
                        color_analysis = ml_detector._analyze_color_ml(region_image)
                        print(f"  Couleur dominante: {color_analysis['dominant_color']}")
                        print(f"  Ratio rouge: {color_analysis['red_ratio']:.2f}")
                        print(f"  Ratio noir: {color_analysis['black_ratio']:.2f}")
                else:
                    print("  Aucune carte détectée pour analyse couleur")
                
                # Comparaison avec OCR classique
                print(f"🔍 Comparaison avec OCR classique...")
                try:
                    import pytesseract
                    
                    # OCR classique
                    if len(region_image.shape) == 3:
                        gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = region_image.copy()
                    
                    start_time = time.time()
                    text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
                    ocr_time = time.time() - start_time
                    
                    clean_text = text.strip()
                    print(f"  Texte OCR: '{clean_text}'")
                    print(f"  Temps OCR: {ocr_time:.3f}s")
                    
                    # Chercher des cartes dans le texte
                    card_chars = []
                    for char in clean_text:
                        if char in '23456789TJQKA':
                            card_chars.append(char)
                    
                    print(f"  Rangs OCR: {card_chars}")
                    
                    # Comparaison
                    ml_ranks = [card.rank for card in ml_cards]
                    print(f"  Rangs ML: {ml_ranks}")
                    
                    if ml_ranks and card_chars:
                        print(f"  ✅ Correspondance trouvée!")
                    elif ml_ranks:
                        print(f"  🧠 ML détecte des cartes (OCR: non)")
                    elif card_chars:
                        print(f"  🔍 OCR détecte des cartes (ML: non)")
                    else:
                        print(f"  ❌ Aucune carte détectée")
                    
                except Exception as e:
                    print(f"  ❌ Erreur OCR: {e}")
                
            except Exception as e:
                print(f"❌ Erreur test {region_name}: {e}")
        
        # Test de performance
        print(f"\n⚡ TEST DE PERFORMANCE")
        print("-" * 25)
        
        # Test sur plusieurs images
        test_images = []
        for region_name in regions_to_test:
            image = screen_capture.capture_region(region_name)
            if image is not None:
                test_images.append(image)
        
        if test_images:
            # Test de vitesse ML
            start_time = time.time()
            total_cards = 0
            
            for image in test_images:
                cards = ml_detector.detect_cards_ml(image)
                total_cards += len(cards)
            
            total_time = time.time() - start_time
            avg_time = total_time / len(test_images)
            
            print(f"📊 Performance ML:")
            print(f"  Images testées: {len(test_images)}")
            print(f"  Temps total: {total_time:.3f}s")
            print(f"  Temps moyen par image: {avg_time:.3f}s")
            print(f"  Cartes détectées total: {total_cards}")
            print(f"  Cartes par image: {total_cards/len(test_images):.1f}")
        
        print(f"\n✅ TEST ML TERMINÉ!")
        print(f"⏰ Fin du test: {datetime.now().strftime('%H:%M:%S')}")
        
        # Recommandations
        print(f"\n💡 RECOMMANDATIONS:")
        if total_cards > 0:
            print(f"  ✅ Le système ML fonctionne!")
            print(f"  🚀 Prêt pour intégration dans le bot")
        else:
            print(f"  ⚠️  Aucune carte détectée")
            print(f"  🔧 Vérifiez que vous êtes en train de jouer")
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ml_card_detection() 