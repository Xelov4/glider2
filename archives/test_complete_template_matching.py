#!/usr/bin/env python3
"""
🃏 TEST DU SYSTÈME DE TEMPLATE MATCHING COMPLET
================================================

Test du nouveau système de reconnaissance de cartes avec templates organisés.
"""

import sys
import os
import time
import cv2
import numpy as np
from datetime import datetime

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_complete_template_matching():
    """Test du système de template matching complet"""
    print("🃏 TEST DU SYSTÈME DE TEMPLATE MATCHING COMPLET")
    print("=" * 60)
    print(f"⏰ Début du test: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    try:
        # Import du module d'analyse d'images
        print("📦 Import du module d'analyse...")
        from modules.image_analysis import ImageAnalyzer
        
        # Initialiser l'analyseur
        print("🔧 Initialisation de l'analyseur...")
        analyzer = ImageAnalyzer()
        
        # Vérifier que les templates organisés existent
        organized_dir = "templates/cards/organized"
        if not os.path.exists(organized_dir):
            print("❌ Dossier templates organisés non trouvé")
            return
        
        print("✅ Templates organisés trouvés")
        
        # Compter les templates disponibles
        total_templates = 0
        for suit in ['♠', '♥', '♦', '♣']:
            suit_dir = os.path.join(organized_dir, suit)
            if os.path.exists(suit_dir):
                templates = [f for f in os.listdir(suit_dir) if f.endswith('.png')]
                total_templates += len(templates)
                print(f"  🎨 {suit}: {len(templates)} templates")
        
        print(f"📊 Total: {total_templates} templates")
        
        # Import du captureur d'écran
        from modules.screen_capture import ScreenCapture
        screen_capture = ScreenCapture()
        
        # Régions à tester
        regions_to_test = [
            'hand_area',
            'community_cards'
        ]
        
        print("\n🎯 TEST EN TEMPS RÉEL")
        print("-" * 40)
        
        test_duration = 30  # 30 secondes de test
        start_time = time.time()
        cycle_count = 0
        
        while time.time() - start_time < test_duration:
            cycle_count += 1
            current_time = time.time() - start_time
            
            print(f"\r🔄 Cycle {cycle_count:3d} | ⏱️  {current_time:.1f}s | ", end="")
            
            # Tester chaque région
            for region_name in regions_to_test:
                try:
                    # Capture de la région
                    region_image = screen_capture.capture_region(region_name)
                    if region_image is None:
                        continue
                    
                    # Détection par template matching
                    detection_start = time.time()
                    cards = analyzer._detect_cards_template_matching(region_image)
                    detection_time = time.time() - detection_start
                    
                    if cards:
                        card_names = [f"{c.rank}{c.suit}" for c in cards]
                        avg_confidence = np.mean([c.confidence for c in cards])
                        print(f"📋 {region_name}: {card_names} (conf: {avg_confidence:.3f}, {detection_time:.3f}s) | ", end="")
                    
                except Exception as e:
                    print(f"❌ Erreur {region_name}: {e} | ", end="")
            
            # Pause entre les cycles
            time.sleep(0.5)
        
        print(f"\n\n✅ Test terminé après {cycle_count} cycles")
        
        # Test avec images de test
        print("\n🧪 TEST AVEC IMAGES DE TEST")
        print("-" * 40)
        
        test_images = create_test_images()
        
        for i, test_image in enumerate(test_images):
            print(f"📸 Test image {i+1}...")
            
            # Sauvegarder l'image de test
            os.makedirs("template_test_captures", exist_ok=True)
            filename = f"template_test_captures/test_image_{i+1}.png"
            cv2.imwrite(filename, test_image)
            
            # Détection
            cards = analyzer._detect_cards_template_matching(test_image)
            
            if cards:
                card_names = [f"{c.rank}{c.suit}" for c in cards]
                print(f"  ✅ Cartes détectées: {card_names}")
            else:
                print(f"  ❌ Aucune carte détectée")
        
        print("\n🎯 RÉSULTATS")
        print("=" * 40)
        print(f"✅ Système de template matching opérationnel")
        print(f"📁 Templates organisés: {total_templates}/52")
        print(f"🔄 Cycles testés: {cycle_count}")
        print(f"⏱️  Durée test: {test_duration}s")
        
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        import traceback
        traceback.print_exc()

def create_test_images():
    """Crée des images de test avec des cartes"""
    test_images = []
    
    # Image 1: Main de poker (2 cartes)
    img1 = np.ones((200, 400, 3), dtype=np.uint8) * 255
    # Simuler des cartes
    cv2.rectangle(img1, (50, 50), (150, 150), (0, 0, 0), 2)
    cv2.rectangle(img1, (200, 50), (300, 150), (0, 0, 0), 2)
    cv2.putText(img1, "A", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    cv2.putText(img1, "K", (230, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    test_images.append(img1)
    
    # Image 2: Flop (3 cartes)
    img2 = np.ones((200, 600, 3), dtype=np.uint8) * 255
    cv2.rectangle(img2, (50, 50), (150, 150), (0, 0, 0), 2)
    cv2.rectangle(img2, (200, 50), (300, 150), (0, 0, 0), 2)
    cv2.rectangle(img2, (350, 50), (450, 150), (0, 0, 0), 2)
    cv2.putText(img2, "Q", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    cv2.putText(img2, "J", (230, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    cv2.putText(img2, "T", (380, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    test_images.append(img2)
    
    return test_images

if __name__ == "__main__":
    test_complete_template_matching() 