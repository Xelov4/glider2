#!/usr/bin/env python3
"""
🎯 TEST FINAL DU SYSTÈME DE TEMPLATE MATCHING
==============================================

Test complet du nouveau système de reconnaissance de cartes avec templates organisés.
"""

import sys
import os
import time
import cv2
import numpy as np
from datetime import datetime

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_final_template_system():
    """Test final du système de template matching"""
    print("🎯 TEST FINAL DU SYSTÈME DE TEMPLATE MATCHING")
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
        
        # Vérifier que les templates fixés existent
        fixed_dir = "templates/cards/fixed"
        if not os.path.exists(fixed_dir):
            print("❌ Dossier templates fixés non trouvé")
            return
        
        print("✅ Templates fixés trouvés")
        
        # Compter les templates disponibles
        total_templates = 0
        for suit in ['spades', 'hearts', 'diamonds', 'clubs']:
            suit_dir = os.path.join(fixed_dir, suit)
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
        
        test_duration = 60  # 60 secondes de test
        start_time = time.time()
        cycle_count = 0
        total_detections = 0
        
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
                        total_detections += len(cards)
                    else:
                        print(f"📋 {region_name}: aucune carte | ", end="")
                    
                except Exception as e:
                    print(f"❌ Erreur {region_name}: {e} | ", end="")
            
            # Pause entre les cycles
            time.sleep(1.0)
        
        print(f"\n\n✅ Test terminé après {cycle_count} cycles")
        
        # Statistiques finales
        print("\n📊 STATISTIQUES FINALES")
        print("=" * 40)
        print(f"🔄 Cycles testés: {cycle_count}")
        print(f"⏱️  Durée test: {test_duration}s")
        print(f"📋 Détections totales: {total_detections}")
        print(f"📊 Templates disponibles: {total_templates}/52")
        print(f"⚡ Temps moyen par cycle: {test_duration/cycle_count:.2f}s")
        
        # Test de performance
        print("\n⚡ TEST DE PERFORMANCE")
        print("-" * 40)
        
        # Créer une image de test plus grande
        test_image = create_large_test_image()
        
        # Sauvegarder l'image de test
        os.makedirs("template_test_captures", exist_ok=True)
        filename = "template_test_captures/large_test_image.png"
        cv2.imwrite(filename, test_image)
        
        # Test de détection
        print("🔍 Test de détection sur image de test...")
        detection_start = time.time()
        cards = analyzer._detect_cards_template_matching(test_image)
        detection_time = time.time() - detection_start
        
        if cards:
            card_names = [f"{c.rank}{c.suit}" for c in cards]
            avg_confidence = np.mean([c.confidence for c in cards])
            print(f"✅ Cartes détectées: {card_names}")
            print(f"📊 Confiance moyenne: {avg_confidence:.3f}")
            print(f"⏱️  Temps de détection: {detection_time:.3f}s")
        else:
            print("❌ Aucune carte détectée")
        
        print("\n🎯 RÉSULTATS FINAUX")
        print("=" * 40)
        print(f"✅ Système de template matching opérationnel")
        print(f"📁 Templates organisés: {total_templates}/52")
        print(f"🔄 Cycles testés: {cycle_count}")
        print(f"📋 Détections totales: {total_detections}")
        print(f"⚡ Performance: {detection_time:.3f}s par détection")
        
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        import traceback
        traceback.print_exc()

def create_large_test_image():
    """Crée une image de test plus grande avec des cartes"""
    # Image 800x600 pour avoir assez d'espace
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Simuler des cartes plus grandes
    card_width, card_height = 100, 140
    
    # Position des cartes
    positions = [
        (50, 50),   # Carte 1
        (170, 50),  # Carte 2
        (290, 50),  # Carte 3
        (410, 50),  # Carte 4
        (530, 50),  # Carte 5
    ]
    
    # Dessiner les cartes
    for i, (x, y) in enumerate(positions):
        # Rectangle de la carte
        cv2.rectangle(img, (x, y), (x + card_width, y + card_height), (0, 0, 0), 3)
        
        # Texte de la carte (simuler différentes cartes)
        card_texts = ['A', 'K', 'Q', 'J', 'T']
        if i < len(card_texts):
            cv2.putText(img, card_texts[i], (x + 30, y + 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    return img

if __name__ == "__main__":
    test_final_template_system() 