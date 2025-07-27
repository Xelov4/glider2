#!/usr/bin/env python3
"""
🔧 TEST SYSTÈME HYBRIDE DE CAPTURE
===================================

Test du système hybride : Capture maximale + Post-traitement intelligent
"""

import sys
import os
import time
import cv2
import numpy as np
from datetime import datetime

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_hybrid_capture():
    """Test du système hybride de capture"""
    print("🔧 TEST SYSTÈME HYBRIDE DE CAPTURE")
    print("=" * 50)
    print(f"⏰ Début: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    try:
        # Import du système hybride
        from modules.hybrid_capture_system import HybridCaptureSystem, CaptureConfig
        
        # Configuration optimisée
        config = CaptureConfig(
            capture_method="max_quality",
            post_processing=True,
            upscale_factor=3.0,
            enhancement_strength=1.8,
            noise_reduction=True,
            sharpening=True,
            contrast_boost=True
        )
        
        hybrid_capture = HybridCaptureSystem(config)
        
        # Régions à tester
        regions_to_test = ['hand_area', 'community_cards']
        
        for region_name in regions_to_test:
            print(f"\n🎯 Test de la région: {region_name}")
            print("-" * 40)
            
            # Test capture hybride
            print("🔧 Capture hybride (max qualité + post-traitement)...")
            start_time = time.time()
            
            image = hybrid_capture.capture_with_max_quality(region_name)
            capture_time = time.time() - start_time
            
            if image is not None:
                height, width = image.shape[:2]
                total_pixels = height * width
                
                print(f"✅ Image capturée: {width}x{height}")
                print(f"📊 Pixels totaux: {total_pixels:,}")
                print(f"⏱️  Temps capture: {capture_time:.3f}s")
                
                # Métriques de performance
                metrics = hybrid_capture.get_performance_metrics()
                print(f"📈 Amélioration moyenne: +{metrics['avg_quality_improvement']:.1f}%")
                print(f"⏱️  Temps post-traitement: {metrics['avg_post_processing_time']:.3f}s")
                
                # Test de détection de cartes
                print("🃏 Test détection de cartes...")
                try:
                    from modules.image_analysis import ImageAnalyzer
                    analyzer = ImageAnalyzer()
                    cards = analyzer.detect_cards(image, region_name)
                    
                    print(f"  🃏 Cartes détectées: {len(cards)}")
                    for card in cards:
                        print(f"    {card.rank}{card.suit} (conf: {card.confidence:.2f})")
                        
                except Exception as e:
                    print(f"  ❌ Erreur détection: {e}")
                
                # Sauvegarder l'image pour inspection
                try:
                    os.makedirs("hybrid_captures", exist_ok=True)
                    filename = f"hybrid_captures/{region_name}_hybrid.png"
                    cv2.imwrite(filename, image)
                    print(f"💾 Image sauvegardée: {filename}")
                except Exception as e:
                    print(f"❌ Erreur sauvegarde: {e}")
                    
            else:
                print("❌ Échec capture hybride")
        
        # Résultats finaux
        print(f"\n📊 RÉSULTATS FINAUX")
        print("-" * 25)
        
        final_metrics = hybrid_capture.get_performance_metrics()
        print(f"📸 Captures totales: {final_metrics['total_captures']}")
        print(f"📈 Amélioration moyenne: +{final_metrics['avg_quality_improvement']:.1f}%")
        print(f"⏱️  Temps capture moyen: {final_metrics['avg_capture_time']:.3f}s")
        print(f"⏱️  Temps post-traitement: {final_metrics['avg_post_processing_time']:.3f}s")
        
        # Évaluation
        print(f"\n💡 ÉVALUATION:")
        if final_metrics['avg_quality_improvement'] > 50:
            print(f"  ✅ Amélioration excellente")
        elif final_metrics['avg_quality_improvement'] > 20:
            print(f"  ⚠️  Amélioration acceptable")
        else:
            print(f"  ❌ Amélioration insuffisante")
        
        if final_metrics['avg_post_processing_time'] < 0.5:
            print(f"  ✅ Post-traitement rapide")
        else:
            print(f"  ⚠️  Post-traitement lent")
        
        print(f"\n✅ TEST TERMINÉ!")
        print(f"⏰ Fin: {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hybrid_capture() 