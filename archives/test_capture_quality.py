#!/usr/bin/env python3
"""
📸 TEST DE QUALITÉ DE CAPTURE
==============================

Test pour comparer différentes méthodes de capture et optimiser la qualité
pour la détection de cartes.
"""

import sys
import os
import time
import cv2
import numpy as np
from datetime import datetime
# import matplotlib.pyplot as plt  # Optionnel pour visualisation
from typing import List

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_capture_quality():
    """Test de qualité de capture"""
    print("📸 TEST DE QUALITÉ DE CAPTURE")
    print("=" * 50)
    print(f"⏰ Début: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    try:
        # Import des modules
        from modules.screen_capture import ScreenCapture
        from modules.high_quality_capture import HighQualityCapture, CaptureConfig
        
        # Initialiser les capteurs
        standard_capture = ScreenCapture()
        
        # Configurations haute qualité
        config_mss = CaptureConfig(
            resolution_scale=2.0,
            quality_enhancement=True,
            sharpening=True,
            contrast_enhancement=True,
            noise_reduction=True,
            capture_method="mss"
        )
        
        config_pil = CaptureConfig(
            resolution_scale=2.0,
            quality_enhancement=True,
            sharpening=True,
            contrast_enhancement=True,
            noise_reduction=True,
            capture_method="pil"
        )
        
        hq_capture_mss = HighQualityCapture(config_mss)
        hq_capture_pil = HighQualityCapture(config_pil)
        
        # Régions à tester
        regions_to_test = ['hand_area', 'community_cards']
        
        for region_name in regions_to_test:
            print(f"\n🎯 Test de la région: {region_name}")
            print("-" * 40)
            
            # 1. CAPTURE STANDARD
            print("📸 Capture standard (pyautogui)...")
            start_time = time.time()
            standard_image = standard_capture.capture_region(region_name)
            standard_time = time.time() - start_time
            
            if standard_image is not None:
                print(f"  ✅ Résolution: {standard_image.shape}")
                print(f"  ⏱️  Temps: {standard_time:.3f}s")
                
                # Métriques de qualité standard
                standard_quality = calculate_image_quality(standard_image)
                print(f"  📊 Qualité: {standard_quality:.2f}")
            else:
                print("  ❌ Échec capture standard")
                continue
            
            # 2. CAPTURE HAUTE QUALITÉ MSS
            print("📸 Capture haute qualité (MSS)...")
            start_time = time.time()
            hq_mss_image = hq_capture_mss.capture_region_high_quality(region_name)
            hq_mss_time = time.time() - start_time
            
            if hq_mss_image is not None:
                print(f"  ✅ Résolution: {hq_mss_image.shape}")
                print(f"  ⏱️  Temps: {hq_mss_time:.3f}s")
                
                # Métriques de qualité MSS
                hq_mss_quality = calculate_image_quality(hq_mss_image)
                print(f"  📊 Qualité: {hq_mss_quality:.2f}")
                
                # Amélioration vs standard
                improvement = ((hq_mss_quality - standard_quality) / standard_quality) * 100
                print(f"  🚀 Amélioration: {improvement:+.1f}%")
            else:
                print("  ❌ Échec capture MSS")
            
            # 3. CAPTURE HAUTE QUALITÉ PIL
            print("📸 Capture haute qualité (PIL)...")
            start_time = time.time()
            hq_pil_image = hq_capture_pil.capture_region_high_quality(region_name)
            hq_pil_time = time.time() - start_time
            
            if hq_pil_image is not None:
                print(f"  ✅ Résolution: {hq_pil_image.shape}")
                print(f"  ⏱️  Temps: {hq_pil_time:.3f}s")
                
                # Métriques de qualité PIL
                hq_pil_quality = calculate_image_quality(hq_pil_image)
                print(f"  📊 Qualité: {hq_pil_quality:.2f}")
                
                # Amélioration vs standard
                improvement = ((hq_pil_quality - standard_quality) / standard_quality) * 100
                print(f"  🚀 Amélioration: {improvement:+.1f}%")
            else:
                print("  ❌ Échec capture PIL")
            
            # 4. OPTIMISATION SPÉCIFIQUE CARTES
            print("🎯 Optimisation spécifique cartes...")
            start_time = time.time()
            optimized_image = hq_capture_mss.optimize_for_card_detection(region_name)
            optimized_time = time.time() - start_time
            
            if optimized_image is not None:
                print(f"  ✅ Résolution: {optimized_image.shape}")
                print(f"  ⏱️  Temps: {optimized_time:.3f}s")
                
                # Métriques de qualité optimisée
                optimized_quality = calculate_image_quality(optimized_image)
                print(f"  📊 Qualité: {optimized_quality:.2f}")
                
                # Amélioration vs standard
                improvement = ((optimized_quality - standard_quality) / standard_quality) * 100
                print(f"  🚀 Amélioration: {improvement:+.1f}%")
            else:
                print("  ❌ Échec optimisation")
            
            # 5. TEST DE DÉTECTION DE CARTES
            print("🃏 Test détection de cartes...")
            
            # Test avec image standard
            if standard_image is not None:
                standard_cards = test_card_detection(standard_image, "Standard")
            
            # Test avec image haute qualité
            if hq_mss_image is not None:
                hq_cards = test_card_detection(hq_mss_image, "Haute Qualité MSS")
            
            # Test avec image optimisée
            if optimized_image is not None:
                optimized_cards = test_card_detection(optimized_image, "Optimisée")
            
            # 6. SAUVEGARDE DES IMAGES POUR COMPARAISON
            save_comparison_images(region_name, standard_image, hq_mss_image, optimized_image)
        
        # Métriques finales
        print(f"\n📊 MÉTRIQUES FINALES")
        print("-" * 25)
        
        mss_metrics = hq_capture_mss.get_quality_metrics()
        pil_metrics = hq_capture_pil.get_quality_metrics()
        
        print(f"📸 MSS - Captures: {mss_metrics['total_captures']}")
        print(f"📊 Qualité moyenne: {mss_metrics['avg_quality_score']:.2f}")
        print(f"⏱️  Temps moyen: {mss_metrics['avg_capture_time']:.3f}s")
        
        print(f"\n📸 PIL - Captures: {pil_metrics['total_captures']}")
        print(f"📊 Qualité moyenne: {pil_metrics['avg_quality_score']:.2f}")
        print(f"⏱️  Temps moyen: {pil_metrics['avg_capture_time']:.3f}s")
        
        print(f"\n✅ TEST TERMINÉ!")
        print(f"⏰ Fin: {datetime.now().strftime('%H:%M:%S')}")
        
        # Recommandations
        print(f"\n💡 RECOMMANDATIONS:")
        if mss_metrics['avg_quality_score'] > pil_metrics['avg_quality_score']:
            print(f"  🏆 MSS est la meilleure méthode")
        else:
            print(f"  🏆 PIL est la meilleure méthode")
        
        if mss_metrics['avg_quality_score'] > 0.7:
            print(f"  ✅ Qualité excellente pour la détection")
        elif mss_metrics['avg_quality_score'] > 0.5:
            print(f"  ⚠️  Qualité acceptable")
        else:
            print(f"  ❌ Qualité insuffisante")
        
    except Exception as e:
        print(f"❌ Erreur test qualité: {e}")
        import traceback
        traceback.print_exc()

def calculate_image_quality(image: np.ndarray) -> float:
    """Calcule un score de qualité d'image"""
    try:
        score = 0.0
        
        # 1. Résolution
        height, width = image.shape[:2]
        resolution_score = min(width * height / 10000, 1.0)
        score += resolution_score * 0.3
        
        # 2. Contraste
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        contrast_score = np.std(gray) / 255.0
        score += contrast_score * 0.3
        
        # 3. Netteté
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        sharpness_score = np.mean(gradient_magnitude) / 255.0
        score += sharpness_score * 0.2
        
        # 4. Bruit (inverse)
        noise_score = 1.0 - (np.std(gray) / np.mean(gray)) if np.mean(gray) > 0 else 0.0
        score += noise_score * 0.2
        
        return min(score, 1.0)
        
    except Exception as e:
        print(f"Erreur calcul qualité: {e}")
        return 0.5

def test_card_detection(image: np.ndarray, method_name: str) -> List:
    """Test de détection de cartes sur une image"""
    try:
        from modules.image_analysis import ImageAnalyzer
        
        analyzer = ImageAnalyzer()
        cards = analyzer.detect_cards(image, "test")
        
        print(f"  🃏 {method_name}: {len(cards)} cartes détectées")
        for card in cards:
            print(f"    {card.rank}{card.suit} (conf: {card.confidence:.2f})")
        
        return cards
        
    except Exception as e:
        print(f"  ❌ Erreur détection {method_name}: {e}")
        return []

def save_comparison_images(region_name: str, standard: np.ndarray, hq_mss: np.ndarray, optimized: np.ndarray):
    """Sauvegarde les images pour comparaison"""
    try:
        # Créer le dossier de comparaison
        comparison_dir = "capture_comparison"
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Sauvegarder les images
        if standard is not None:
            cv2.imwrite(f"{comparison_dir}/{region_name}_standard.png", standard)
        
        if hq_mss is not None:
            cv2.imwrite(f"{comparison_dir}/{region_name}_hq_mss.png", hq_mss)
        
        if optimized is not None:
            cv2.imwrite(f"{comparison_dir}/{region_name}_optimized.png", optimized)
        
        print(f"  💾 Images sauvegardées dans {comparison_dir}/")
        
    except Exception as e:
        print(f"  ❌ Erreur sauvegarde: {e}")

if __name__ == "__main__":
    test_capture_quality() 