#!/usr/bin/env python3
"""
🔧 TEST COMPLET SYSTÈME HYBRIDE + ML
=====================================

Test long et complet du système hybride de capture + ML
pour évaluer les performances en conditions réelles.
"""

import sys
import os
import time
import cv2
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hybrid_ml_test.log'),
        logging.StreamHandler()
    ]
)

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_hybrid_ml_comprehensive():
    """Test complet du système hybride + ML"""
    print("🔧 TEST COMPLET SYSTÈME HYBRIDE + ML")
    print("=" * 60)
    print(f"⏰ Début: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    try:
        # Import des modules
        from modules.screen_capture import ScreenCapture
        from modules.image_analysis import ImageAnalyzer
        from modules.card_ml_detector import CardMLDetector
        
        # Initialisation
        screen_capture = ScreenCapture()
        image_analyzer = ImageAnalyzer()
        ml_detector = CardMLDetector()
        
        # Configuration du test
        test_duration = 300  # 5 minutes
        capture_interval = 2.0  # 2 secondes entre captures
        regions_to_test = ['hand_area', 'community_cards', 'pot_area', 'my_stack_area']
        
        # Métriques de performance
        metrics = {
            'total_captures': 0,
            'ml_detections': 0,
            'ocr_detections': 0,
            'fast_detections': 0,
            'contour_detections': 0,
            'color_detections': 0,
            'no_detections': 0,
            'capture_times': [],
            'detection_times': [],
            'ml_times': [],
            'ocr_times': [],
            'total_cards_detected': 0,
            'hand_cards_history': [],
            'community_cards_history': [],
            'confidence_scores': [],
            'image_qualities': []
        }
        
        print("🎯 Configuration du test:")
        print(f"  ⏱️  Durée: {test_duration} secondes")
        print(f"  📸 Intervalle: {capture_interval} secondes")
        print(f"  🎯 Régions: {regions_to_test}")
        print(f"  🤖 ML entraîné: {ml_detector.is_trained}")
        print()
        
        # Début du test
        start_time = time.time()
        last_capture_time = 0
        
        print("🚀 DÉBUT DU TEST - Capture en cours...")
        print("-" * 50)
        
        while time.time() - start_time < test_duration:
            current_time = time.time()
            
            # Capture toutes les 2 secondes
            if current_time - last_capture_time >= capture_interval:
                last_capture_time = current_time
                metrics['total_captures'] += 1
                
                print(f"\n📸 Capture #{metrics['total_captures']} - {datetime.now().strftime('%H:%M:%S')}")
                print("-" * 40)
                
                # Test de chaque région
                for region_name in regions_to_test:
                    try:
                        # 1. CAPTURE HYBRIDE
                        capture_start = time.time()
                        image = screen_capture.capture_region(region_name)
                        capture_time = time.time() - capture_start
                        metrics['capture_times'].append(capture_time)
                        
                        if image is not None:
                            height, width = image.shape[:2]
                            total_pixels = height * width
                            
                            print(f"  📸 {region_name}: {width}x{height} ({total_pixels:,} pixels)")
                            print(f"  ⏱️  Capture: {capture_time:.3f}s")
                            
                            # 2. DÉTECTION DE CARTES
                            if region_name in ['hand_area', 'community_cards']:
                                detection_start = time.time()
                                cards = image_analyzer.detect_cards(image, region_name)
                                detection_time = time.time() - detection_start
                                metrics['detection_times'].append(detection_time)
                                
                                print(f"  🃏 Cartes détectées: {len(cards)}")
                                
                                if cards:
                                    # Analyser les cartes détectées
                                    card_list = []
                                    for card in cards:
                                        card_info = f"{card.rank}{card.suit}"
                                        card_list.append(card_info)
                                        metrics['confidence_scores'].append(card.confidence)
                                        metrics['total_cards_detected'] += 1
                                    
                                    print(f"  🎯 Cartes: {', '.join(card_list)}")
                                    
                                    # Historique des cartes
                                    if region_name == 'hand_area':
                                        metrics['hand_cards_history'].append({
                                            'timestamp': datetime.now(),
                                            'cards': card_list,
                                            'count': len(cards)
                                        })
                                    elif region_name == 'community_cards':
                                        metrics['community_cards_history'].append({
                                            'timestamp': datetime.now(),
                                            'cards': card_list,
                                            'count': len(cards)
                                        })
                                    
                                    # Compter les méthodes de détection
                                    if len(cards) > 0:
                                        if card.confidence > 0.7:  # ML probable
                                            metrics['ml_detections'] += 1
                                        elif card.confidence > 0.5:  # OCR probable
                                            metrics['ocr_detections'] += 1
                                        elif card.confidence > 0.3:  # Fast probable
                                            metrics['fast_detections'] += 1
                                        else:  # Fallback
                                            metrics['contour_detections'] += 1
                                else:
                                    print(f"  ❌ Aucune carte détectée")
                                    metrics['no_detections'] += 1
                                
                                print(f"  ⏱️  Détection: {detection_time:.3f}s")
                            
                            # 3. ANALYSE DE QUALITÉ
                            if image is not None:
                                quality_score = calculate_image_quality(image)
                                metrics['image_qualities'].append(quality_score)
                                print(f"  📊 Qualité: {quality_score:.2f}")
                        
                    except Exception as e:
                        print(f"  ❌ Erreur {region_name}: {e}")
                
                # Affichage des statistiques en cours
                if metrics['total_captures'] % 10 == 0:
                    print(f"\n📊 STATISTIQUES INTERMÉDIAIRES:")
                    print(f"  📸 Captures: {metrics['total_captures']}")
                    print(f"  🃏 Cartes totales: {metrics['total_cards_detected']}")
                    print(f"  🤖 ML: {metrics['ml_detections']}")
                    print(f"  📝 OCR: {metrics['ocr_detections']}")
                    print(f"  ⚡ Fast: {metrics['fast_detections']}")
                    print(f"  🎯 Contours: {metrics['contour_detections']}")
                    print(f"  ❌ Échecs: {metrics['no_detections']}")
                    
                    if metrics['capture_times']:
                        avg_capture = np.mean(metrics['capture_times'])
                        print(f"  ⏱️  Temps capture moyen: {avg_capture:.3f}s")
                    
                    if metrics['detection_times']:
                        avg_detection = np.mean(metrics['detection_times'])
                        print(f"  ⏱️  Temps détection moyen: {avg_detection:.3f}s")
                    
                    if metrics['image_qualities']:
                        avg_quality = np.mean(metrics['image_qualities'])
                        print(f"  📊 Qualité moyenne: {avg_quality:.2f}")
            
            # Pause entre captures
            time.sleep(0.1)
        
        # RÉSULTATS FINAUX
        print(f"\n🎉 TEST TERMINÉ!")
        print("=" * 60)
        print(f"⏰ Durée totale: {test_duration} secondes")
        print(f"📸 Captures totales: {metrics['total_captures']}")
        print(f"🃏 Cartes détectées: {metrics['total_cards_detected']}")
        print()
        
        print("📊 RÉPARTITION DES DÉTECTIONS:")
        print(f"  🤖 Machine Learning: {metrics['ml_detections']} ({metrics['ml_detections']/max(metrics['total_captures'],1)*100:.1f}%)")
        print(f"  📝 OCR: {metrics['ocr_detections']} ({metrics['ocr_detections']/max(metrics['total_captures'],1)*100:.1f}%)")
        print(f"  ⚡ Fast: {metrics['fast_detections']} ({metrics['fast_detections']/max(metrics['total_captures'],1)*100:.1f}%)")
        print(f"  🎯 Contours: {metrics['contour_detections']} ({metrics['contour_detections']/max(metrics['total_captures'],1)*100:.1f}%)")
        print(f"  ❌ Échecs: {metrics['no_detections']} ({metrics['no_detections']/max(metrics['total_captures'],1)*100:.1f}%)")
        print()
        
        print("⏱️  PERFORMANCE:")
        if metrics['capture_times']:
            avg_capture = np.mean(metrics['capture_times'])
            min_capture = np.min(metrics['capture_times'])
            max_capture = np.max(metrics['capture_times'])
            print(f"  📸 Capture: {avg_capture:.3f}s (min: {min_capture:.3f}s, max: {max_capture:.3f}s)")
        
        if metrics['detection_times']:
            avg_detection = np.mean(metrics['detection_times'])
            min_detection = np.min(metrics['detection_times'])
            max_detection = np.max(metrics['detection_times'])
            print(f"  🃏 Détection: {avg_detection:.3f}s (min: {min_detection:.3f}s, max: {max_detection:.3f}s)")
        
        if metrics['confidence_scores']:
            avg_confidence = np.mean(metrics['confidence_scores'])
            print(f"  🎯 Confiance moyenne: {avg_confidence:.2f}")
        
        if metrics['image_qualities']:
            avg_quality = np.mean(metrics['image_qualities'])
            print(f"  📊 Qualité moyenne: {avg_quality:.2f}")
        print()
        
        print("📈 HISTORIQUE DES CARTES:")
        if metrics['hand_cards_history']:
            print(f"  🃏 Main: {len(metrics['hand_cards_history'])} détections")
            for i, hist in enumerate(metrics['hand_cards_history'][-5:]):  # 5 dernières
                print(f"    {hist['timestamp'].strftime('%H:%M:%S')}: {hist['count']} cartes - {', '.join(hist['cards'])}")
        
        if metrics['community_cards_history']:
            print(f"  🃏 Communautaires: {len(metrics['community_cards_history'])} détections")
            for i, hist in enumerate(metrics['community_cards_history'][-5:]):  # 5 dernières
                print(f"    {hist['timestamp'].strftime('%H:%M:%S')}: {hist['count']} cartes - {', '.join(hist['cards'])}")
        print()
        
        # ÉVALUATION FINALE
        print("💡 ÉVALUATION FINALE:")
        success_rate = (metrics['total_captures'] - metrics['no_detections']) / max(metrics['total_captures'], 1) * 100
        if success_rate > 80:
            print(f"  ✅ Taux de succès excellent: {success_rate:.1f}%")
        elif success_rate > 60:
            print(f"  ⚠️  Taux de succès acceptable: {success_rate:.1f}%")
        else:
            print(f"  ❌ Taux de succès faible: {success_rate:.1f}%")
        
        if avg_capture < 0.5:
            print(f"  ✅ Capture rapide")
        else:
            print(f"  ⚠️  Capture lente")
        
        if avg_detection < 0.3:
            print(f"  ✅ Détection rapide")
        else:
            print(f"  ⚠️  Détection lente")
        
        print(f"\n✅ TEST TERMINÉ!")
        print(f"⏰ Fin: {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        import traceback
        traceback.print_exc()

def calculate_image_quality(image: np.ndarray) -> float:
    """Calcule un score de qualité d'image"""
    try:
        score = 0.0
        
        # Résolution
        height, width = image.shape[:2]
        resolution_score = min(width * height / 10000, 1.0)
        score += resolution_score * 0.4
        
        # Contraste
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        contrast_score = np.std(gray) / 255.0
        score += contrast_score * 0.3
        
        # Netteté
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        sharpness_score = np.mean(gradient_magnitude) / 255.0
        score += sharpness_score * 0.3
        
        return min(score, 1.0)
        
    except Exception as e:
        return 0.5

if __name__ == "__main__":
    test_hybrid_ml_comprehensive() 