#!/usr/bin/env python3
"""
🔍 TEST ULTRA-COMPLET DE DÉTECTION DE CARTES
============================================

Ce script fait un test très détaillé et complet de la détection de cartes.
Il analyse tous les aspects : OCR, couleurs, contours, templates, etc.
"""

import sys
import os
import cv2
import numpy as np
import time
import json
from typing import List, Dict, Tuple
from datetime import datetime

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.image_analysis import ImageAnalyzer
from modules.screen_capture import ScreenCapture

def comprehensive_card_detection_test():
    """Test ultra-complet de la détection de cartes"""
    print("🔍 TEST ULTRA-COMPLET DE DÉTECTION DE CARTES")
    print("=" * 80)
    print(f"⏰ Début du test: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Initialiser les modules
    image_analyzer = ImageAnalyzer()
    screen_capture = ScreenCapture()
    
    # Résultats complets
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'regions_tested': [],
        'overall_performance': {},
        'detailed_analysis': {}
    }
    
    try:
        # 1. ANALYSE DES RÉGIONS DISPONIBLES
        print("📋 1. ANALYSE DES RÉGIONS DISPONIBLES")
        print("-" * 50)
        
        regions_to_test = [
            'hand_area',
            'community_cards', 
            'pot_area',
            'my_stack_area',
            'fold_button',
            'call_button',
            'raise_button'
        ]
        
        for region_name in regions_to_test:
            print(f"\n🎯 Test de la région: {region_name}")
            print("=" * 40)
            
            region_result = {
                'region_name': region_name,
                'capture_success': False,
                'image_info': {},
                'ocr_analysis': {},
                'color_analysis': {},
                'contour_analysis': {},
                'card_detection': {},
                'performance_metrics': {}
            }
            
            try:
                # Capture de la région
                start_time = time.time()
                region_image = screen_capture.capture_region(region_name)
                capture_time = time.time() - start_time
                
                if region_image is None:
                    print(f"❌ Impossible de capturer {region_name}")
                    region_result['capture_success'] = False
                    test_results['regions_tested'].append(region_result)
                    continue
                
                region_result['capture_success'] = True
                region_result['performance_metrics']['capture_time'] = capture_time
                
                # Informations sur l'image
                height, width = region_image.shape[:2]
                channels = region_image.shape[2] if len(region_image.shape) == 3 else 1
                total_pixels = height * width
                
                region_result['image_info'] = {
                    'width': width,
                    'height': height,
                    'channels': channels,
                    'total_pixels': total_pixels,
                    'aspect_ratio': width / height if height > 0 else 0
                }
                
                print(f"✅ Image capturée: {width}x{height} ({channels} canaux)")
                print(f"📊 Pixels totaux: {total_pixels:,}")
                print(f"⏱️  Temps capture: {capture_time:.3f}s")
                
                # 2. ANALYSE OCR DÉTAILLÉE
                print(f"\n🔍 2. ANALYSE OCR DÉTAILLÉE")
                print("-" * 30)
                
                ocr_configs = [
                    ('Standard', '--oem 3 --psm 6'),
                    ('Dense', '--oem 3 --psm 8'),
                    ('Single char', '--oem 3 --psm 10'),
                    ('Cards only', '--oem 3 --psm 6 -c tessedit_char_whitelist=23456789TJQKA♠♥♦♣'),
                    ('Cards dense', '--oem 3 --psm 8 -c tessedit_char_whitelist=23456789TJQKA♠♥♦♣'),
                    ('Numbers only', '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'),
                    ('Letters only', '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
                ]
                
                ocr_results = {}
                for config_name, config in ocr_configs:
                    try:
                        start_time = time.time()
                        
                        # Prétraitement pour OCR
                        if len(region_image.shape) == 3:
                            gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
                        else:
                            gray = region_image.copy()
                        
                        # Amélioration du contraste
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        enhanced = clahe.apply(gray)
                        
                        # OCR
                        text = pytesseract.image_to_string(enhanced, config=config)
                        clean_text = text.strip()
                        
                        ocr_time = time.time() - start_time
                        
                        ocr_results[config_name] = {
                            'text': clean_text,
                            'length': len(clean_text),
                            'time': ocr_time,
                            'has_cards': any(char in '23456789TJQKA' for char in clean_text),
                            'has_suits': any(char in '♠♥♦♣' for char in clean_text),
                            'has_numbers': any(char.isdigit() for char in clean_text),
                            'has_letters': any(char.isalpha() for char in clean_text)
                        }
                        
                        print(f"  {config_name}: '{clean_text}' ({len(clean_text)} chars, {ocr_time:.3f}s)")
                        
                    except Exception as e:
                        print(f"  ❌ {config_name}: Erreur - {e}")
                        ocr_results[config_name] = {'error': str(e)}
                
                region_result['ocr_analysis'] = ocr_results
                
                # 3. ANALYSE DES COULEURS DÉTAILLÉE
                print(f"\n🎨 3. ANALYSE DES COULEURS DÉTAILLÉE")
                print("-" * 35)
                
                try:
                    # Conversion HSV
                    hsv = cv2.cvtColor(region_image, cv2.COLOR_BGR2HSV)
                    
                    # Masques de couleurs
                    color_masks = {
                        'red_light': (np.array([0, 50, 50]), np.array([10, 255, 255])),
                        'red_dark': (np.array([170, 50, 50]), np.array([180, 255, 255])),
                        'black': (np.array([0, 0, 0]), np.array([180, 255, 50])),
                        'white': (np.array([0, 0, 200]), np.array([180, 30, 255])),
                        'green': (np.array([40, 50, 50]), np.array([80, 255, 255])),
                        'blue': (np.array([100, 50, 50]), np.array([130, 255, 255]))
                    }
                    
                    color_analysis = {}
                    for color_name, (lower, upper) in color_masks.items():
                        mask = cv2.inRange(hsv, lower, upper)
                        pixel_count = cv2.countNonZero(mask)
                        pixel_ratio = pixel_count / total_pixels if total_pixels > 0 else 0
                        
                        color_analysis[color_name] = {
                            'pixel_count': pixel_count,
                            'pixel_ratio': pixel_ratio,
                            'percentage': pixel_ratio * 100
                        }
                        
                        print(f"  {color_name}: {pixel_count:,} pixels ({pixel_ratio*100:.1f}%)")
                    
                    region_result['color_analysis'] = color_analysis
                    
                except Exception as e:
                    print(f"  ❌ Erreur analyse couleurs: {e}")
                    region_result['color_analysis'] = {'error': str(e)}
                
                # 4. ANALYSE DES CONTOURS DÉTAILLÉE
                print(f"\n📐 4. ANALYSE DES CONTOURS DÉTAILLÉE")
                print("-" * 35)
                
                try:
                    # Conversion en niveaux de gris
                    if len(region_image.shape) == 3:
                        gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = region_image.copy()
                    
                    # Seuillage adaptatif
                    thresh = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                        cv2.THRESH_BINARY, 11, 2
                    )
                    
                    # Recherche de contours
                    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Analyse des contours
                    contour_analysis = {
                        'total_contours': len(contours),
                        'large_contours': 0,
                        'card_like_contours': 0,
                        'contour_details': []
                    }
                    
                    print(f"  Contours totaux: {len(contours)}")
                    
                    for i, contour in enumerate(contours):
                        area = cv2.contourArea(contour)
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h if h > 0 else 0
                        perimeter = cv2.arcLength(contour, True)
                        
                        # Classification du contour
                        is_large = area > 1000
                        is_card_like = (area > 200 and 0.5 < aspect_ratio < 2.0)
                        
                        if is_large:
                            contour_analysis['large_contours'] += 1
                        if is_card_like:
                            contour_analysis['card_like_contours'] += 1
                        
                        contour_info = {
                            'index': i,
                            'area': area,
                            'width': w,
                            'height': h,
                            'aspect_ratio': aspect_ratio,
                            'perimeter': perimeter,
                            'is_large': is_large,
                            'is_card_like': is_card_like
                        }
                        
                        contour_analysis['contour_details'].append(contour_info)
                        
                        if area > 100:  # Afficher seulement les contours significatifs
                            print(f"    Contour {i+1}: {w}x{h} (aire={area:.0f}, ratio={aspect_ratio:.2f})")
                    
                    region_result['contour_analysis'] = contour_analysis
                    
                except Exception as e:
                    print(f"  ❌ Erreur analyse contours: {e}")
                    region_result['contour_analysis'] = {'error': str(e)}
                
                # 5. DÉTECTION DE CARTES COMPLÈTE
                print(f"\n🃏 5. DÉTECTION DE CARTES COMPLÈTE")
                print("-" * 35)
                
                try:
                    # Test de toutes les méthodes de détection
                    detection_methods = [
                        ('Ultra-rapide', image_analyzer._detect_cards_ultra_fast),
                        ('OCR optimisé', image_analyzer._detect_cards_ocr_optimized),
                        ('Contours', image_analyzer._detect_cards_by_contours),
                        ('Couleur', image_analyzer._detect_cards_by_color),
                        ('Complète', image_analyzer.detect_cards)
                    ]
                    
                    card_detection_results = {}
                    
                    for method_name, detection_func in detection_methods:
                        try:
                            start_time = time.time()
                            cards = detection_func(region_image)
                            detection_time = time.time() - start_time
                            
                            card_detection_results[method_name] = {
                                'cards_found': len(cards),
                                'cards_list': [f"{c.rank}{c.suit}" for c in cards],
                                'confidence_scores': [c.confidence for c in cards],
                                'detection_time': detection_time,
                                'success': len(cards) > 0
                            }
                            
                            print(f"  {method_name}: {len(cards)} cartes en {detection_time:.3f}s")
                            if cards:
                                print(f"    Cartes: {[f'{c.rank}{c.suit}' for c in cards]}")
                            
                        except Exception as e:
                            print(f"  ❌ {method_name}: Erreur - {e}")
                            card_detection_results[method_name] = {'error': str(e)}
                    
                    region_result['card_detection'] = card_detection_results
                    
                except Exception as e:
                    print(f"  ❌ Erreur détection cartes: {e}")
                    region_result['card_detection'] = {'error': str(e)}
                
                # 6. MÉTRIQUES DE PERFORMANCE
                print(f"\n⚡ 6. MÉTRIQUES DE PERFORMANCE")
                print("-" * 30)
                
                total_time = sum(result.get('detection_time', 0) for result in card_detection_results.values() if 'detection_time' in result)
                total_cards = sum(result.get('cards_found', 0) for result in card_detection_results.values() if 'cards_found' in result)
                
                performance_metrics = {
                    'total_detection_time': total_time,
                    'total_cards_detected': total_cards,
                    'average_time_per_card': total_time / total_cards if total_cards > 0 else 0,
                    'success_rate': len([r for r in card_detection_results.values() if r.get('success', False)]) / len(detection_methods)
                }
                
                region_result['performance_metrics'].update(performance_metrics)
                
                print(f"  Temps total détection: {total_time:.3f}s")
                print(f"  Cartes détectées total: {total_cards}")
                print(f"  Temps moyen par carte: {performance_metrics['average_time_per_card']:.3f}s")
                print(f"  Taux de succès: {performance_metrics['success_rate']*100:.1f}%")
                
                # Ajouter aux résultats globaux
                test_results['regions_tested'].append(region_result)
                
            except Exception as e:
                print(f"❌ Erreur générale pour {region_name}: {e}")
                region_result['error'] = str(e)
                test_results['regions_tested'].append(region_result)
        
        # 7. ANALYSE GLOBALE
        print(f"\n📊 7. ANALYSE GLOBALE")
        print("=" * 50)
        
        # Calculer les métriques globales
        total_regions = len(test_results['regions_tested'])
        successful_captures = len([r for r in test_results['regions_tested'] if r.get('capture_success', False)])
        total_cards_detected = sum(
            sum(method.get('cards_found', 0) for method in r.get('card_detection', {}).values() 
                if isinstance(method, dict) and 'cards_found' in method)
            for r in test_results['regions_tested']
        )
        
        test_results['overall_performance'] = {
            'total_regions': total_regions,
            'successful_captures': successful_captures,
            'capture_success_rate': successful_captures / total_regions if total_regions > 0 else 0,
            'total_cards_detected': total_cards_detected,
            'test_duration': time.time() - time.time()  # À calculer
        }
        
        print(f"📋 Régions testées: {total_regions}")
        print(f"✅ Captures réussies: {successful_captures}")
        print(f"📈 Taux de succès capture: {test_results['overall_performance']['capture_success_rate']*100:.1f}%")
        print(f"🃏 Cartes détectées total: {total_cards_detected}")
        
        # 8. SAUVEGARDE DES RÉSULTATS
        print(f"\n💾 8. SAUVEGARDE DES RÉSULTATS")
        print("-" * 30)
        
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Résultats sauvegardés dans: {results_file}")
        
        # 9. RECOMMANDATIONS
        print(f"\n💡 9. RECOMMANDATIONS")
        print("-" * 25)
        
        if total_cards_detected == 0:
            print("⚠️  AUCUNE CARTE DÉTECTÉE")
            print("   - Vérifiez que vous êtes en train de jouer")
            print("   - Assurez-vous que les cartes sont visibles")
            print("   - Vérifiez la calibration des régions")
        else:
            print(f"✅ {total_cards_detected} CARTES DÉTECTÉES")
            print("   - La détection fonctionne correctement")
            print("   - Le bot devrait pouvoir jouer")
        
        # Analyser les meilleures méthodes
        best_methods = []
        for region in test_results['regions_tested']:
            if 'card_detection' in region:
                for method_name, results in region['card_detection'].items():
                    if isinstance(results, dict) and results.get('success', False):
                        best_methods.append((method_name, results.get('cards_found', 0)))
        
        if best_methods:
            best_method = max(best_methods, key=lambda x: x[1])
            print(f"🏆 Meilleure méthode: {best_method[0]} ({best_method[1]} cartes)")
        
        print(f"\n🎉 TEST ULTRA-COMPLET TERMINÉ!")
        print(f"⏰ Fin du test: {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Erreur générale du test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    comprehensive_card_detection_test() 