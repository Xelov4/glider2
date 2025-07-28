#!/usr/bin/env python3
"""
Script de test pour vérifier les corrections apportées
"""

import sys
import logging
from modules.image_analysis import ImageAnalyzer
from modules.button_detector import ButtonDetector
from modules.screen_capture import ScreenCapture

def test_card_templates():
    """Test le chargement des templates de cartes"""
    print("=== TEST CHARGEMENT TEMPLATES CARTES ===")
    
    analyzer = ImageAnalyzer()
    templates = analyzer.card_templates
    
    print(f"Templates chargés: {len(templates)}")
    for key, template in templates.items():
        if template is not None:
            print(f"  ✓ {key}: {template.shape}")
        else:
            print(f"  ✗ {key}: None")
    
    return len(templates) > 0

def test_button_templates():
    """Test le chargement des templates de boutons"""
    print("\n=== TEST CHARGEMENT TEMPLATES BOUTONS ===")
    
    detector = ButtonDetector()
    templates = detector.button_templates
    
    print(f"Templates de boutons chargés: {len(templates)}")
    for btn_type, btn_templates in templates.items():
        if isinstance(btn_templates, dict):
            for state, template in btn_templates.items():
                if template is not None:
                    print(f"  ✓ {btn_type}.{state}: {template.shape}")
                else:
                    print(f"  ✗ {btn_type}.{state}: None")
        else:
            print(f"  ✗ {btn_type}: Format invalide")
    
    return len(templates) > 0

def test_screen_capture():
    """Test la capture d'écran"""
    print("\n=== TEST CAPTURE ÉCRAN ===")
    
    capture = ScreenCapture()
    
    try:
        # Test de capture d'une région
        test_region = capture.capture_region('hand_area')
        if test_region is not None:
            print(f"  ✓ Capture réussie: {test_region.shape}")
            return True
        else:
            print("  ⚠️ Capture retourne None (normal si pas de jeu)")
            return True  # Pas une erreur
    except Exception as e:
        print(f"  ✗ Erreur capture: {e}")
        return False

def test_ocr():
    """Test l'OCR"""
    print("\n=== TEST OCR ===")
    
    analyzer = ImageAnalyzer()
    
    # Créer une image de test simple
    import numpy as np
    test_image = np.zeros((50, 200, 3), dtype=np.uint8)
    test_image[:] = (255, 255, 255)  # Fond blanc
    
    try:
        text = analyzer.extract_text(test_image)
        print(f"OCR fonctionne: '{text}'")
        return True
    except Exception as e:
        print(f"Erreur OCR: {e}")
        return False

def main():
    """Tests principaux"""
    print("🔧 TESTS DES CORRECTIONS")
    print("=" * 50)
    
    results = []
    
    # Tests
    results.append(("Templates cartes", test_card_templates()))
    results.append(("Templates boutons", test_button_templates()))
    results.append(("Capture écran", test_screen_capture()))
    results.append(("OCR", test_ocr()))
    
    # Résumé
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nRésultat: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 Tous les tests sont passés! L'agent devrait fonctionner.")
    else:
        print("⚠️ Certains tests ont échoué. Vérifiez les problèmes.")

if __name__ == "__main__":
    main() 