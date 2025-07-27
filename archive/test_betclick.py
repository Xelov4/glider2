#!/usr/bin/env python3
"""
Script de test spécifique pour Betclick Poker
"""

import pygetwindow as gw
import pyautogui
import time
import logging
from modules.screen_capture import ScreenCapture
from modules.image_analysis import ImageAnalyzer

def test_betclick_window():
    """Test de détection de la fenêtre Betclick Poker"""
    print("=== TEST FENÊTRE BETCLICK POKER ===")
    
    try:
        # Chercher la fenêtre Betclic Poker
        windows = gw.getWindowsWithTitle("Betclic Poker")
        
        if windows:
            window = windows[0]
            print(f"✅ Fenêtre trouvée: {window.title}")
            print(f"   Position: ({window.left}, {window.top})")
            print(f"   Taille: {window.width}x{window.height}")
            
            # Activer la fenêtre
            window.activate()
            time.sleep(1)
            print("✅ Fenêtre activée")
            
            return window
        else:
            print("❌ Fenêtre 'Betclic Poker' non trouvée")
            print("💡 Vérifiez que:")
            print("   - Betclic Poker est ouvert")
            print("   - Le nom de la fenêtre est exactement 'Betclic Poker'")
            
            # Lister toutes les fenêtres pour debug
            all_windows = gw.getAllTitles()
            print("\nFenêtres disponibles:")
            for i, title in enumerate(all_windows[:10], 1):  # Afficher les 10 premières
                if title.strip():
                    print(f"   {i}. {title}")
            
            return None
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

def test_screen_capture_betclick():
    """Test de capture d'écran pour Betclick Poker"""
    print("\n=== TEST CAPTURE D'ÉCRAN ===")
    
    try:
        # Initialiser le module de capture
        screen_capture = ScreenCapture()
        
        # Tester la capture d'une région
        test_region = screen_capture.capture_region('hand_area')
        
        if test_region is not None:
            print(f"✅ Capture réussie - Taille: {test_region.shape}")
            return True
        else:
            print("❌ Capture échouée - Région vide")
            return False
            
    except Exception as e:
        print(f"❌ Erreur capture: {e}")
        return False

def test_image_analysis():
    """Test d'analyse d'images"""
    print("\n=== TEST ANALYSE D'IMAGES ===")
    
    try:
        # Initialiser l'analyseur d'images
        analyzer = ImageAnalyzer()
        print("✅ ImageAnalyzer initialisé")
        
        # Tester la capture d'une petite zone
        screenshot = pyautogui.screenshot(region=(0, 0, 200, 100))
        print("✅ Screenshot de test créé")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur analyse: {e}")
        return False

def test_full_pipeline():
    """Test du pipeline complet"""
    print("\n=== TEST PIPELINE COMPLET ===")
    
    try:
        # 1. Trouver la fenêtre
        window = test_betclick_window()
        if not window:
            return False
        
        # 2. Activer la fenêtre
        window.activate()
        time.sleep(2)
        
        # 3. Capturer les régions
        screen_capture = ScreenCapture()
        regions = screen_capture.capture_all_regions()
        
        print(f"✅ Régions capturées: {len(regions)}")
        
        # 4. Analyser les images
        analyzer = ImageAnalyzer()
        
        for region_name, image in regions.items():
            if image is not None and image.size > 0:
                print(f"   ✅ {region_name}: {image.shape}")
            else:
                print(f"   ❌ {region_name}: vide")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur pipeline: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🎰 TEST BETCLIC POKER")
    print("=" * 50)
    
    # Tests
    success = True
    success &= test_betclick_window() is not None
    success &= test_screen_capture_betclick()
    success &= test_image_analysis()
    success &= test_full_pipeline()
    
    print("\n" + "=" * 50)
    
    if success:
        print("🎉 TOUS LES TESTS PASSÉS!")
        print("L'agent devrait maintenant fonctionner avec Betclick Poker")
        print("Lancez avec: py main.py")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("Vérifiez que Betclick Poker est ouvert et visible")
    
    print("\n💡 CONSEILS:")
    print("- Assurez-vous que Betclic Poker est ouvert")
    print("- La fenêtre doit être visible (pas minimisée)")
    print("- Les coordonnées dans calibrated_regions.json doivent correspondre")
    print("- Si problème, recalibrez avec: py tools/calibration_tool.py") 