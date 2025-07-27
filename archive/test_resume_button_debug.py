#!/usr/bin/env python3
"""
Test de debug pour le bouton "Reprendre"
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.screen_capture import ScreenCapture
from modules.button_detector import ButtonDetector
from modules.image_analysis import ImageAnalyzer
import cv2
import numpy as np

def test_resume_button_detection():
    """Test spécifique pour la détection du bouton 'Reprendre'"""
    
    print("=== TEST DÉTECTION BOUTON 'REPRENDRE' ===")
    
    # Initialisation
    screen_capture = ScreenCapture()
    button_detector = ButtonDetector()
    image_analyzer = ImageAnalyzer()
    
    print(f"✅ Modules initialisés")
    
    # Capture de la région resume_button
    try:
        resume_region = screen_capture.capture_region('resume_button')
        if resume_region is None:
            print("❌ Impossible de capturer la région 'resume_button'")
            return
            
        print(f"✅ Région 'resume_button' capturée: {resume_region.shape}")
        
        # Sauvegarde pour inspection
        cv2.imwrite('debug_resume_button.png', resume_region)
        print("💾 Image sauvegardée: debug_resume_button.png")
        
        # Test OCR sur la région
        text = image_analyzer.extract_text(resume_region)
        print(f"📝 Texte détecté: '{text}'")
        
        # Test de détection de boutons
        buttons = button_detector.detect_available_actions(resume_region)
        print(f"🔘 Boutons détectés: {buttons}")
        
        # Test avec OCR pour "Reprendre", "Resume", etc.
        resume_keywords = ["Reprendre", "Resume", "Continue", "Rejoindre", "Repartir"]
        for keyword in resume_keywords:
            if keyword.lower() in text.lower():
                print(f"✅ Mot-clé trouvé: '{keyword}'")
                break
        else:
            print("❌ Aucun mot-clé 'Reprendre' trouvé")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

def test_resume_button_coordinates():
    """Test des coordonnées du bouton 'Reprendre'"""
    
    print("\n=== TEST COORDONNÉES BOUTON 'REPRENDRE' ===")
    
    screen_capture = ScreenCapture()
    
    # Vérifier les coordonnées
    resume_coords = screen_capture.regions.get('resume_button')
    if resume_coords:
        print(f"📍 Coordonnées 'resume_button': {resume_coords}")
        
        # Test si les coordonnées sont valides
        import pyautogui
        screen_width, screen_height = pyautogui.size()
        
        print(f"📐 Écran: {screen_width}x{screen_height}")
        
        if (resume_coords.x + resume_coords.width > screen_width or 
            resume_coords.y + resume_coords.height > screen_height):
            print("⚠️ Coordonnées hors écran!")
        else:
            print("✅ Coordonnées valides")
    else:
        print("❌ Région 'resume_button' non trouvée")

def test_full_screen_capture():
    """Test de capture d'écran complète pour debug"""
    
    print("\n=== TEST CAPTURE ÉCRAN COMPLÈTE ===")
    
    screen_capture = ScreenCapture()
    
    try:
        # Capture de toutes les régions
        all_regions = screen_capture.capture_all_regions()
        
        print(f"✅ {len(all_regions)} régions capturées")
        
        # Sauvegarder les images importantes
        for region_name, image in all_regions.items():
            if image is not None:
                filename = f"debug_{region_name}.png"
                cv2.imwrite(filename, image)
                print(f"💾 {region_name}: {filename}")
                
    except Exception as e:
        print(f"❌ Erreur capture: {e}")

if __name__ == "__main__":
    print("🔍 Démarrage du debug du bouton 'Reprendre'...")
    
    test_resume_button_coordinates()
    test_resume_button_detection()
    test_full_screen_capture()
    
    print("\n🎯 Debug terminé - Vérifiez les images générées") 