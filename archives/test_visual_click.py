#!/usr/bin/env python3
"""
Test visuel des clics de l'agent
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.screen_capture import ScreenCapture
from modules.automation import Automation

def test_visual_click():
    """Test visuel d'un clic sur new_hand_button"""
    
    print("=== TEST VISUEL CLIC ===")
    
    screen_capture = ScreenCapture()
    automation = Automation()
    
    # 1. Capturer l'écran avant le clic
    print("📸 Capture de l'écran avant clic...")
    import pyautogui
    screenshot_before = pyautogui.screenshot()
    screenshot_before.save('before_click.png')
    print("💾 Image sauvegardée: before_click.png")
    
    # 2. Obtenir les coordonnées du bouton
    if 'new_hand_button' in screen_capture.regions:
        region = screen_capture.regions['new_hand_button']
        center_x = region.x + region.width // 2
        center_y = region.y + region.height // 2
        
        print(f"🎯 Position de clic: ({center_x}, {center_y})")
        print(f"📍 Région: ({region.x}, {region.y}, {region.width}, {region.height})")
        
        # 3. Attendre confirmation
        print("\n⚠️ ATTENTION: L'agent va cliquer dans 3 secondes!")
        print("Assurez-vous que la fenêtre Betclic est visible et bien positionnée.")
        for i in range(3, 0, -1):
            print(f"⏰ {i}...")
            time.sleep(1)
        
        # 4. Effectuer le clic
        print("🖱️ Clic en cours...")
        automation.click_at_position(center_x, center_y)
        
        # 5. Attendre un peu
        time.sleep(2)
        
        # 6. Capturer l'écran après le clic
        print("📸 Capture de l'écran après clic...")
        screenshot_after = pyautogui.screenshot()
        screenshot_after.save('after_click.png')
        print("💾 Image sauvegardée: after_click.png")
        
        print("\n✅ Test terminé!")
        print("📋 Vérifiez les images:")
        print("   - before_click.png (avant le clic)")
        print("   - after_click.png (après le clic)")
        print("   - test_new_hand_button.png (région capturée)")
        
    else:
        print("❌ Région new_hand_button non trouvée")

def test_window_position():
    """Test de la position de la fenêtre Betclic"""
    
    print("\n=== TEST POSITION FENÊTRE ===")
    
    import pygetwindow as gw
    
    # Chercher la fenêtre Betclic
    windows = gw.getAllTitles()
    betclic_windows = [w for w in windows if 'betclic' in w.lower()]
    
    if betclic_windows:
        print(f"✅ Fenêtres Betclic trouvées: {betclic_windows}")
        
        for window_title in betclic_windows:
            try:
                window = gw.getWindowsWithTitle(window_title)[0]
                print(f"📍 Fenêtre '{window_title}':")
                print(f"   Position: ({window.left}, {window.top})")
                print(f"   Taille: {window.width}x{window.height}")
                print(f"   État: {'Actif' if window.isActive else 'Inactif'}")
            except Exception as e:
                print(f"❌ Erreur fenêtre {window_title}: {e}")
    else:
        print("❌ Aucune fenêtre Betclic trouvée")
        print("Fenêtres disponibles:")
        for w in windows[:10]:  # Afficher les 10 premières
            print(f"   - {w}")

if __name__ == "__main__":
    print("🔍 Test visuel des clics...")
    
    test_window_position()
    test_visual_click()
    
    print("\n🎯 Instructions:")
    print("1. Ouvrez Betclic Poker")
    print("2. Positionnez la fenêtre à gauche de l'écran")
    print("3. Relancez ce test")
    print("4. Vérifiez les images générées") 