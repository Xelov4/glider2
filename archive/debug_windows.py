#!/usr/bin/env python3
"""
Script pour lister toutes les fenêtres ouvertes et identifier la fenêtre de poker
"""

import pygetwindow as gw
import pyautogui
import time

def list_all_windows():
    """Liste toutes les fenêtres ouvertes"""
    print("=== FENÊTRES OUVERTES ===")
    
    try:
        # Obtenir toutes les fenêtres
        windows = gw.getAllTitles()
        
        print(f"Nombre total de fenêtres: {len(windows)}")
        print("\nFenêtres trouvées:")
        
        for i, title in enumerate(windows, 1):
            if title.strip():  # Ignorer les titres vides
                print(f"{i:2d}. {title}")
                
        return windows
        
    except Exception as e:
        print(f"Erreur: {e}")
        return []

def find_poker_windows(windows):
    """Trouve les fenêtres qui pourraient être du poker"""
    print("\n=== FENÊTRES POKER POTENTIELLES ===")
    
    poker_keywords = [
        'poker', 'betclick', 'stars', '888', 'partypoker', 'unibet', 'winamax',
        'pokerstars', 'spin', 'rush', 'zoom', 'fast', 'speed'
    ]
    
    poker_windows = []
    
    for title in windows:
        title_lower = title.lower()
        for keyword in poker_keywords:
            if keyword in title_lower:
                poker_windows.append(title)
                print(f"🎰 POTENTIEL: {title}")
                break
    
    if not poker_windows:
        print("❌ Aucune fenêtre de poker détectée")
        print("💡 Suggestions:")
        print("   - Ouvrez votre client de poker")
        print("   - Vérifiez le nom exact de la fenêtre")
        print("   - Modifiez config.ini avec le bon nom")
    
    return poker_windows

def test_screen_capture():
    """Test de capture d'écran"""
    print("\n=== TEST CAPTURE D'ÉCRAN ===")
    
    try:
        # Obtenir la taille de l'écran
        screen_width, screen_height = pyautogui.size()
        print(f"Résolution écran: {screen_width}x{screen_height}")
        
        # Test capture d'une petite zone
        screenshot = pyautogui.screenshot(region=(0, 0, 100, 100))
        print("✅ Capture d'écran fonctionne")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur capture d'écran: {e}")
        return False

def test_window_focus():
    """Test de focus sur une fenêtre"""
    print("\n=== TEST FOCUS FENÊTRE ===")
    
    try:
        # Essayer de trouver une fenêtre de poker
        windows = gw.getAllTitles()
        poker_windows = find_poker_windows(windows)
        
        if poker_windows:
            # Essayer de focus sur la première fenêtre de poker
            window_title = poker_windows[0]
            window = gw.getWindowsWithTitle(window_title)[0]
            
            print(f"Tentative de focus sur: {window_title}")
            window.activate()
            time.sleep(1)
            
            print("✅ Focus réussi")
            return True
        else:
            print("❌ Aucune fenêtre de poker à tester")
            return False
            
    except Exception as e:
        print(f"❌ Erreur focus: {e}")
        return False

if __name__ == "__main__":
    print("🔍 DIAGNOSTIC FENÊTRES POKER")
    print("=" * 50)
    
    # Lister toutes les fenêtres
    windows = list_all_windows()
    
    # Trouver les fenêtres de poker
    poker_windows = find_poker_windows(windows)
    
    # Tests
    test_screen_capture()
    test_window_focus()
    
    print("\n" + "=" * 50)
    print("📋 INSTRUCTIONS:")
    print("1. Ouvrez votre client de poker")
    print("2. Notez le nom exact de la fenêtre")
    print("3. Modifiez config.ini avec le bon nom")
    print("4. Relancez l'agent avec: py main.py")
    
    if poker_windows:
        print(f"\n🎯 FENÊTRE RECOMMANDÉE: {poker_windows[0]}")
        print("Copiez ce nom dans config.ini sous [Display] target_window_title=") 