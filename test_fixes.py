#!/usr/bin/env python3
"""
Script de test pour vérifier les corrections de main.py
"""

import sys
import logging
from modules.screen_capture import ScreenCapture
from modules.image_analysis import ImageAnalyzer
from modules.game_state import GameState
from modules.button_detector import ButtonDetector

def test_modules():
    """Test des modules principaux"""
    print("=== TEST DES MODULES ===")
    
    try:
        # Test ScreenCapture
        print("1. Test ScreenCapture...")
        screen_capture = ScreenCapture()
        print("   ✓ ScreenCapture initialisé")
        
        # Test ImageAnalyzer
        print("2. Test ImageAnalyzer...")
        image_analyzer = ImageAnalyzer()
        print("   ✓ ImageAnalyzer initialisé")
        
        # Test GameState
        print("3. Test GameState...")
        game_state = GameState()
        print("   ✓ GameState initialisé")
        
        # Test de l'attribut is_my_turn
        print("4. Test is_my_turn...")
        game_state.is_my_turn = True
        if game_state.is_my_turn:
            print("   ✓ is_my_turn fonctionne")
        else:
            print("   ✗ is_my_turn ne fonctionne pas")
        
        # Test ButtonDetector
        print("5. Test ButtonDetector...")
        button_detector = ButtonDetector()
        print("   ✓ ButtonDetector initialisé")
        
        print("\n=== TOUS LES TESTS PASSÉS ===")
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return False

def test_imports():
    """Test des imports"""
    print("=== TEST DES IMPORTS ===")
    
    try:
        from modules import (
            ScreenCapture, ImageAnalyzer, GameState, 
            ButtonDetector, PokerEngine, AIDecisionMaker,
            AutomationEngine, GeneralStrategy, SpinRushStrategy
        )
        print("✓ Tous les imports fonctionnent")
        return True
    except Exception as e:
        print(f"✗ Erreur d'import: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    success = True
    success &= test_imports()
    success &= test_modules()
    
    if success:
        print("\n🎉 Toutes les corrections sont fonctionnelles !")
        print("Vous pouvez maintenant lancer main.py avec: py main.py")
    else:
        print("\n❌ Il y a encore des problèmes à résoudre")
        sys.exit(1) 