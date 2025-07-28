"""
Script de test pour vérifier l'intégration des nouveaux templates
"""

import sys
import logging
from modules.button_detector import ButtonDetector
from modules.image_analysis import ImageAnalyzer
from modules.screen_capture import ScreenCapture

def test_templates_integration():
    """Teste l'intégration des nouveaux templates"""
    
    # Configuration du logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("=== TEST D'INTÉGRATION DES TEMPLATES ===\n")
    
    # Test 1: Chargement des templates de boutons
    print("1. Test chargement templates de boutons...")
    try:
        button_detector = ButtonDetector()
        print(f"✅ Templates de boutons chargés: {len(button_detector.button_templates)}")
        
        # Vérifier les templates spéciaux
        if hasattr(button_detector, 'special_templates'):
            print(f"✅ Templates spéciaux chargés: {len(button_detector.special_templates)}")
            for template_name in button_detector.special_templates.keys():
                print(f"   - {template_name}")
        else:
            print("❌ Templates spéciaux non trouvés")
            
    except Exception as e:
        print(f"❌ Erreur chargement templates boutons: {e}")
    
    # Test 2: Chargement des templates de cartes
    print("\n2. Test chargement templates de cartes...")
    try:
        image_analyzer = ImageAnalyzer()
        print(f"✅ Templates de cartes chargés: {len(image_analyzer.card_templates)}")
        
        # Lister les rangs disponibles
        rank_templates = [k for k in image_analyzer.card_templates.keys() if k.startswith('rank_')]
        suit_templates = [k for k in image_analyzer.card_templates.keys() if k.startswith('suit_')]
        
        print(f"   - Rangs disponibles: {len(rank_templates)}")
        print(f"   - Couleurs disponibles: {len(suit_templates)}")
        
        # Vérifier que tous les rangs sont présents
        expected_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        missing_ranks = []
        for rank in expected_ranks:
            if f'rank_{rank}' not in rank_templates:
                missing_ranks.append(rank)
        
        if missing_ranks:
            print(f"   ⚠️ Rangs manquants: {missing_ranks}")
        else:
            print("   ✅ Tous les rangs sont présents!")
            
    except Exception as e:
        print(f"❌ Erreur chargement templates cartes: {e}")
    
    # Test 3: Test de capture d'écran
    print("\n3. Test capture d'écran...")
    try:
        screen_capture = ScreenCapture()
        print("✅ Module de capture d'écran initialisé")
        
        # Test capture écran complet
        full_screen = screen_capture.capture_full_screen()
        if full_screen is not None:
            print(f"✅ Capture écran complet: {full_screen.shape}")
        else:
            print("❌ Échec capture écran complet")
            
    except Exception as e:
        print(f"❌ Erreur capture d'écran: {e}")
    
    # Test 4: Test de détection de couronne (simulation)
    print("\n4. Test détection couronne de victoire...")
    try:
        if hasattr(button_detector, 'detect_winner_crown'):
            print("✅ Méthode de détection couronne disponible")
            
            # Test avec une image vide (simulation)
            import numpy as np
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            result = button_detector.detect_winner_crown(test_image)
            print(f"✅ Test détection couronne: {result}")
        else:
            print("❌ Méthode de détection couronne non trouvée")
            
    except Exception as e:
        print(f"❌ Erreur test détection couronne: {e}")
    
    # Test 5: Vérification des fichiers templates
    print("\n5. Vérification des fichiers templates...")
    import os
    
    # Vérifier les boutons
    button_files = [
        'fold_button.png', 'check_button.png', 'raise_button.png',
        'all_in_button.png', 'bet_button.png', 'cann_button.png',
        'new_hand_button.png', 'resume_button.png', 'winner.png', 'winner2.png'
    ]
    
    missing_buttons = []
    for button_file in button_files:
        if os.path.exists(f"templates/buttons/{button_file}"):
            print(f"✅ {button_file}")
        else:
            print(f"❌ {button_file}")
            missing_buttons.append(button_file)
    
    # Vérifier les rangs de cartes
    rank_files = [f"card_{rank}.png" for rank in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']]
    
    missing_ranks = []
    for rank_file in rank_files:
        if os.path.exists(f"templates/cards/ranks/{rank_file}"):
            print(f"✅ {rank_file}")
        else:
            print(f"❌ {rank_file}")
            missing_ranks.append(rank_file)
    
    # Résumé
    print(f"\n📊 RÉSUMÉ:")
    print(f"   - Boutons manquants: {len(missing_buttons)}")
    print(f"   - Rangs manquants: {len(missing_ranks)}")
    
    if not missing_buttons and not missing_ranks:
        print("🎉 TOUS LES TEMPLATES SONT PRÉSENTS!")
    else:
        print("⚠️ Certains templates manquent")
    
    print("\n=== TEST TERMINÉ ===")

if __name__ == "__main__":
    test_templates_integration() 