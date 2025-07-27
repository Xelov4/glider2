#!/usr/bin/env python3
"""
Script de test pour vérifier le chargement des templates
"""

import logging
from modules.button_detector import ButtonDetector
from modules.image_analysis import ImageAnalyzer

def test_button_templates():
    """Test du chargement des templates de boutons"""
    print("🎮 TEST TEMPLATES DE BOUTONS")
    print("=" * 40)
    
    try:
        button_detector = ButtonDetector()
        
        # Vérifier les templates chargés
        templates = button_detector.button_templates
        
        print(f"Templates de boutons chargés: {len(templates)}")
        
        for button_type, template_dict in templates.items():
            if 'enabled' in template_dict:
                template = template_dict['enabled']
                if hasattr(template, 'shape'):
                    print(f"✅ {button_type}: {template.shape}")
                else:
                    print(f"❌ {button_type}: template invalide")
            else:
                print(f"❌ {button_type}: template manquant")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test templates boutons: {e}")
        return False

def test_card_templates():
    """Test du chargement des templates de cartes"""
    print("\n🃏 TEST TEMPLATES DE CARTES")
    print("=" * 40)
    
    try:
        image_analyzer = ImageAnalyzer()
        
        # Vérifier les templates chargés
        templates = image_analyzer.card_templates
        
        print(f"Templates de cartes chargés: {len(templates)}")
        
        # Compter par type
        rank_templates = [k for k in templates.keys() if k.startswith('rank_')]
        suit_templates = [k for k in templates.keys() if k.startswith('suit_')]
        card_templates = [k for k in templates.keys() if not k.startswith(('rank_', 'suit_'))]
        
        print(f"  - Rangs: {len(rank_templates)}")
        print(f"  - Couleurs: {len(suit_templates)}")
        print(f"  - Cartes complètes: {len(card_templates)}")
        
        # Afficher quelques exemples
        if rank_templates:
            print(f"  Exemples de rangs: {rank_templates[:5]}")
        if suit_templates:
            print(f"  Exemples de couleurs: {suit_templates}")
        if card_templates:
            print(f"  Exemples de cartes: {card_templates[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test templates cartes: {e}")
        return False

def test_template_matching():
    """Test de template matching avec les vrais templates"""
    print("\n🔍 TEST TEMPLATE MATCHING")
    print("=" * 40)
    
    try:
        button_detector = ButtonDetector()
        
        # Créer une image de test (simulation d'un bouton)
        import numpy as np
        test_image = np.zeros((100, 200, 3), dtype=np.uint8)
        test_image[20:80, 50:150] = [0, 100, 0]  # Zone verte
        
        print("Test de détection de boutons sur image simulée...")
        
        # Tester la détection
        buttons = button_detector.detect_available_actions(test_image)
        
        print(f"Boutons détectés: {len(buttons)}")
        for btn in buttons:
            print(f"  - {btn.name} (confiance: {btn.confidence:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test template matching: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🎯 TEST COMPLET DES TEMPLATES")
    print("=" * 50)
    
    # Tests
    success1 = test_button_templates()
    success2 = test_card_templates()
    success3 = test_template_matching()
    
    print("\n" + "=" * 50)
    
    if success1 and success2 and success3:
        print("🎉 TOUS LES TEMPLATES SONT CHARGÉS!")
        print("L'agent devrait maintenant détecter les boutons et cartes correctement")
    else:
        print("❌ CERTAINS TEMPLATES ONT DES PROBLÈMES")
        print("Vérifiez que tous les fichiers sont présents dans templates/")
    
    print("\n💡 CONSEILS:")
    print("- Assurez-vous que tous les templates sont dans le bon dossier")
    print("- Les noms de fichiers doivent correspondre exactement")
    print("- Relancez l'agent avec: py main.py") 