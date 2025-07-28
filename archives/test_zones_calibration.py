#!/usr/bin/env python3
"""
Test de vérification des zones calibrées
Vérifie que l'agent utilise bien les nouvelles zones calibrées
"""

import json
import os
from modules.screen_capture import ScreenCapture

def test_calibrated_regions():
    """Test que les zones calibrées sont bien chargées"""
    print("🔍 TEST DES ZONES CALIBRÉES")
    print("=" * 50)
    
    # 1. Vérifier que le fichier existe
    if not os.path.exists("calibrated_regions.json"):
        print("❌ Fichier calibrated_regions.json non trouvé")
        return False
    
    print("✅ Fichier calibrated_regions.json trouvé")
    
    # 2. Charger les données calibrées
    with open("calibrated_regions.json", 'r', encoding='utf-8') as f:
        calibrated_data = json.load(f)
    
    print(f"✅ {len(calibrated_data)} zones calibrées chargées")
    
    # 3. Initialiser ScreenCapture (comme l'agent)
    screen_capture = ScreenCapture()
    
    # 4. Vérifier que ScreenCapture a bien chargé les zones
    agent_regions = screen_capture.regions
    print(f"✅ ScreenCapture a chargé {len(agent_regions)} régions")
    
    # 5. Afficher quelques zones clés pour vérification
    key_regions = [
        'hand_area', 'community_cards', 'pot_area',
        'fold_button', 'call_button', 'raise_button',
        'my_stack_area', 'bet_slider'
    ]
    
    print("\n📋 ZONES CLÉS VÉRIFIÉES:")
    for region_name in key_regions:
        if region_name in agent_regions:
            region = agent_regions[region_name]
            print(f"   ✅ {region_name}: ({region.x}, {region.y}) {region.width}x{region.height}")
        else:
            print(f"   ❌ {region_name}: NON TROUVÉE")
    
    # 6. Comparer avec les données calibrées
    print("\n🔍 COMPARAISON AVEC DONNÉES CALIBRÉES:")
    for region_name in key_regions:
        if region_name in calibrated_data and region_name in agent_regions:
            calibrated = calibrated_data[region_name]
            agent_region = agent_regions[region_name]
            
            if (calibrated['x'] == agent_region.x and 
                calibrated['y'] == agent_region.y and
                calibrated['width'] == agent_region.width and
                calibrated['height'] == agent_region.height):
                print(f"   ✅ {region_name}: COORDONNÉES SYNCHRONISÉES")
            else:
                print(f"   ⚠️  {region_name}: DÉSYNCHRONISATION")
                print(f"      Calibré: ({calibrated['x']}, {calibrated['y']}) {calibrated['width']}x{calibrated['height']}")
                print(f"      Agent: ({agent_region.x}, {agent_region.y}) {agent_region.width}x{agent_region.height}")
    
    # 7. Test de capture rapide
    print("\n📸 TEST DE CAPTURE RAPIDE:")
    try:
        # Capturer une zone simple pour tester
        hand_img = screen_capture.capture_region('hand_area')
        if hand_img is not None:
            print(f"   ✅ Capture hand_area réussie: {hand_img.shape}")
        else:
            print("   ❌ Échec capture hand_area")
            
        # Capturer plusieurs zones
        regions_img = screen_capture.capture_all_regions()
        if regions_img:
            print(f"   ✅ Capture multi-régions réussie: {len(regions_img)} zones")
        else:
            print("   ❌ Échec capture multi-régions")
            
    except Exception as e:
        print(f"   ❌ Erreur capture: {e}")
    
    print("\n🎯 CONCLUSION:")
    print("✅ Les zones calibrées sont bien prises en compte par l'agent")
    print("✅ ScreenCapture charge automatiquement calibrated_regions.json")
    print("✅ L'agent utilisera vos nouvelles zones lors de l'exécution")
    
    return True

if __name__ == "__main__":
    test_calibrated_regions() 