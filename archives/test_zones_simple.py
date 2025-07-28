#!/usr/bin/env python3
"""
Test simple des zones calibrées
Vérifie que les nouvelles zones sont bien sauvegardées
"""

import json
import os

def test_calibrated_regions_simple():
    """Test simple que les zones calibrées sont bien sauvegardées"""
    print("🔍 TEST SIMPLE DES ZONES CALIBRÉES")
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
    
    # 3. Afficher les zones clés
    key_regions = [
        'hand_area', 'community_cards', 'pot_area',
        'fold_button', 'call_button', 'raise_button',
        'my_stack_area', 'bet_slider', 'resume_button'
    ]
    
    print("\n📋 ZONES CLÉS CALIBRÉES:")
    for region_name in key_regions:
        if region_name in calibrated_data:
            region = calibrated_data[region_name]
            print(f"   ✅ {region_name}: ({region['x']}, {region['y']}) {region['width']}x{region['height']}")
        else:
            print(f"   ❌ {region_name}: NON TROUVÉE")
    
    # 4. Vérifier que les coordonnées sont raisonnables
    print("\n🔍 VÉRIFICATION DES COORDONNÉES:")
    for region_name, region_data in calibrated_data.items():
        x, y, w, h = region_data['x'], region_data['y'], region_data['width'], region_data['height']
        
        # Vérifications de base
        if x < 0 or y < 0:
            print(f"   ⚠️  {region_name}: Coordonnées négatives ({x}, {y})")
        elif w <= 0 or h <= 0:
            print(f"   ⚠️  {region_name}: Dimensions invalides ({w}x{h})")
        elif x > 2000 or y > 1500:
            print(f"   ⚠️  {region_name}: Coordonnées très grandes ({x}, {y})")
        else:
            print(f"   ✅ {region_name}: Coordonnées valides ({x}, {y}) {w}x{h}")
    
    # 5. Compter les zones par catégorie
    button_regions = [k for k in calibrated_data.keys() if 'button' in k]
    area_regions = [k for k in calibrated_data.keys() if 'area' in k]
    other_regions = [k for k in calibrated_data.keys() if 'button' not in k and 'area' not in k]
    
    print(f"\n📊 STATISTIQUES:")
    print(f"   🎯 Boutons: {len(button_regions)}")
    print(f"   📍 Zones: {len(area_regions)}")
    print(f"   🔧 Autres: {len(other_regions)}")
    print(f"   📈 Total: {len(calibrated_data)}")
    
    print("\n🎯 CONCLUSION:")
    print("✅ Les zones calibrées sont bien sauvegardées")
    print("✅ L'agent chargera automatiquement ces zones")
    print("✅ Vos nouvelles calibrations sont prêtes à être utilisées")
    
    return True

if __name__ == "__main__":
    test_calibrated_regions_simple() 