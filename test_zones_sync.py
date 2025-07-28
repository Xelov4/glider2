#!/usr/bin/env python3
"""
Test de synchronisation des zones entre calibrated_regions.json et l'agent
"""

import json
import os
from modules.screen_capture import ScreenCapture

def test_zones_sync():
    """Test que les zones du JSON sont bien chargées par l'agent"""
    print("🔍 TEST DE SYNCHRONISATION DES ZONES")
    print("=" * 50)
    
    # 1. Charger les zones du JSON
    if not os.path.exists("calibrated_regions.json"):
        print("❌ Fichier calibrated_regions.json non trouvé")
        return False
    
    with open("calibrated_regions.json", 'r', encoding='utf-8') as f:
        json_regions = json.load(f)
    
    print(f"✅ JSON contient {len(json_regions)} zones")
    
    # 2. Initialiser ScreenCapture (comme l'agent)
    screen_capture = ScreenCapture()
    agent_regions = screen_capture.regions
    
    print(f"✅ Agent a chargé {len(agent_regions)} zones")
    
    # 3. Vérifier la synchronisation
    print("\n🔍 VÉRIFICATION DE SYNCHRONISATION:")
    
    json_names = set(json_regions.keys())
    agent_names = set(agent_regions.keys())
    
    # Zones manquantes dans l'agent
    missing_in_agent = json_names - agent_names
    if missing_in_agent:
        print(f"❌ Zones manquantes dans l'agent: {missing_in_agent}")
    else:
        print("✅ Toutes les zones du JSON sont dans l'agent")
    
    # Zones en trop dans l'agent
    extra_in_agent = agent_names - json_names
    if extra_in_agent:
        print(f"⚠️  Zones en trop dans l'agent: {extra_in_agent}")
    else:
        print("✅ Aucune zone en trop dans l'agent")
    
    # 4. Vérifier les coordonnées des zones clés
    print("\n📋 VÉRIFICATION DES COORDONNÉES CLÉS:")
    
    key_regions = [
        'hand_area', 'community_cards', 'pot_area',
        'fold_button', 'call_button', 'raise_button',
        'my_stack_area', 'bet_slider', 'resume_button'
    ]
    
    for region_name in key_regions:
        if region_name in json_regions and region_name in agent_regions:
            json_coords = json_regions[region_name]
            agent_region = agent_regions[region_name]
            
            # Comparer les coordonnées
            if (json_coords['x'] == agent_region.x and 
                json_coords['y'] == agent_region.y and
                json_coords['width'] == agent_region.width and
                json_coords['height'] == agent_region.height):
                print(f"   ✅ {region_name}: Synchronisé")
            else:
                print(f"   ❌ {region_name}: DÉSYNCHRONISÉ")
                print(f"      JSON: ({json_coords['x']}, {json_coords['y']}) {json_coords['width']}x{json_coords['height']}")
                print(f"      Agent: ({agent_region.x}, {agent_region.y}) {agent_region.width}x{agent_region.height}")
        else:
            print(f"   ⚠️  {region_name}: Manquant dans {'JSON' if region_name not in json_regions else 'Agent'}")
    
    # 5. Résumé
    print(f"\n📊 RÉSUMÉ:")
    print(f"   Zones JSON: {len(json_regions)}")
    print(f"   Zones Agent: {len(agent_regions)}")
    print(f"   Manquantes: {len(missing_in_agent)}")
    print(f"   En trop: {len(extra_in_agent)}")
    
    if not missing_in_agent and not extra_in_agent:
        print("🎉 SYNCHRONISATION PARFAITE !")
        return True
    else:
        print("⚠️  PROBLÈMES DE SYNCHRONISATION DÉTECTÉS")
        return False

if __name__ == "__main__":
    test_zones_sync() 