#!/usr/bin/env python3
"""
Test de synchronisation des coordonnées entre JSON et Agent
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.screen_capture import ScreenCapture

def test_coordinates_sync():
    """Test si les coordonnées du JSON sont bien chargées par l'agent"""
    
    print("=== TEST SYNCHRONISATION COORDONNÉES ===")
    
    # 1. Lire les coordonnées du JSON
    try:
        with open('calibrated_regions.json', 'r', encoding='utf-8') as f:
            json_coords = json.load(f)
        print(f"✅ JSON chargé: {len(json_coords)} régions")
    except Exception as e:
        print(f"❌ Erreur lecture JSON: {e}")
        return False
    
    # 2. Initialiser l'agent et vérifier ses coordonnées
    try:
        screen_capture = ScreenCapture()
        agent_coords = screen_capture.regions
        print(f"✅ Agent initialisé: {len(agent_coords)} régions")
    except Exception as e:
        print(f"❌ Erreur initialisation agent: {e}")
        return False
    
    # 3. Comparer les coordonnées
    print("\n=== COMPARAISON COORDONNÉES ===")
    
    mismatches = []
    for region_name in json_coords.keys():
        if region_name in agent_coords:
            json_region = json_coords[region_name]
            agent_region = agent_coords[region_name]
            
            # Comparer les coordonnées
            json_x = json_region['x']
            json_y = json_region['y']
            json_w = json_region['width']
            json_h = json_region['height']
            
            agent_x = agent_region.x
            agent_y = agent_region.y
            agent_w = agent_region.width
            agent_h = agent_region.height
            
            if (json_x != agent_x or json_y != agent_y or 
                json_w != agent_w or json_h != agent_h):
                mismatches.append({
                    'region': region_name,
                    'json': (json_x, json_y, json_w, json_h),
                    'agent': (agent_x, agent_y, agent_w, agent_h)
                })
                print(f"❌ MISMATCH {region_name}:")
                print(f"   JSON:  ({json_x}, {json_y}, {json_w}, {json_h})")
                print(f"   Agent: ({agent_x}, {agent_y}, {agent_w}, {agent_h})")
            else:
                print(f"✅ {region_name}: OK")
        else:
            print(f"⚠️ {region_name}: Manquant dans l'agent")
    
    # 4. Résumé
    print(f"\n=== RÉSUMÉ ===")
    print(f"Total régions JSON: {len(json_coords)}")
    print(f"Total régions Agent: {len(agent_coords)}")
    print(f"Mismatches trouvés: {len(mismatches)}")
    
    if mismatches:
        print("\n🚨 PROBLÈME: Coordonnées non synchronisées!")
        print("L'agent utilise des coordonnées obsolètes.")
        return False
    else:
        print("\n✅ SUCCÈS: Toutes les coordonnées sont synchronisées!")
        return True

def test_specific_regions():
    """Test des régions critiques pour le poker"""
    
    print("\n=== TEST RÉGIONS CRITIQUES ===")
    
    critical_regions = [
        'new_hand_button',
        'action_buttons', 
        'resume_button',
        'hand_area',
        'community_cards'
    ]
    
    screen_capture = ScreenCapture()
    
    for region_name in critical_regions:
        if region_name in screen_capture.regions:
            region = screen_capture.regions[region_name]
            print(f"✅ {region_name}: ({region.x}, {region.y}, {region.width}, {region.height})")
        else:
            print(f"❌ {region_name}: MANQUANT")

if __name__ == "__main__":
    print("🔍 Vérification synchronisation coordonnées...")
    
    # Test principal
    sync_ok = test_coordinates_sync()
    
    # Test régions critiques
    test_specific_regions()
    
    if sync_ok:
        print("\n🎯 Toutes les coordonnées sont à jour!")
    else:
        print("\n🚨 Problème de synchronisation détecté!")
        print("L'agent doit être redémarré pour charger les nouvelles coordonnées.") 