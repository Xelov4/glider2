#!/usr/bin/env python3
"""
Script de debug pour vérifier les coordonnées utilisées par l'agent
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.screen_capture import ScreenCapture
from modules.automation import AutomationEngine

def main():
    print("=== DEBUG COORDONNÉES AGENT ===")
    
    # 1. Charger le JSON
    try:
        with open('calibrated_regions.json', 'r') as f:
            json_regions = json.load(f)
        print(f"✅ JSON chargé: {len(json_regions)} régions")
    except Exception as e:
        print(f"❌ Erreur chargement JSON: {e}")
        return
    
    # 2. Initialiser l'agent
    try:
        screen_capture = ScreenCapture()
        automation = AutomationEngine()
        print(f"✅ Agent initialisé")
    except Exception as e:
        print(f"❌ Erreur initialisation agent: {e}")
        return
    
    # 3. Comparer les coordonnées
    print("\n=== COMPARAISON COORDONNÉES ===")
    
    for region_name in json_regions.keys():
        try:
            # Coordonnées du JSON
            json_coords = json_regions[region_name]
            json_x, json_y = json_coords['x'], json_coords['y']
            json_w, json_h = json_coords['width'], json_coords['height']
            
            # Coordonnées de l'agent
            region_info = screen_capture.get_region_info(region_name)
            if region_info:
                agent_x, agent_y = region_info['x'], region_info['y']
                agent_w, agent_h = region_info['width'], region_info['height']
                
                # Vérifier si identiques
                if (json_x == agent_x and json_y == agent_y and 
                    json_w == agent_w and json_h == agent_h):
                    print(f"✅ {region_name:20} - IDENTIQUES")
                else:
                    print(f"❌ {region_name:20} - DIFFÉRENTES!")
                    print(f"   JSON:  ({json_x:4d}, {json_y:4d}) {json_w:3d}x{json_h:3d}")
                    print(f"   Agent: ({agent_x:4d}, {agent_y:4d}) {agent_w:3d}x{agent_h:3d}")
            else:
                print(f"❌ {region_name:20} - NON TROUVÉE dans l'agent")
                
        except Exception as e:
            print(f"❌ {region_name:20} - Erreur: {e}")
    
    # 4. Test des clics
    print("\n=== TEST CLICS ===")
    
    test_regions = ['resume_button', 'new_hand_button']
    
    for region_name in test_regions:
        try:
            region_info = screen_capture.get_region_info(region_name)
            if region_info:
                x, y = region_info['x'], region_info['y']
                w, h = region_info['width'], region_info['height']
                center_x = x + w // 2
                center_y = y + h // 2
                
                print(f"🎯 {region_name:15} - Centre: ({center_x:4d}, {center_y:4d})")
                print(f"   Rectangle: ({x:4d}, {y:4d}) {w:3d}x{h:3d}")
            else:
                print(f"❌ {region_name:15} - Région non trouvée")
                
        except Exception as e:
            print(f"❌ {region_name:15} - Erreur: {e}")
    
    print("\n=== FIN DEBUG ===")

if __name__ == "__main__":
    main() 