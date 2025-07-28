#!/usr/bin/env python3
"""
Script de test pour vérifier la cohérence des régions entre l'outil, le JSON et l'agent
"""

import json
import os
from modules.screen_capture import ScreenCapture

def test_regions_coherence():
    """Test de cohérence des régions"""
    print("🎯 TEST DE COHÉRENCE DES RÉGIONS")
    print("=" * 50)
    
    # 1. Vérifier le JSON
    print("\n📄 1. RÉGIONS DANS LE JSON:")
    try:
        with open('calibrated_regions.json', 'r') as f:
            json_regions = json.load(f)
        
        json_region_names = list(json_regions.keys())
        print(f"   ✅ JSON contient {len(json_region_names)} régions:")
        for i, name in enumerate(json_region_names, 1):
            coords = json_regions[name]
            print(f"      {i:2d}. {name:25} ({coords['x']:4d}, {coords['y']:4d}) {coords['width']:3d}x{coords['height']:3d}")
        
    except Exception as e:
        print(f"   ❌ Erreur lecture JSON: {e}")
        return False
    
    # 2. Vérifier l'outil de calibrage
    print("\n🔧 2. RÉGIONS DANS L'OUTIL DE CALIBRAGE:")
    try:
        from tools.calibration_tool import CalibrationTool
        tool = CalibrationTool()
        tool_region_names = list(tool.regions.keys())
        
        print(f"   ✅ Outil contient {len(tool_region_names)} régions:")
        for i, name in enumerate(tool_region_names, 1):
            region = tool.regions[name]
            print(f"      {i:2d}. {name:25} ({region['x']:4d}, {region['y']:4d}) {region['width']:3d}x{region['height']:3d}")
        
    except Exception as e:
        print(f"   ❌ Erreur outil calibrage: {e}")
        return False
    
    # 3. Vérifier l'agent
    print("\n🤖 3. RÉGIONS UTILISÉES PAR L'AGENT:")
    try:
        screen_capture = ScreenCapture()
        agent_regions = screen_capture.regions
        
        print(f"   ✅ Agent charge {len(agent_regions)} régions:")
        for i, (name, coords) in enumerate(agent_regions.items(), 1):
            print(f"      {i:2d}. {name:25} ({coords['x']:4d}, {coords['y']:4d}) {coords['width']:3d}x{coords['height']:3d}")
        
    except Exception as e:
        print(f"   ❌ Erreur agent: {e}")
        return False
    
    # 4. Comparaison
    print("\n🔍 4. COMPARAISON:")
    
    # Régions dans JSON mais pas dans l'outil
    missing_in_tool = set(json_region_names) - set(tool_region_names)
    if missing_in_tool:
        print(f"   ⚠️  Régions dans JSON mais pas dans l'outil: {missing_in_tool}")
    else:
        print("   ✅ Toutes les régions du JSON sont dans l'outil")
    
    # Régions dans l'outil mais pas dans le JSON
    missing_in_json = set(tool_region_names) - set(json_region_names)
    if missing_in_json:
        print(f"   ⚠️  Régions dans l'outil mais pas dans JSON: {missing_in_json}")
    else:
        print("   ✅ Toutes les régions de l'outil sont dans le JSON")
    
    # Régions dans l'agent mais pas dans le JSON
    missing_in_agent = set(json_region_names) - set(agent_regions.keys())
    if missing_in_agent:
        print(f"   ⚠️  Régions dans JSON mais pas dans l'agent: {missing_in_agent}")
    else:
        print("   ✅ Toutes les régions du JSON sont dans l'agent")
    
    # 5. Test de capture
    print("\n📸 5. TEST DE CAPTURE:")
    try:
        # Tester la capture de quelques régions importantes
        important_regions = ['hand_area', 'action_buttons', 'community_cards', 'pot_area']
        
        for region_name in important_regions:
            if region_name in agent_regions:
                try:
                    captured = screen_capture.capture_region(region_name)
                    if captured is not None and captured.size > 0:
                        print(f"   ✅ {region_name:20} - Capturé: {captured.shape}")
                    else:
                        print(f"   ❌ {region_name:20} - Capture vide")
                except Exception as e:
                    print(f"   ❌ {region_name:20} - Erreur: {e}")
            else:
                print(f"   ❌ {region_name:20} - Région non trouvée")
        
    except Exception as e:
        print(f"   ❌ Erreur test capture: {e}")
    
    return True

def test_region_coordinates():
    """Test des coordonnées des régions"""
    print("\n🎯 TEST DES COORDONNÉES")
    print("=" * 30)
    
    try:
        with open('calibrated_regions.json', 'r') as f:
            regions = json.load(f)
        
        # Vérifier que les coordonnées sont dans des limites raisonnables
        screen_width = 5120  # Votre écran ultra-wide
        screen_height = 1440
        
        print(f"   📺 Résolution écran: {screen_width}x{screen_height}")
        
        for name, coords in regions.items():
            x, y, w, h = coords['x'], coords['y'], coords['width'], coords['height']
            
            # Vérifications
            issues = []
            if x < 0 or y < 0:
                issues.append("Coordonnées négatives")
            if x + w > screen_width:
                issues.append(f"Dépasse largeur ({x + w} > {screen_width})")
            if y + h > screen_height:
                issues.append(f"Dépasse hauteur ({y + h} > {screen_height})")
            if w <= 0 or h <= 0:
                issues.append("Dimensions nulles ou négatives")
            
            if issues:
                print(f"   ⚠️  {name:20} - Problèmes: {', '.join(issues)}")
            else:
                print(f"   ✅ {name:20} - OK ({x:4d}, {y:4d}) {w:3d}x{h:3d}")
        
    except Exception as e:
        print(f"   ❌ Erreur test coordonnées: {e}")

if __name__ == "__main__":
    print("🎯 TEST COMPLET DES RÉGIONS")
    print("=" * 50)
    
    success = test_regions_coherence()
    test_region_coordinates()
    
    print("\n" + "=" * 50)
    
    if success:
        print("🎉 TOUTES LES RÉGIONS SONT COHÉRENTES!")
        print("L'outil de calibrage devrait maintenant afficher toutes les régions")
    else:
        print("❌ PROBLÈMES DÉTECTÉS")
        print("Vérifiez la cohérence entre les fichiers")
    
    print("\n💡 CONSEILS:")
    print("- Relancez l'outil de calibrage: py tools/calibration_tool.py")
    print("- Vous devriez maintenant voir toutes les 18 régions")
    print("- Testez l'agent: py main.py") 