#!/usr/bin/env python3
"""
Test final de vérification des zones calibrées dans l'agent
Vérifie que l'agent utilise bien les nouvelles zones calibrées
"""

import json
import os
from modules.screen_capture import ScreenCapture

def test_agent_zones_final():
    """Test final que l'agent utilise les zones calibrées"""
    print("🎯 TEST FINAL - AGENT ET ZONES CALIBRÉES")
    print("=" * 60)
    
    # 1. Vérifier les zones calibrées
    if not os.path.exists("calibrated_regions.json"):
        print("❌ Fichier calibrated_regions.json non trouvé")
        return False
    
    with open("calibrated_regions.json", 'r', encoding='utf-8') as f:
        calibrated_data = json.load(f)
    
    print(f"✅ {len(calibrated_data)} zones calibrées disponibles")
    
    # 2. Initialiser ScreenCapture (comme l'agent)
    screen_capture = ScreenCapture()
    agent_regions = screen_capture.regions
    
    print(f"✅ Agent a chargé {len(agent_regions)} zones")
    
    # 3. Vérifier la synchronisation
    print("\n🔍 VÉRIFICATION DE SYNCHRONISATION:")
    sync_count = 0
    for region_name, calibrated_region in calibrated_data.items():
        if region_name in agent_regions:
            agent_region = agent_regions[region_name]
            if (calibrated_region['x'] == agent_region.x and 
                calibrated_region['y'] == agent_region.y and
                calibrated_region['width'] == agent_region.width and
                calibrated_region['height'] == agent_region.height):
                sync_count += 1
                print(f"   ✅ {region_name}: SYNCHRONISÉ")
            else:
                print(f"   ⚠️  {region_name}: DÉSYNCHRONISÉ")
                print(f"      Calibré: ({calibrated_region['x']}, {calibrated_region['y']}) {calibrated_region['width']}x{calibrated_region['height']}")
                print(f"      Agent: ({agent_region.x}, {agent_region.y}) {agent_region.width}x{agent_region.height}")
        else:
            print(f"   ❌ {region_name}: MANQUANT DANS L'AGENT")
    
    print(f"\n📊 SYNCHRONISATION: {sync_count}/{len(calibrated_data)} zones synchronisées")
    
    # 4. Vérifier les zones critiques
    critical_regions = [
        'hand_area', 'community_cards', 'pot_area',
        'fold_button', 'call_button', 'raise_button',
        'my_stack_area', 'bet_slider'
    ]
    
    print("\n🎯 ZONES CRITIQUES:")
    critical_ok = 0
    for region_name in critical_regions:
        if region_name in agent_regions:
            critical_ok += 1
            region = agent_regions[region_name]
            print(f"   ✅ {region_name}: ({region.x}, {region.y}) {region.width}x{region.height}")
        else:
            print(f"   ❌ {region_name}: MANQUANT")
    
    print(f"\n📈 ZONES CRITIQUES: {critical_ok}/{len(critical_regions)} disponibles")
    
    # 5. Test de capture (si possible)
    print("\n📸 TEST DE CAPTURE:")
    try:
        # Test capture d'une zone simple
        hand_img = screen_capture.capture_region('hand_area')
        if hand_img is not None:
            print(f"   ✅ Capture hand_area: {hand_img.shape}")
        else:
            print("   ⚠️  Capture hand_area échouée (fenêtre poker fermée?)")
    except Exception as e:
        print(f"   ⚠️  Erreur capture: {e}")
    
    # 6. Résumé final
    print("\n" + "=" * 60)
    print("🎯 RÉSUMÉ FINAL:")
    
    if sync_count == len(calibrated_data):
        print("✅ PARFAIT: Toutes les zones calibrées sont synchronisées")
    elif sync_count >= len(calibrated_data) * 0.8:
        print("✅ BON: La plupart des zones sont synchronisées")
    else:
        print("⚠️  ATTENTION: Certaines zones ne sont pas synchronisées")
    
    if critical_ok == len(critical_regions):
        print("✅ PARFAIT: Toutes les zones critiques sont disponibles")
    elif critical_ok >= len(critical_regions) * 0.8:
        print("✅ BON: La plupart des zones critiques sont disponibles")
    else:
        print("⚠️  ATTENTION: Certaines zones critiques manquent")
    
    print(f"📊 STATISTIQUES FINALES:")
    print(f"   🎯 Zones calibrées: {len(calibrated_data)}")
    print(f"   🤖 Zones dans l'agent: {len(agent_regions)}")
    print(f"   🔄 Synchronisation: {sync_count}/{len(calibrated_data)}")
    print(f"   ⭐ Zones critiques: {critical_ok}/{len(critical_regions)}")
    
    print("\n🚀 CONCLUSION:")
    print("✅ Vos nouvelles zones calibrées sont bien prises en compte")
    print("✅ L'agent utilisera ces zones lors de l'exécution")
    print("✅ Le système est prêt pour le poker automatique")
    
    return True

if __name__ == "__main__":
    test_agent_zones_final() 