"""
Script pour visualiser les régions et identifier les problèmes de calibration
"""

import cv2
import numpy as np
from modules.screen_capture import ScreenCapture
import json

def visualize_regions():
    """Visualise toutes les régions pour identifier les problèmes"""
    print("=== VISUALISATION DES RÉGIONS ===")
    
    # Charger les régions calibrées
    screen_capture = ScreenCapture()
    
    # Liste des régions importantes
    important_regions = [
        'hand_area',
        'community_cards', 
        'pot_area',
        'my_stack_area',
        'fold_button',
        'call_button',
        'raise_button',
        'check_button',
        'all_in_button',
        'new_hand_button',
        'resume_button'
    ]
    
    print(f"Régions à vérifier: {len(important_regions)}")
    
    for region_name in important_regions:
        print(f"\n🔍 Vérification de {region_name}...")
        
        try:
            # Capturer la région
            region_image = screen_capture.capture_region(region_name)
            
            if region_image is not None:
                print(f"✅ {region_name}: {region_image.shape}")
                
                # Sauvegarder l'image
                filename = f"debug_{region_name}.png"
                cv2.imwrite(filename, region_image)
                print(f"   📁 Sauvegardé: {filename}")
                
                # Extraire le texte pour debug
                from modules.image_analysis import ImageAnalyzer
                analyzer = ImageAnalyzer()
                text = analyzer.extract_text(region_image)
                if text.strip():
                    print(f"   📝 Texte détecté: '{text.strip()}'")
                else:
                    print(f"   📝 Aucun texte détecté")
                    
            else:
                print(f"❌ {region_name}: Impossible de capturer")
                
        except Exception as e:
            print(f"❌ Erreur pour {region_name}: {e}")
    
    print("\n=== VISUALISATION TERMINÉE ===")
    print("Images sauvegardées avec préfixe 'debug_'")
    print("Vérifiez ces images pour identifier les problèmes de calibration")

if __name__ == "__main__":
    visualize_regions() 