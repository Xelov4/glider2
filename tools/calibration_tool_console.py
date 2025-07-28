#!/usr/bin/env python3
"""
🎯 OUTIL DE CALIBRATION CONSOLE
================================

Version console de l'outil de calibration pour éviter les problèmes de fenêtres OpenCV.
"""

import json
import os
import sys
import numpy as np
import cv2
import pygetwindow as gw
from typing import Optional, Dict, Any
import mss
import time

class ConsoleCalibrationTool:
    """
    Outil de calibration en mode console
    """
    
    def __init__(self):
        self.regions = {
            'hand_area': {'x': 512, 'y': 861, 'width': 240, 'height': 130, 'name': 'Zone cartes joueur'},
            'community_cards': {'x': 348, 'y': 597, 'width': 590, 'height': 170, 'name': 'Cartes communautaires'},
            'pot_area': {'x': 465, 'y': 510, 'width': 410, 'height': 70, 'name': 'Zone du pot'},
            'fold_button': {'x': 773, 'y': 1008, 'width': 120, 'height': 40, 'name': 'Bouton Fold'},
            'call_button': {'x': 937, 'y': 1010, 'width': 120, 'height': 40, 'name': 'Bouton Call'},
            'raise_button': {'x': 1105, 'y': 1006, 'width': 120, 'height': 40, 'name': 'Bouton Raise'},
            'check_button': {'x': 936, 'y': 1008, 'width': 120, 'height': 40, 'name': 'Bouton Check'},
            'all_in_button': {'x': 1267, 'y': 907, 'width': 120, 'height': 40, 'name': 'Bouton All-In'},
            'my_stack_area': {'x': 548, 'y': 1015, 'width': 200, 'height': 50, 'name': 'Stack joueur'},
            'opponent1_stack_area': {'x': 35, 'y': 657, 'width': 150, 'height': 50, 'name': 'Stack adversaire 1'},
            'opponent2_stack_area': {'x': 1102, 'y': 662, 'width': 150, 'height': 40, 'name': 'Stack adversaire 2'},
            'my_current_bet': {'x': 525, 'y': 824, 'width': 200, 'height': 30, 'name': 'Mise actuelle joueur'},
            'opponent1_current_bet': {'x': 210, 'y': 638, 'width': 110, 'height': 80, 'name': 'Mise adv1'},
            'opponent2_current_bet': {'x': 961, 'y': 645, 'width': 110, 'height': 60, 'name': 'Mise adv2'},
            'bet_slider': {'x': 747, 'y': 953, 'width': 360, 'height': 40, 'name': 'Slider de mise'},
            'bet_input': {'x': 1115, 'y': 952, 'width': 100, 'height': 25, 'name': 'Input de mise'},
            'new_hand_button': {'x': 599, 'y': 979, 'width': 100, 'height': 30, 'name': 'Bouton New Hand'},
            'resume_button': {'x': 600, 'y': 400, 'width': 120, 'height': 40, 'name': 'Bouton Reprendre'},
            'blinds_area': {'x': 1166, 'y': 326, 'width': 120, 'height': 30, 'name': 'Zone des blinds'},
            'blinds_timer': {'x': 1168, 'y': 299, 'width': 100, 'height': 20, 'name': 'Timer des blinds'},
            'my_dealer_button': {'x': 297, 'y': 843, 'width': 50, 'height': 50, 'name': 'Bouton dealer (vous)'},
            'opponent1_dealer_button': {'x': 210, 'y': 545, 'width': 50, 'height': 50, 'name': 'Bouton dealer (adv1)'},
            'opponent2_dealer_button': {'x': 1012, 'y': 547, 'width': 80, 'height': 50, 'name': 'Bouton dealer (adv2)'}
        }
        
        self.current_region = None
        self.screen_capture = mss.mss()
        
        # Charger la configuration existante si elle existe
        self.load_configuration()
    
    def find_poker_window(self) -> Optional[gw.Window]:
        """Trouve la fenêtre poker"""
        try:
            windows = gw.getAllTitles()
            for window_title in windows:
                if any(keyword in window_title.lower() for keyword in ['poker', 'betclic', 'winamax', 'pmu']):
                    return gw.getWindowsWithTitle(window_title)[0]
        except Exception as e:
            print(f"Erreur recherche fenêtre: {e}")
        return None
    
    def capture_screen(self) -> np.ndarray:
        """Capture l'écran"""
        try:
            with self.screen_capture.grab(self.screen_capture.monitors[1]) as screenshot:
                return np.array(screenshot)
        except Exception as e:
            print(f"Erreur capture écran: {e}")
            return np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    def save_configuration(self, filename: str = 'calibrated_regions.json'):
        """Sauvegarde la configuration"""
        try:
            config = {}
            for region_name, region in self.regions.items():
                config[region_name] = {
                    'x': region['x'],
                    'y': region['y'],
                    'width': region['width'],
                    'height': region['height'],
                    'name': region['name']
                }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Configuration sauvegardée dans {filename}")
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
    
    def load_configuration(self, filename: str = 'calibrated_regions.json'):
        """Charge la configuration"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                for region_name, coords in config.items():
                    if region_name in self.regions:
                        self.regions[region_name].update({
                            'x': coords['x'],
                            'y': coords['y'],
                            'width': coords['width'],
                            'height': coords['height'],
                            'name': coords.get('name', region_name.replace('_', ' ').title())
                        })
                
                print(f"✅ Configuration chargée depuis {filename}")
            else:
                print(f"⚠️  Fichier {filename} non trouvé - utilisation des valeurs par défaut")
                
        except Exception as e:
            print(f"❌ Erreur chargement: {e}")
    
    def print_regions(self):
        """Affiche toutes les régions disponibles"""
        print(f"\n=== RÉGIONS DISPONIBLES ({len(self.regions)}) ===")
        for i, (region_name, region) in enumerate(self.regions.items(), 1):
            print(f"{i:2d}. {region_name:20} - ({region['x']:4d}, {region['y']:4d}) {region['width']:3d}x{region['height']:3d} - {region['name']}")
        print("=" * 50)
    
    def edit_region(self, region_name: str):
        """Édite une région spécifique"""
        if region_name not in self.regions:
            print(f"❌ Région {region_name} non trouvée")
            return
        
        region = self.regions[region_name]
        print(f"\n🎯 ÉDITION DE LA RÉGION: {region_name}")
        print(f"Nom: {region['name']}")
        print(f"Position actuelle: ({region['x']}, {region['y']})")
        print(f"Taille actuelle: {region['width']}x{region['height']}")
        
        while True:
            print(f"\nOptions pour {region_name}:")
            print("1. Modifier X")
            print("2. Modifier Y")
            print("3. Modifier largeur")
            print("4. Modifier hauteur")
            print("5. Modifier nom")
            print("6. Réinitialiser")
            print("7. Retour")
            
            choice = input("Votre choix (1-7): ").strip()
            
            if choice == '1':
                try:
                    new_x = int(input(f"Nouvelle valeur X (actuel: {region['x']}): "))
                    region['x'] = new_x
                    print(f"✅ X mis à jour: {new_x}")
                except ValueError:
                    print("❌ Valeur invalide")
            
            elif choice == '2':
                try:
                    new_y = int(input(f"Nouvelle valeur Y (actuel: {region['y']}): "))
                    region['y'] = new_y
                    print(f"✅ Y mis à jour: {new_y}")
                except ValueError:
                    print("❌ Valeur invalide")
            
            elif choice == '3':
                try:
                    new_width = int(input(f"Nouvelle largeur (actuel: {region['width']}): "))
                    region['width'] = new_width
                    print(f"✅ Largeur mise à jour: {new_width}")
                except ValueError:
                    print("❌ Valeur invalide")
            
            elif choice == '4':
                try:
                    new_height = int(input(f"Nouvelle hauteur (actuel: {region['height']}): "))
                    region['height'] = new_height
                    print(f"✅ Hauteur mise à jour: {new_height}")
                except ValueError:
                    print("❌ Valeur invalide")
            
            elif choice == '5':
                new_name = input(f"Nouveau nom (actuel: {region['name']}): ").strip()
                if new_name:
                    region['name'] = new_name
                    print(f"✅ Nom mis à jour: {new_name}")
            
            elif choice == '6':
                # Réinitialiser aux valeurs par défaut
                default_regions = {
                    'hand_area': {'x': 512, 'y': 861, 'width': 240, 'height': 130},
                    'community_cards': {'x': 348, 'y': 597, 'width': 590, 'height': 170},
                    'pot_area': {'x': 465, 'y': 510, 'width': 410, 'height': 70},
                    'fold_button': {'x': 773, 'y': 1008, 'width': 120, 'height': 40},
                    'call_button': {'x': 937, 'y': 1010, 'width': 120, 'height': 40},
                    'raise_button': {'x': 1105, 'y': 1006, 'width': 120, 'height': 40},
                    'check_button': {'x': 936, 'y': 1008, 'width': 120, 'height': 40},
                    'all_in_button': {'x': 1267, 'y': 907, 'width': 120, 'height': 40},
                    'my_stack_area': {'x': 548, 'y': 1015, 'width': 200, 'height': 50},
                    'opponent1_stack_area': {'x': 35, 'y': 657, 'width': 150, 'height': 50},
                    'opponent2_stack_area': {'x': 1102, 'y': 662, 'width': 150, 'height': 40},
                    'my_current_bet': {'x': 525, 'y': 824, 'width': 200, 'height': 30},
                    'opponent1_current_bet': {'x': 210, 'y': 638, 'width': 110, 'height': 80},
                    'opponent2_current_bet': {'x': 961, 'y': 645, 'width': 110, 'height': 60},
                    'bet_slider': {'x': 747, 'y': 953, 'width': 360, 'height': 40},
                    'bet_input': {'x': 1115, 'y': 952, 'width': 100, 'height': 25},
                    'new_hand_button': {'x': 599, 'y': 979, 'width': 100, 'height': 30},
                    'resume_button': {'x': 600, 'y': 400, 'width': 120, 'height': 40},
                    'blinds_area': {'x': 1166, 'y': 326, 'width': 120, 'height': 30},
                    'blinds_timer': {'x': 1168, 'y': 299, 'width': 100, 'height': 20},
                    'my_dealer_button': {'x': 297, 'y': 843, 'width': 50, 'height': 50},
                    'opponent1_dealer_button': {'x': 210, 'y': 545, 'width': 50, 'height': 50},
                    'opponent2_dealer_button': {'x': 1012, 'y': 547, 'width': 80, 'height': 50}
                }
                
                if region_name in default_regions:
                    region.update(default_regions[region_name])
                    print(f"✅ Région {region_name} réinitialisée")
                else:
                    print(f"❌ Pas de valeur par défaut pour {region_name}")
            
            elif choice == '7':
                break
            
            else:
                print("❌ Choix invalide")
    
    def run_calibration(self):
        """Lance l'outil de calibration en mode console"""
        print("🎯 OUTIL DE CALIBRATION CONSOLE")
        print("=" * 50)
        
        # Trouver la fenêtre poker
        poker_window = self.find_poker_window()
        if poker_window:
            print(f"✅ Fenêtre poker trouvée: {poker_window.title}")
        else:
            print("⚠️  Aucune fenêtre poker trouvée")
        
        # Afficher les régions au démarrage
        self.print_regions()
        
        while True:
            print(f"\n📋 MENU PRINCIPAL")
            print("1. Afficher toutes les régions")
            print("2. Éditer une région")
            print("3. Sauvegarder la configuration")
            print("4. Charger la configuration")
            print("5. Tester une région (capture)")
            print("6. Quitter")
            
            choice = input("\nVotre choix (1-6): ").strip()
            
            if choice == '1':
                self.print_regions()
            
            elif choice == '2':
                self.print_regions()
                try:
                    region_index = int(input("Numéro de la région à éditer: ")) - 1
                    region_names = list(self.regions.keys())
                    if 0 <= region_index < len(region_names):
                        region_name = region_names[region_index]
                        self.edit_region(region_name)
                    else:
                        print("❌ Numéro de région invalide")
                except ValueError:
                    print("❌ Valeur invalide")
            
            elif choice == '3':
                self.save_configuration()
            
            elif choice == '4':
                self.load_configuration()
                self.print_regions()
            
            elif choice == '5':
                self.print_regions()
                try:
                    region_index = int(input("Numéro de la région à tester: ")) - 1
                    region_names = list(self.regions.keys())
                    if 0 <= region_index < len(region_names):
                        region_name = region_names[region_index]
                        self.test_region(region_name)
                    else:
                        print("❌ Numéro de région invalide")
                except ValueError:
                    print("❌ Valeur invalide")
            
            elif choice == '6':
                print("👋 Calibration terminée")
                break
            
            else:
                print("❌ Choix invalide")
    
    def test_region(self, region_name: str):
        """Teste une région en capturant et sauvegardant l'image"""
        if region_name not in self.regions:
            print(f"❌ Région {region_name} non trouvée")
            return
        
        region = self.regions[region_name]
        print(f"🔍 Test de la région: {region_name}")
        print(f"Position: ({region['x']}, {region['y']})")
        print(f"Taille: {region['width']}x{region['height']}")
        
        try:
            # Capture d'écran
            img = self.capture_screen()
            
            # Extraire la région
            x, y = region['x'], region['y']
            w, h = region['width'], region['height']
            
            # Vérifier les limites
            if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
                print("⚠️  Région hors limites de l'écran")
                return
            
            region_img = img[y:y+h, x:x+w]
            
            # Sauvegarder l'image
            os.makedirs("test_captures", exist_ok=True)
            filename = f"test_captures/{region_name}_test.png"
            cv2.imwrite(filename, region_img)
            
            print(f"✅ Image sauvegardée: {filename}")
            print(f"📏 Taille de l'image: {region_img.shape[1]}x{region_img.shape[0]}")
            
        except Exception as e:
            print(f"❌ Erreur test région: {e}")

def main():
    """Point d'entrée principal"""
    tool = ConsoleCalibrationTool()
    tool.run_calibration()

if __name__ == "__main__":
    main() 