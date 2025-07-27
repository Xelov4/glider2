"""
Outil de calibration pour ajuster les coordonnées des régions de capture
"""

import cv2
import numpy as np
import mss
import pygetwindow as gw
import json
import os
from typing import Dict, Tuple, Optional

class CalibrationTool:
    """
    Outil pour calibrer les coordonnées des régions de capture
    """
    
    def __init__(self):
        self.sct = mss.mss()
        self.regions = {
            # Zones principales (essentielles)
            'hand_area': {'x': 400, 'y': 500, 'width': 200, 'height': 100, 'name': 'Cartes du joueur'},
            'community_cards': {'x': 300, 'y': 300, 'width': 400, 'height': 80, 'name': 'Cartes communes'},
            'pot_area': {'x': 350, 'y': 250, 'width': 300, 'height': 50, 'name': 'Zone du pot'},
            'action_buttons': {'x': 350, 'y': 600, 'width': 500, 'height': 150, 'name': 'Boutons d\'action (ROUGE=TOUR)'},
            
            # Zones de stacks
            'my_stack_area': {'x': 50, 'y': 100, 'width': 150, 'height': 400, 'name': 'Votre stack'},
            'opponent1_stack_area': {'x': 200, 'y': 100, 'width': 150, 'height': 400, 'name': 'Stack adversaire 1'},
            'opponent2_stack_area': {'x': 350, 'y': 100, 'width': 150, 'height': 400, 'name': 'Stack adversaire 2'},
            
            # Zones de mises
            'current_bet_to_call': {'x': 400, 'y': 350, 'width': 200, 'height': 30, 'name': 'Mise à payer'},
            'my_current_bet': {'x': 400, 'y': 350, 'width': 200, 'height': 30, 'name': 'Votre mise actuelle'},
            'opponent1_current_bet': {'x': 200, 'y': 350, 'width': 200, 'height': 30, 'name': 'Mise adversaire 1'},
            'opponent2_current_bet': {'x': 600, 'y': 350, 'width': 200, 'height': 30, 'name': 'Mise adversaire 2'},
            'bet_slider': {'x': 400, 'y': 550, 'width': 200, 'height': 20, 'name': 'Slider de mise'},
            'bet_input': {'x': 400, 'y': 580, 'width': 100, 'height': 25, 'name': 'Input de mise'},
            
            # Zones de navigation
            'new_hand_button': {'x': 500, 'y': 700, 'width': 100, 'height': 30, 'name': 'Bouton New Hand'},
            'sit_out_button': {'x': 600, 'y': 700, 'width': 100, 'height': 30, 'name': 'Bouton Sit Out'},
            'leave_table_button': {'x': 800, 'y': 700, 'width': 100, 'height': 30, 'name': 'Leave Table'},
            
            # Zones avancées
            'blinds_area': {'x': 200, 'y': 200, 'width': 150, 'height': 40, 'name': 'Zone des blinds'},
            'blinds_timer': {'x': 200, 'y': 250, 'width': 100, 'height': 30, 'name': 'Timer des blinds'},
            
            # Zones bouton dealer
            'my_dealer_button': {'x': 400, 'y': 800, 'width': 50, 'height': 50, 'name': 'Bouton dealer (vous)'},
            'opponent1_dealer_button': {'x': 200, 'y': 300, 'width': 50, 'height': 50, 'name': 'Bouton dealer (adv1)'},
            'opponent2_dealer_button': {'x': 600, 'y': 300, 'width': 50, 'height': 50, 'name': 'Bouton dealer (adv2)'}
        }
        self.current_region = None
        self.dragging = False
        self.start_pos = None
        
    def find_poker_window(self) -> Optional[gw.Window]:
        """Trouve la fenêtre de poker"""
        try:
            windows = gw.getAllTitles()
            poker_windows = [w for w in windows if 'poker' in w.lower() or 'stars' in w.lower()]
            
            if poker_windows:
                return gw.getWindowsWithTitle(poker_windows[0])[0]
            return None
        except Exception as e:
            print(f"Erreur recherche fenêtre: {e}")
            return None
    
    def capture_screen(self) -> np.ndarray:
        """Capture l'écran complet"""
        try:
            screenshot = self.sct.grab(self.sct.monitors[1])  # Écran principal
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
        except Exception as e:
            print(f"Erreur capture écran: {e}")
            return np.zeros((800, 1200, 3), dtype=np.uint8)
    
    def draw_regions(self, img: np.ndarray) -> np.ndarray:
        """Dessine les régions sur l'image"""
        img_copy = img.copy()
        
        for region_name, region in self.regions.items():
            x, y, w, h = region['x'], region['y'], region['width'], region['height']
            
            # Couleur selon la région active
            if self.current_region == region_name:
                color = (0, 255, 0)  # Vert pour région active
                thickness = 3
            else:
                color = (255, 0, 0)  # Rouge pour régions normales
                thickness = 2
            
            # Rectangle
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
            
            # Nom de la région
            cv2.putText(img_copy, region['name'], (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Coordonnées
            coord_text = f"({x},{y}) {w}x{h}"
            cv2.putText(img_copy, coord_text, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img_copy
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback pour les événements de souris"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Trouver la région cliquée
            for region_name, region in self.regions.items():
                rx, ry, rw, rh = region['x'], region['y'], region['width'], region['height']
                if rx <= x <= rx + rw and ry <= y <= ry + rh:
                    self.current_region = region_name
                    self.dragging = True
                    self.start_pos = (x, y)
                    break
        
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self.current_region and self.start_pos:
                dx = x - self.start_pos[0]
                dy = y - self.start_pos[1]
                
                # Déplacer la région
                self.regions[self.current_region]['x'] += dx
                self.regions[self.current_region]['y'] += dy
                self.start_pos = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.start_pos = None
    
    def resize_region(self, region_name: str, new_width: int, new_height: int):
        """Redimensionne une région"""
        if region_name in self.regions:
            self.regions[region_name]['width'] = new_width
            self.regions[region_name]['height'] = new_height
    
    def save_configuration(self, filename: str = 'calibrated_regions.json'):
        """Sauvegarde la configuration calibrée"""
        try:
            # Nettoyer les données pour la sauvegarde
            config = {}
            for region_name, region in self.regions.items():
                config[region_name] = {
                    'x': region['x'],
                    'y': region['y'],
                    'width': region['width'],
                    'height': region['height']
                }
            
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Configuration sauvegardée dans {filename}")
            
        except Exception as e:
            print(f"Erreur sauvegarde: {e}")
    
    def load_configuration(self, filename: str = 'calibrated_regions.json'):
        """Charge une configuration sauvegardée"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                for region_name, coords in config.items():
                    if region_name in self.regions:
                        self.regions[region_name].update(coords)
                
                print(f"Configuration chargée depuis {filename}")
            else:
                print(f"Fichier {filename} non trouvé")
                
        except Exception as e:
            print(f"Erreur chargement: {e}")
    
    def run_calibration(self):
        """Lance l'outil de calibration interactif"""
        print("=== OUTIL DE CALIBRATION ===")
        print("Instructions:")
        print("1. Cliquez sur une région pour la sélectionner (devient verte)")
        print("2. Glissez pour déplacer la région")
        print("3. Utilisez les touches:")
        print("   - 'S'/'Z' : Augmenter/diminuer la hauteur")
        print("   - 'Q'/'D' : Diminuer/augmenter la largeur")
        print("   - 'r' : Réinitialiser la région")
        print("   - 'P' : Sauvegarder")
        print("   - 'l' : Charger")
        print("   - 'I' : Quitter")
        print("4. Assurez-vous que votre fenêtre poker est visible")
        
        # Trouver la fenêtre poker
        poker_window = self.find_poker_window()
        if poker_window:
            print(f"Fenêtre poker trouvée: {poker_window.title}")
        else:
            print("Aucune fenêtre poker trouvée - utilisez l'écran complet")
        
        # Créer la fenêtre
        cv2.namedWindow('Calibration Tool', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Calibration Tool', self.mouse_callback)
        
        while True:
            # Capture d'écran
            img = self.capture_screen()
            
            # Dessiner les régions
            img_with_regions = self.draw_regions(img)
            
            # Afficher
            cv2.imshow('Calibration Tool', img_with_regions)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('i'):
                break
            elif key == ord('p'):
                self.save_configuration()
            elif key == ord('l'):
                self.load_configuration()
            elif key == ord('r') and self.current_region:
                # Réinitialiser la région actuelle
                self.regions[self.current_region]['x'] = 400
                self.regions[self.current_region]['y'] = 500
                self.regions[self.current_region]['width'] = 200
                self.regions[self.current_region]['height'] = 100
            elif self.current_region:
                if key == ord('s'):
                    self.regions[self.current_region]['height'] += 10
                elif key == ord('z'):
                    self.regions[self.current_region]['height'] = max(10, self.regions[self.current_region]['height'] - 10)
                elif key == ord('q'):
                    self.regions[self.current_region]['width'] = max(10, self.regions[self.current_region]['width'] - 10)
                elif key == ord('d'):
                    self.regions[self.current_region]['width'] += 10
        
        cv2.destroyAllWindows()
        print("Calibration terminée")

def main():
    """Point d'entrée principal"""
    tool = CalibrationTool()
    tool.run_calibration()

if __name__ == "__main__":
    main() 