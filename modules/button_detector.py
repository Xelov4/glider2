"""
Module unifié de détection des boutons d'interface pour l'agent IA Poker
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

class ActionType(Enum):
    FOLD = "fold"
    CALL = "call"
    CHECK = "check"
    RAISE = "raise"
    ALL_IN = "all_in"
    BET = "bet"

@dataclass
class UIButton:
    name: str
    coordinates: Tuple[int, int]
    confidence: float
    enabled: bool
    text: str = ""

class ButtonDetector:
    """
    Module unifié de détection des boutons d'interface de poker
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Templates pré-enregistrés des boutons
        self.button_templates = self.load_button_templates()
        # Zones de recherche pour optimiser la performance
        self.search_regions = {
            'action_buttons': (700, 400, 300, 100),  # x, y, width, height
            'bet_slider': (600, 350, 400, 50),
            'bet_input': (650, 300, 100, 30)
        }
        
    def load_button_templates(self) -> Dict:
        """Charge les templates d'images des boutons"""
        templates = {}
        button_types = ['fold', 'call', 'check', 'raise', 'all_in', 'bet']
        
        for btn_type in button_types:
            # Simulation des templates (en production, charger des images réelles)
            templates[btn_type] = {
                'enabled': self.create_button_template(btn_type, enabled=True),
                'disabled': self.create_button_template(btn_type, enabled=False),
                'hover': self.create_button_template(btn_type, enabled=True)
            }
        return templates
    
    def create_button_template(self, button_type: str, enabled: bool = True) -> np.ndarray:
        """Crée un template de bouton simulé"""
        # Simulation d'un template de bouton (en production, charger des images réelles)
        # Réduire la taille pour éviter les erreurs OpenCV
        template = np.zeros((20, 40, 3), dtype=np.uint8)
        
        # Différencier enabled/disabled par la couleur
        if enabled:
            template[:, :, 1] = 100  # Vert pour enabled
        else:
            template[:, :, 2] = 50   # Rouge pour disabled
            
        return template
    
    def detect_available_actions(self, screenshot: np.ndarray) -> List[UIButton]:
        """Détecte tous les boutons d'action disponibles"""
        buttons = []
        
        try:
            # Extraire la zone des boutons d'action
            x, y, w, h = self.search_regions['action_buttons']
            button_area = screenshot[y:y+h, x:x+w]
            
            for action_type in ActionType:
                button = self.detect_specific_button(button_area, action_type.value)
                if button:
                    # Ajuster les coordonnées relatives à l'écran complet
                    button.coordinates = (button.coordinates[0] + x, 
                                        button.coordinates[1] + y)
                    buttons.append(button)
                    
        except Exception as e:
            self.logger.error(f"Erreur détection boutons: {e}")
            
        return buttons
    
    def detect_specific_button(self, image: np.ndarray, button_type: str) -> Optional[UIButton]:
        """Détecte un bouton spécifique par template matching"""
        try:
            templates = self.button_templates[button_type]
            best_match = None
            best_confidence = 0.0
            
            for state, template in templates.items():
                if template is None:
                    continue
                
                # Vérifier que l'image est assez grande pour le template
                if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
                    continue
                    
                # Template matching
                result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_confidence and max_val > 0.8:  # Seuil de confiance
                    best_confidence = max_val
                    best_match = UIButton(
                        name=button_type,
                        coordinates=max_loc,
                        confidence=max_val,
                        enabled=(state == 'enabled')
                    )
                    
            return best_match
            
        except Exception as e:
            self.logger.error(f"Erreur détection bouton {button_type}: {e}")
            return None
    
    def detect_bet_controls(self, screenshot: np.ndarray) -> Dict:
        """Détecte les contrôles de mise (slider, input, boutons prédéfinis)"""
        controls = {
            'slider': None,
            'input_box': None,
            'preset_buttons': []  # 1/2 pot, pot, all-in, etc.
        }
        
        try:
            # Détecter le slider de mise
            slider_region = self.extract_region(screenshot, 'bet_slider')
            controls['slider'] = self.detect_bet_slider(slider_region)
            
            # Détecter la zone de saisie
            input_region = self.extract_region(screenshot, 'bet_input')
            controls['input_box'] = self.detect_bet_input(input_region)
            
            # Détecter les boutons de mise prédéfinis
            controls['preset_buttons'] = self.detect_preset_bet_buttons(screenshot)
            
        except Exception as e:
            self.logger.error(f"Erreur détection contrôles de mise: {e}")
            
        return controls
    
    def extract_region(self, screenshot: np.ndarray, region_name: str) -> np.ndarray:
        """Extrait une région spécifique de l'écran"""
        x, y, w, h = self.search_regions[region_name]
        return screenshot[y:y+h, x:x+w]
    
    def detect_bet_slider(self, slider_image: np.ndarray) -> Optional[Dict]:
        """Détecte le slider de mise et sa position actuelle"""
        try:
            # Convertir en niveaux de gris
            gray = cv2.cvtColor(slider_image, cv2.COLOR_BGR2GRAY)
            
            # Détecter les contours pour trouver le slider
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 500:  # Taille approximative d'un slider
                    x, y, w, h = cv2.boundingRect(contour)
                    return {
                        'bounds': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'min_pos': x,
                        'max_pos': x + w,
                        'current_pos': self.get_slider_position(slider_image, x, y, w, h)
                    }
            return None
            
        except Exception as e:
            self.logger.error(f"Erreur détection slider: {e}")
            return None
    
    def get_slider_position(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> int:
        """Détermine la position actuelle du curseur du slider"""
        try:
            # Chercher le curseur (partie plus foncée/claire)
            slider_region = image[y:y+h, x:x+w]
            gray = cv2.cvtColor(slider_region, cv2.COLOR_BGR2GRAY)
            
            # Trouver les variations de luminosité pour localiser le curseur
            profile = np.mean(gray, axis=0)
            cursor_pos = np.argmin(profile)  # ou argmax selon l'apparence
            
            return x + cursor_pos
            
        except Exception as e:
            self.logger.error(f"Erreur position slider: {e}")
            return x + w//2  # Position par défaut au centre
    
    def detect_bet_input(self, input_image: np.ndarray) -> Optional[Dict]:
        """Détecte la zone de saisie de mise"""
        try:
            # Simulation de détection d'input box
            height, width = input_image.shape[:2]
            return {
                'bounds': (0, 0, width, height),
                'center': (width//2, height//2),
                'clickable': True
            }
        except Exception as e:
            self.logger.error(f"Erreur détection input: {e}")
            return None
    
    def detect_preset_bet_buttons(self, screenshot: np.ndarray) -> List[Dict]:
        """Détecte les boutons de mise prédéfinis (1/2 pot, pot, all-in)"""
        preset_buttons = []
        
        try:
            # Simulation de détection de boutons prédéfinis
            button_types = ['half_pot', 'pot', 'all_in']
            
            for i, btn_type in enumerate(button_types):
                # Position simulée des boutons
                x = 600 + i * 80
                y = 350
                
                preset_buttons.append({
                    'type': btn_type,
                    'coordinates': (x, y),
                    'enabled': True
                })
                
        except Exception as e:
            self.logger.error(f"Erreur détection boutons prédéfinis: {e}")
            
        return preset_buttons 