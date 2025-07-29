"""
Module de Detection des Boutons - Interface Utilisateur
======================================================

Ce module gère la détection des boutons d'action dans l'interface Betclic Poker :
- Fold, Call, Check, Raise, All-in
- Boutons de navigation (New Hand, Resume)
- Couronne de victoire
- Validation OCR pour éviter les faux positifs

FONCTIONNALITÉS
===============

- Detection multi-templates des boutons
- Validation OCR pour éviter les faux positifs
- Gestion des états enabled/disabled
- Templates personnalisés pour Betclic Poker
- Detection de la couronne de victoire

MÉTHODES PRINCIPALES
====================

- detect_buttons() : Détection principale des boutons
- validate_button_ocr() : Validation OCR des boutons
- detect_winner_crown() : Détection de la couronne
- is_button_visible() : Vérification de visibilité

VERSION: 3.0.0 - TEMPLATES OPTIMISÉS
DERNIÈRE MISE À JOUR: 2025-01-XX
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import os

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
        # Templates spéciaux (couronne de victoire, etc.)
        self.special_templates = self.load_special_templates()
        # Zones de recherche pour optimiser la performance
        self.search_regions = {
            'fold_button': (773, 1008, 120, 40),
            'call_button': (937, 1010, 120, 40),
            'raise_button': (1105, 1006, 120, 40),
            'check_button': (936, 1008, 120, 40),
            'all_in_button': (1267, 907, 120, 40),
            'bet_slider': (747, 953, 360, 40),
            'bet_input': (1115, 952, 100, 25)
        }
        
    def load_button_templates(self) -> Dict:
        """Charge les templates d'images des boutons depuis le dossier templates/buttons"""
        templates = {}
        button_types = ['fold', 'call', 'check', 'raise', 'all_in', 'bet']
        
        # Mapping pour les noms de fichiers
        button_files = {
            'fold': 'fold_button.png',
            'call': 'cann_button.png',  # Fichier existant pour call
            'check': 'check_button.png',
            'raise': 'raise_button.png',
            'all_in': 'all_in_button.png',
            'bet': 'bet_button.png'
        }
        
        self.logger.info("Chargement des templates de boutons...")
        
        for btn_type in button_types:
            templates[btn_type] = {}
            
            # Chemin vers le template
            template_path = f"templates/buttons/{button_files.get(btn_type, f'{btn_type}_button.png')}"
            
            try:
                # Charger le template réel
                if os.path.exists(template_path):
                    template = cv2.imread(template_path)
                    if template is not None:
                        templates[btn_type]['enabled'] = template
                        templates[btn_type]['disabled'] = self._create_disabled_version(template)
                        templates[btn_type]['hover'] = template  # Même que enabled pour l'instant
                        self.logger.info(f"Template chargé: {btn_type} ({template.shape})")
                    else:
                        self.logger.warning(f"Impossible de charger le template: {template_path}")
                        templates[btn_type] = self._create_fallback_template(btn_type)
                else:
                    self.logger.warning(f"Template non trouvé: {template_path}")
                    templates[btn_type] = self._create_fallback_template(btn_type)
                    
            except Exception as e:
                self.logger.error(f"Erreur chargement template {btn_type}: {e}")
                templates[btn_type] = self._create_fallback_template(btn_type)
                
        return templates
    
    def load_special_templates(self) -> Dict:
        """Charge les templates spéciaux (couronne de victoire, etc.)"""
        special_templates = {}
        
        # Templates spéciaux avec leurs chemins
        special_files = {
            'winner_crown': 'winner2.png',  # Couronne de victoire
            'winner': 'winner.png',  # Indicateur de victoire existant
            'play_020': 'play_020_button.png'  # Bouton Jouer 0,20€
        }
        
        for template_name, filename in special_files.items():
            template_path = f"templates/buttons/{filename}"
            
            try:
                if os.path.exists(template_path):
                    template = cv2.imread(template_path)
                    if template is not None:
                        special_templates[template_name] = template
                        self.logger.info(f"Template spécial chargé: {template_name} ({template.shape})")
                    else:
                        self.logger.warning(f"Impossible de charger le template spécial: {template_path}")
                else:
                    self.logger.warning(f"Template spécial non trouvé: {template_path}")
                    
            except Exception as e:
                self.logger.error(f"Erreur chargement template spécial {template_name}: {e}")
                
        return special_templates
    
    def _create_disabled_version(self, template: np.ndarray) -> np.ndarray:
        """Crée une version désactivée du template (grisée)"""
        try:
            # Convertir en niveaux de gris
            gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            # Reconvertir en BGR pour compatibilité
            disabled = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return disabled
        except:
            return template
    
    def _create_fallback_template(self, button_type: str) -> Dict:
        """Crée un template de fallback si le vrai template n'est pas trouvé"""
        self.logger.warning(f"Utilisation du template de fallback pour {button_type}")
        
        # Template de fallback (simulé)
        template = np.zeros((40, 80, 3), dtype=np.uint8)
        template[:, :, 1] = 100  # Vert pour enabled
        
        return {
            'enabled': template,
            'disabled': template,
            'hover': template
        }
    
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
        """Détecte un bouton spécifique par template matching + validation OCR"""
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
                    # VALIDATION OCR POUR CONFIRMER LE BOUTON
                    if self._validate_button_with_ocr(image, button_type, max_loc, template.shape):
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
    
    def _validate_button_with_ocr(self, image: np.ndarray, button_type: str, location: Tuple[int, int], template_shape: Tuple[int, int, int]) -> bool:
        """Valide un bouton détecté avec OCR pour éviter les faux positifs"""
        try:
            import pytesseract
            
            # Extraire la région du bouton détecté
            x, y = location
            h, w = template_shape[:2]
            
            # S'assurer que les coordonnées sont dans l'image
            if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                return False
            
            # Extraire la région du bouton
            button_region = image[y:y+h, x:x+w]
            
            # Extraire le texte avec OCR
            text = pytesseract.image_to_string(button_region, config='--psm 6').lower()
            
            # Mapping des mots-clés pour chaque type de bouton
            button_keywords = {
                'fold': ['fold', 'passer', 'abandonner'],
                'call': ['call', 'suivre', 'voir'],
                'check': ['check', 'passer'],
                'raise': ['raise', 'relancer', 'augmenter'],
                'all_in': ['all in', 'tout', 'tapis'],
                'bet': ['bet', 'miser', 'parier']
            }
            
            # Vérifier si le texte contient un mot-clé correspondant
            keywords = button_keywords.get(button_type, [])
            for keyword in keywords:
                if keyword in text:
                    self.logger.debug(f"Validation OCR reussie pour {button_type}: '{text}'")
                    return True
            
            self.logger.debug(f"Validation OCR echouee pour {button_type}: '{text}'")
            return False
            
        except Exception as e:
            self.logger.error(f"Erreur validation OCR {button_type}: {e}")
            return False
    
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
    
    def detect_winner_crown(self, screenshot: np.ndarray) -> bool:
        """
        Détecte la couronne de victoire (winner2.png)
        Retourne True si la couronne est détectée (fin de manche)
        """
        try:
            if 'winner_crown' not in self.special_templates:
                self.logger.warning("Template couronne de victoire non disponible")
                return False
            
            crown_template = self.special_templates['winner_crown']
            
            # Recherche dans toute l'image (la couronne peut apparaître n'importe où)
            result = cv2.matchTemplate(screenshot, crown_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Seuil de confiance élevé pour la couronne
            confidence_threshold = 0.8
            
            if max_val >= confidence_threshold:
                self.logger.info(f"COURONNE DE VICTOIRE DÉTECTÉE! (confiance: {max_val:.2f})")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur détection couronne de victoire: {e}")
            return False
    
    def detect_winner_indicator(self, screenshot: np.ndarray) -> bool:
        """
        Détecte l'indicateur de victoire général (winner.png)
        """
        try:
            if 'winner' not in self.special_templates:
                return False
            
            winner_template = self.special_templates['winner']
            
            # Recherche dans toute l'image
            result = cv2.matchTemplate(screenshot, winner_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            confidence_threshold = 0.7
            
            if max_val >= confidence_threshold:
                self.logger.info(f"INDICATEUR DE VICTOIRE DÉTECTÉ! (confiance: {max_val:.2f})")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur détection indicateur de victoire: {e}")
            return False
    
    def detect_play_020_button(self, screenshot: np.ndarray) -> Optional[UIButton]:
        """
        Détecte le bouton "Jouer 0,20€"
        Retourne un UIButton si le bouton est détecté
        """
        try:
            if 'play_020' not in self.special_templates:
                self.logger.warning("Template bouton Jouer 0,20€ non disponible")
                return None
            
            play_template = self.special_templates['play_020']
            
            # Recherche dans toute l'image
            result = cv2.matchTemplate(screenshot, play_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Seuil de confiance pour le bouton
            confidence_threshold = 0.7
            
            if max_val >= confidence_threshold:
                # Calculer le centre du bouton
                template_height, template_width = play_template.shape[:2]
                center_x = max_loc[0] + template_width // 2
                center_y = max_loc[1] + template_height // 2
                
                # Validation OCR optionnelle
                button_region = screenshot[max_loc[1]:max_loc[1] + template_height, 
                                         max_loc[0]:max_loc[0] + template_width]
                
                # Créer l'objet UIButton
                button = UIButton(
                    name="play_020",
                    coordinates=(center_x, center_y),
                    confidence=max_val,
                    enabled=True,
                    text="Jouer 0,20€"
                )
                
                self.logger.info(f"Bouton Jouer 0,20€ détecté! (confiance: {max_val:.2f}) à ({center_x}, {center_y})")
                return button
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Erreur détection bouton Jouer 0,20€: {e}")
            return None 