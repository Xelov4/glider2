"""
🔍 Module d'Analyse d'Images - Détection de Cartes et OCR
========================================================

Ce module gère la reconnaissance visuelle des éléments de poker :
- Détection des cartes (template matching + OCR)
- Reconnaissance des jetons et montants
- Extraction de texte avec Tesseract
- Validation et filtrage des résultats

FONCTIONNALITÉS
===============

✅ Détection multi-méthodes des cartes
✅ OCR robuste pour les montants
✅ Template matching optimisé
✅ Validation automatique des résultats
✅ Gestion d'erreurs complète

MÉTHODES PRINCIPALES
====================

- detect_cards() : Détection principale des cartes
- extract_text() : OCR avec Tesseract
- detect_chips() : Reconnaissance des jetons
- validate_card() : Validation des cartes détectées

TEMPLATES
=========

Les templates de cartes sont chargés depuis templates/cards/ :
- Rangs : 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A
- Couleurs : ♠, ♥, ♦, ♣

VERSION: 2.0.0
DERNIÈRE MISE À JOUR: 2025-07-27
"""

import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from skimage import measure, morphology
from skimage.filters import threshold_otsu
import os

@dataclass
class Card:
    """Représente une carte de poker"""
    rank: str  # A, K, Q, J, 10, 9, ..., 2
    suit: str  # ♠, ♥, ♦, ♣
    confidence: float  # Confiance de la détection (0-1)
    
    def __str__(self):
        return f"{self.rank}{self.suit}"
    
    def __repr__(self):
        return f"Card({self.rank}{self.suit}, conf={self.confidence:.2f})"

class ImageAnalyzer:
    """
    Module d'analyse d'images pour reconnaissance de cartes, jetons, boutons
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.card_templates = self.load_card_templates()
        self.ocr_config = r'--oem 3 --psm 6 outputbase digits'
        
        # Configuration Tesseract
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read('config.ini')
            if 'Tesseract' in config and 'tesseract_path' in config['Tesseract']:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = config['Tesseract']['tesseract_path']
                self.logger.info("Tesseract configuré avec le chemin personnalisé")
        except Exception as e:
            self.logger.warning(f"Configuration Tesseract échouée: {e}")
        
        # Couleurs pour détection des cartes
        self.card_colors = {
            'red': ([0, 100, 100], [10, 255, 255]),  # Rouge (♥, ♦)
            'black': ([0, 0, 0], [180, 255, 30])     # Noir (♠, ♣)
        }
        
        # Templates pour les boutons
        self.button_templates = {
            'fold': self.create_button_template('FOLD'),
            'call': self.create_button_template('CALL'),
            'raise': self.create_button_template('RAISE'),
            'check': self.create_button_template('CHECK'),
            'bet': self.create_button_template('BET')
        }
    
    def load_card_templates(self) -> Dict:
        """
        Charge les templates de cartes depuis le dossier templates/cards
        """
        templates = {}
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        
        # Charger les templates de rangs disponibles
        for rank in ranks:
            rank_file = f"templates/cards/ranks/card_{rank}.png"
            if os.path.exists(rank_file):
                try:
                    template = cv2.imread(rank_file)
                    if template is not None:
                        templates[f"rank_{rank}"] = template
                        self.logger.info(f"Template rang chargé: {rank} ({template.shape})")
                    else:
                        self.logger.warning(f"Template rang {rank} chargé mais None")
                except Exception as e:
                    self.logger.warning(f"Erreur chargement template rang {rank}: {e}")
            else:
                self.logger.warning(f"Fichier template rang non trouvé: {rank_file}")
        
        # Créer des templates combinés pour les cartes complètes
        # Pour l'instant, on utilise juste les rangs car les couleurs ne sont pas disponibles
        for rank in ranks:
            if f"rank_{rank}" in templates:
                # Créer un template de carte complète basé sur le rang
                templates[f"card_{rank}"] = templates[f"rank_{rank}"]
        
        self.logger.info(f"Templates de cartes chargés: {len(templates)} templates")
        return templates
    
    def create_button_template(self, text: str) -> np.ndarray:
        """
        Crée un template pour un bouton (simulation)
        """
        # Simulation d'un template de bouton
        template = np.zeros((30, 80, 3), dtype=np.uint8)
        # En production, on utiliserait des images réelles des boutons
        return template
    
    def detect_cards(self, image: np.ndarray) -> List[Card]:
        """
        Détecte les cartes dans l'image avec une approche plus robuste
        """
        cards = []
        
        try:
            # Vérifier que l'image est valide
            if image is None or image.size == 0:
                self.logger.debug("Image vide pour détection de cartes")
                return cards
                
            # Préprocessing de l'image
            processed_image = self.preprocess_image(image)
            
            # APPROCHE 1: Template matching avec les templates de cartes
            template_cards = self._detect_cards_by_template_matching(processed_image)
            if template_cards:
                cards.extend(template_cards)
                self.logger.info(f"Cartes détectées par template: {[str(c) for c in template_cards]}")
            
            # APPROCHE 2: OCR direct pour détecter les rangs et couleurs
            if not cards:  # Si template matching échoue
                ocr_cards = self._detect_cards_by_ocr(processed_image)
                if ocr_cards:
                    cards.extend(ocr_cards)
                    self.logger.info(f"Cartes détectées par OCR: {[str(c) for c in ocr_cards]}")
            
            # APPROCHE 3: Détection par contours si les autres échouent
            if not cards:
                contour_cards = self._detect_cards_by_contours(processed_image)
                if contour_cards:
                    cards.extend(contour_cards)
                    self.logger.info(f"Cartes détectées par contours: {[str(c) for c in contour_cards]}")
            
            # Validation finale des cartes détectées
            valid_cards = []
            for card in cards:
                if self._validate_card(card):
                    valid_cards.append(card)
            
            self.logger.debug(f"Détecté {len(valid_cards)} cartes valides: {valid_cards}")
            return valid_cards
                        
        except Exception as e:
            self.logger.error(f"Erreur détection cartes: {e}")
            return cards
    
    def _detect_cards_by_template_matching(self, image: np.ndarray) -> List[Card]:
        """Détecte les cartes par template matching"""
        cards = []
        
        try:
            # Diviser l'image en régions potentielles de cartes
            card_regions = self._extract_card_regions(image)
            
            for region in card_regions:
                # Template matching pour chaque template de carte
                best_match = None
                best_confidence = 0.0
                
                for card_key, template in self.card_templates.items():
                    try:
                        if template is not None and template.size > 0:
                            # Vérifier que le template n'est pas plus grand que la région
                            if template.shape[0] > region.shape[0] or template.shape[1] > region.shape[1]:
                                # Redimensionner le template pour correspondre à la région
                                resized_template = cv2.resize(template, (region.shape[1], region.shape[0]))
                            else:
                                resized_template = template
                            
                            # Template matching
                            result = cv2.matchTemplate(region, resized_template, cv2.TM_CCOEFF_NORMED)
                            confidence = np.max(result)
                            
                            if confidence > best_confidence and confidence > 0.3:  # Seuil de confiance plus bas
                                best_confidence = confidence
                                best_match = card_key
                                self.logger.debug(f"Match trouvé: {card_key} avec confiance {confidence:.2f}")
                                
                    except Exception as e:
                        self.logger.debug(f"Erreur template matching {card_key}: {e}")
                        continue
                
                if best_match:
                    # Créer l'objet Card
                    rank, suit = self._parse_card_key(best_match)
                    card = Card(rank=rank, suit=suit, confidence=best_confidence)
                    cards.append(card)
                    
        except Exception as e:
            self.logger.error(f"Erreur template matching: {e}")
            
        return cards
    
    def _detect_cards_by_ocr(self, image: np.ndarray) -> List[Card]:
        """Détecte les cartes par OCR direct"""
        cards = []
        
        try:
            import pytesseract
            
            # Extraire le texte avec OCR
            text = pytesseract.image_to_string(image, config='--psm 6').upper()
            self.logger.debug(f"OCR texte: '{text}'")
            
            # Chercher les rangs et couleurs
            ranks = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']
            suits = ['♠', '♥', '♦', '♣', 'S', 'H', 'D', 'C']
            
            # Mapping pour les erreurs OCR courantes
            ocr_mapping = {
                '12': 'Q',  # 12 peut être Q
                '13': 'K',  # 13 peut être K
                '14': 'A',  # 14 peut être A
                '1': 'A',   # 1 peut être A
                '0': '10',  # 0 peut être 10
            }
            
            # Nettoyer le texte
            cleaned_text = text.replace(' ', '').replace('\n', '')
            self.logger.debug(f"Texte nettoyé: '{cleaned_text}'")
            
            # Chercher les patterns de cartes
            for rank in ranks:
                if rank in cleaned_text:
                    for suit in suits:
                        if suit in cleaned_text:
                            # Vérifier que rank et suit sont proches
                            rank_pos = cleaned_text.find(rank)
                            suit_pos = cleaned_text.find(suit)
                            if abs(rank_pos - suit_pos) <= 3:  # Proximité réduite
                                card = Card(rank=rank, suit=suit, confidence=0.8)
                                cards.append(card)
                                self.logger.debug(f"Carte détectée par OCR: {card}")
            
            # Si aucune carte trouvée, essayer avec le mapping OCR
            if not cards:
                for ocr_text, real_rank in ocr_mapping.items():
                    if ocr_text in cleaned_text:
                        for suit in suits:
                            if suit in cleaned_text:
                                card = Card(rank=real_rank, suit=suit, confidence=0.7)
                                cards.append(card)
                                self.logger.debug(f"Carte détectée par mapping OCR: {card}")
                                break
                        
        except Exception as e:
            self.logger.error(f"Erreur OCR cartes: {e}")
            
        return cards
    
    def _detect_cards_by_contours(self, image: np.ndarray) -> List[Card]:
        """Détecte les cartes par analyse de contours"""
        cards = []
        
        try:
            # Conversion en niveaux de gris
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Seuillage adaptatif
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Recherche de contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filtre les petits contours
                    # Approximation du rectangle
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    # Extraction de la région de la carte
                    x, y, w, h = cv2.boundingRect(contour)
                    card_region = image[y:y+h, x:x+w]
                    
                    # Analyse de la carte par OCR
                    card = self._analyze_card_region_ocr(card_region)
                    if card:
                        cards.append(card)
                        
        except Exception as e:
            self.logger.error(f"Erreur détection par contours: {e}")
            
        return cards
    
    def _extract_card_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """Extrait les régions potentielles de cartes"""
        regions = []
        
        try:
            # Diviser l'image en zones (approche simple)
            height, width = image.shape[:2]
            
            # Pour les cartes du joueur (hand_area) - généralement 2 cartes
            if width > height:  # Image horizontale
                # Diviser en 2 zones pour 2 cartes
                card_width = width // 2
                regions.append(image[:, :card_width])
                regions.append(image[:, card_width:])
            else:
                # Image verticale - prendre toute l'image
                regions.append(image)
                
        except Exception as e:
            self.logger.error(f"Erreur extraction régions: {e}")
            
        return regions
    
    def _parse_card_key(self, card_key: str) -> Tuple[str, str]:
        """Parse une clé de carte (ex: 'Ah') en rank et suit"""
        try:
            if len(card_key) >= 2:
                rank = card_key[0]
                suit = card_key[1]
                return rank, suit
            return 'A', '♠'  # Valeur par défaut
        except:
            return 'A', '♠'
    
    def _analyze_card_region_ocr(self, card_region: np.ndarray) -> Optional[Card]:
        """Analyse une région de carte par OCR"""
        try:
            import pytesseract
            
            # Extraire le texte
            text = pytesseract.image_to_string(card_region, config='--psm 6').upper()
            
            # Chercher rank et suit
            ranks = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']
            suits = ['♠', '♥', '♦', '♣', 'S', 'H', 'D', 'C']
            
            for rank in ranks:
                if rank in text:
                    for suit in suits:
                        if suit in text:
                            return Card(rank=rank, suit=suit, confidence=0.6)
                            
            return None
            
        except Exception as e:
            self.logger.debug(f"Erreur analyse OCR région: {e}")
            return None
    
    def analyze_card_region(self, card_region: np.ndarray) -> Optional[Card]:
        """
        Analyse une région de carte pour déterminer rank et suit avec validation OCR
        """
        try:
            # VALIDATION OCR POUR CONFIRMER QUE C'EST UNE CARTE
            if not self._validate_card_with_ocr(card_region):
                return None
            # Vérification des dimensions et type de données
            if card_region is None or card_region.size == 0:
                return None
                
            # S'assurer que card_region est en uint8
            if card_region.dtype != np.uint8:
                card_region = card_region.astype(np.uint8)
            
            # Template matching pour chaque carte possible
            best_match = None
            best_confidence = 0.0
            
            for card_key, template in self.card_templates.items():
                try:
                    if template.shape[0] > 0 and template.shape[1] > 0:
                        # S'assurer que template est en uint8
                        if template.dtype != np.uint8:
                            template = template.astype(np.uint8)
                        
                        # Redimensionnement pour matching
                        resized_template = cv2.resize(template, (card_region.shape[1], card_region.shape[0]))
                        
                        # Vérification de compatibilité des dimensions
                        if (card_region.shape[0] >= resized_template.shape[0] and 
                            card_region.shape[1] >= resized_template.shape[1]):
                            
                            # Template matching
                            result = cv2.matchTemplate(card_region, resized_template, cv2.TM_CCOEFF_NORMED)
                            confidence = np.max(result)
                            
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_match = card_key
                                
                except Exception as template_error:
                    self.logger.debug(f"Erreur template {card_key}: {template_error}")
                    continue
            
            if best_match and best_confidence > 0.6:  # Seuil de confiance
                rank = best_match[:-1]
                suit = best_match[-1]
                return Card(rank=rank, suit=suit, confidence=best_confidence)
                
        except Exception as e:
            self.logger.error(f"Erreur analyse carte: {e}")
            
        return None
    
    def _validate_card_with_ocr(self, card_region: np.ndarray) -> bool:
        """Valide qu'une région contient bien une carte avec OCR"""
        try:
            import pytesseract
            
            # Extraire le texte avec OCR
            text = pytesseract.image_to_string(card_region, config='--psm 6').lower()
            
            # Mots-clés qui indiquent une carte de poker
            card_indicators = [
                'a', 'k', 'q', 'j', '10', '9', '8', '7', '6', '5', '4', '3', '2',  # Rangs
                '♠', '♥', '♦', '♣', 'spades', 'hearts', 'diamonds', 'clubs'  # Couleurs
            ]
            
            # Vérifier si le texte contient des indicateurs de carte
            for indicator in card_indicators:
                if indicator in text:
                    self.logger.debug(f"✅ Validation carte réussie: '{text}'")
                    return True
            
            self.logger.debug(f"❌ Validation carte échouée: '{text}'")
            return False
            
        except Exception as e:
            self.logger.error(f"Erreur validation carte OCR: {e}")
            return False
    
    def _validate_card(self, card: Card) -> bool:
        """Valide une carte détectée"""
        try:
            # Vérifier que le rang est valide
            valid_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
            if card.rank not in valid_ranks:
                self.logger.debug(f"Rang invalide: {card.rank}")
                return False
            
            # Vérifier que la couleur est valide
            valid_suits = ['♠', '♥', '♦', '♣', '?']
            if card.suit not in valid_suits:
                self.logger.debug(f"Couleur invalide: {card.suit}")
                return False
            
            # Vérifier la confiance
            if card.confidence < 0.1:
                self.logger.debug(f"Confiance trop faible: {card.confidence}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Erreur validation carte: {e}")
            return False
    
    def detect_chips(self, image: np.ndarray) -> Dict[str, int]:
        """
        Détecte les montants de jetons
        """
        chip_amounts = {}
        
        try:
            # Conversion en HSV pour détection des couleurs de jetons
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Détection des cercles (jetons)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 1, 20,
                param1=50, param2=30, minRadius=10, maxRadius=50
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                
                for circle in circles[0, :]:
                    x, y, radius = circle
                    
                    # Extraction de la région du jeton
                    chip_region = image[max(0, y-radius):min(image.shape[0], y+radius),
                                      max(0, x-radius):min(image.shape[1], x+radius)]
                    
                    # Estimation de la valeur du jeton basée sur la couleur
                    chip_value = self.estimate_chip_value(chip_region)
                    
                    if chip_value > 0:
                        chip_amounts[f"chip_{x}_{y}"] = chip_value
                        
        except Exception as e:
            self.logger.error(f"Erreur détection jetons: {e}")
            
        return chip_amounts
    
    def estimate_chip_value(self, chip_region: np.ndarray) -> int:
        """
        Estime la valeur d'un jeton basée sur sa couleur
        """
        try:
            # Analyse des couleurs dominantes
            hsv = cv2.cvtColor(chip_region, cv2.COLOR_BGR2HSV)
            
            # Calcul de la couleur moyenne
            mean_color = np.mean(hsv, axis=(0, 1))
            hue = mean_color[0]
            
            # Mapping couleur -> valeur (approximatif)
            if hue < 30:  # Rouge/Orange
                return 25
            elif hue < 60:  # Jaune
                return 100
            elif hue < 120:  # Vert
                return 500
            elif hue < 180:  # Bleu
                return 1000
            else:  # Violet/Blanc
                return 5000
                
        except:
            return 0
    
    def detect_buttons(self, image: np.ndarray) -> List[str]:
        """
        Détecte les boutons d'action disponibles (rouges = actifs, gris = inactifs)
        """
        available_buttons = []
        
        try:
            # Conversion en HSV pour détection de couleur
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Détection des boutons rouges (actifs)
            # Rouge en HSV (0-10 et 170-180)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            # Détection des boutons gris (inactifs)
            lower_gray = np.array([0, 0, 50])
            upper_gray = np.array([180, 30, 150])
            mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
            
            # Calculer le pourcentage de pixels rouges vs gris
            total_pixels = mask_red.shape[0] * mask_red.shape[1]
            red_pixels = cv2.countNonZero(mask_red)
            gray_pixels = cv2.countNonZero(mask_gray)
            
            red_percentage = red_pixels / total_pixels
            gray_percentage = gray_pixels / total_pixels
            
            # Si plus de 5% de pixels rouges, les boutons sont actifs
            if red_percentage > 0.05:
                # Détecter quels boutons sont présents (template matching simplifié)
                button_names = ['fold', 'call', 'raise', 'check', 'bet']
                for button_name in button_names:
                    # Simulation de détection (en production, utiliser de vrais templates)
                    available_buttons.append(button_name)
                    
                self.logger.info(f"Boutons actifs détectés (rouge: {red_percentage:.2%})")
            else:
                self.logger.info(f"Boutons inactifs (gris: {gray_percentage:.2%})")
                        
        except Exception as e:
            self.logger.error(f"Erreur détection boutons: {e}")
            
        return available_buttons
    
    def read_text_amount(self, image: np.ndarray) -> int:
        """
        Lit un montant textuel via OCR
        """
        try:
            # Prétraitement pour OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Amélioration du contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Seuillage
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR avec gestion d'erreur robuste
            try:
                text = pytesseract.image_to_string(thresh, config=self.ocr_config)
                
                # Extraction des chiffres
                import re
                numbers = re.findall(r'\d+', text)
                
                if numbers:
                    return int(numbers[0])
                    
            except Exception as ocr_error:
                # Si Tesseract n'est pas installé, utiliser une estimation basée sur la couleur
                self.logger.warning(f"Tesseract non disponible, utilisation de l'estimation par couleur: {ocr_error}")
                return self.estimate_amount_by_color(image)
                
        except Exception as e:
            self.logger.error(f"Erreur OCR: {e}")
            
        return 0
    
    def estimate_amount_by_color(self, image: np.ndarray) -> int:
        """
        Estime un montant basé sur la couleur dominante (alternative à l'OCR)
        """
        try:
            # Conversion en HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calcul de la couleur moyenne
            mean_color = np.mean(hsv, axis=(0, 1))
            hue = mean_color[0]
            saturation = mean_color[1]
            value = mean_color[2]
            
            # Estimation basée sur la couleur dominante
            if value < 50:  # Très sombre
                return 0
            elif saturation < 30:  # Gris/blanc
                return 1000
            elif hue < 30:  # Rouge/Orange
                return 25
            elif hue < 60:  # Jaune
                return 100
            elif hue < 120:  # Vert
                return 500
            elif hue < 180:  # Bleu
                return 1000
            else:  # Violet/Blanc
                return 5000
                
        except Exception as e:
            self.logger.error(f"Erreur estimation couleur: {e}")
            return 0
    
    def detect_pot_size(self, image: np.ndarray) -> int:
        """
        Détecte la taille du pot
        """
        try:
            # Recherche de la zone du pot (généralement au centre)
            height, width = image.shape[:2]
            pot_region = image[height//3:2*height//3, width//3:2*width//3]
            
            # OCR sur la zone du pot
            pot_amount = self.read_text_amount(pot_region)
            
            return pot_amount
            
        except Exception as e:
            self.logger.error(f"Erreur détection pot: {e}")
            return 0
    
    def detect_stack_size(self, image: np.ndarray) -> int:
        """
        Détecte la taille d'un stack de joueur
        """
        try:
            # Utiliser la méthode de lecture de texte
            stack_amount = self.read_text_amount(image)
            return stack_amount
            
        except Exception as e:
            self.logger.error(f"Erreur détection stack: {e}")
            return 0
    
    def detect_blinds_timer(self, image: np.ndarray) -> int:
        """
        Détecte le temps restant avant l'augmentation des blinds (en secondes)
        """
        try:
            # Lecture du timer (format MM:SS ou SS)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            try:
                text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:')
                
                # Parsing du timer (format MM:SS ou SS)
                import re
                time_pattern = r'(\d+):(\d+)|(\d+)'
                match = re.search(time_pattern, text)
                
                if match:
                    if match.group(1) and match.group(2):  # Format MM:SS
                        minutes = int(match.group(1))
                        seconds = int(match.group(2))
                        return minutes * 60 + seconds
                    elif match.group(3):  # Format SS
                        return int(match.group(3))
                        
            except Exception as ocr_error:
                self.logger.warning(f"Erreur OCR timer: {ocr_error}")
                return 300  # Valeur par défaut: 5 minutes
                
        except Exception as e:
            self.logger.error(f"Erreur détection timer: {e}")
            return 300  # Valeur par défaut: 5 minutes
    
    def detect_dealer_button(self, image: np.ndarray) -> bool:
        """
        Détecte si le bouton dealer est présent dans l'image
        """
        try:
            # Conversion en HSV pour détection de couleur
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Détection du bouton dealer (généralement blanc ou jaune)
            # Masque pour les couleurs claires (blanc, jaune)
            lower_light = np.array([0, 0, 200])  # Blanc
            upper_light = np.array([180, 30, 255])
            mask_light = cv2.inRange(hsv, lower_light, upper_light)
            
            # Masque pour le jaune (bouton dealer typique)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Combiner les masques
            mask_combined = cv2.bitwise_or(mask_light, mask_yellow)
            
            # Calculer le pourcentage de pixels détectés
            total_pixels = mask_combined.shape[0] * mask_combined.shape[1]
            detected_pixels = cv2.countNonZero(mask_combined)
            percentage = detected_pixels / total_pixels
            
            # Seuil de détection (ajuster selon l'interface)
            return percentage > 0.1  # 10% de pixels détectés
            
        except Exception as e:
            self.logger.error(f"Erreur détection bouton dealer: {e}")
            return False
    
    def is_my_turn(self, image: np.ndarray) -> bool:
        """
        Détecte si c'est notre tour de jouer (boutons rouges)
        """
        try:
            # Conversion en HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Détection des boutons rouges (actifs)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            # Calculer le pourcentage de pixels rouges
            total_pixels = mask_red.shape[0] * mask_red.shape[1]
            red_pixels = cv2.countNonZero(mask_red)
            red_percentage = red_pixels / total_pixels
            
            # Seuil de détection (ajuster selon l'interface)
            is_turn = red_percentage > 0.05
            
            if is_turn:
                self.logger.info(f"C'est notre tour (rouge: {red_percentage:.2%})")
            else:
                self.logger.info(f"Pas notre tour (rouge: {red_percentage:.2%})")
                
            return is_turn
            
        except Exception as e:
            self.logger.error(f"Erreur détection tour: {e}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Prétraitement d'image pour améliorer la reconnaissance
        """
        try:
            # Redimensionnement
            height, width = image.shape[:2]
            if width > 800:
                scale = 800 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # Réduction du bruit
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            
            # Amélioration du contraste
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Erreur prétraitement: {e}")
            return image
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extrait le texte d'une image via OCR
        """
        try:
            # Prétraitement pour OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Amélioration du contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Seuillage
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR avec gestion d'erreur robuste
            try:
                text = pytesseract.image_to_string(thresh, config=self.ocr_config)
                return text.strip()
                
            except Exception as ocr_error:
                # Si Tesseract n'est pas installé, utiliser une estimation basée sur la couleur
                self.logger.warning(f"Tesseract non disponible, utilisation de l'estimation par couleur: {ocr_error}")
                return str(self.estimate_amount_by_color(image))
                
        except Exception as e:
            self.logger.error(f"Erreur OCR: {e}")
            return "" 