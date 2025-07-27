"""
🔍 Module d'Analyse d'Images - Détection de Cartes et OCR
========================================================

Ce module gère la reconnaissance visuelle des éléments de poker :
- Détection des cartes (OCR uniquement)
- Reconnaissance des jetons et montants
- Extraction de texte avec Tesseract
- Validation et filtrage des résultats

FONCTIONNALITÉS
===============

✅ Détection OCR pure des cartes
✅ OCR robuste pour les montants
✅ Validation automatique des résultats
✅ Gestion d'erreurs complète

MÉTHODES PRINCIPALES
====================

- detect_cards() : Détection principale des cartes (OCR uniquement)
- extract_text() : OCR avec Tesseract
- detect_chips() : Reconnaissance des jetons
- validate_card() : Validation des cartes détectées

VERSION: 2.1.0 - OCR PUR
DERNIÈRE MISE À JOUR: 2025-07-27
"""

import cv2
import numpy as np
import pytesseract
import logging
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

@dataclass
class Card:
    rank: str
    suit: str
    confidence: float
    position: Tuple[int, int]

class ImageAnalyzer:
    def __init__(self, tesseract_path: str = None):
        self.logger = logging.getLogger(__name__)
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Configuration OCR optimisée pour les cartes
        self.ocr_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=23456789TJQKA♠♥♦♣'
        
        # Mapping des erreurs OCR courantes
        self.ocr_mapping = {
            '12': 'Q', '13': 'K', '14': 'A', '15': 'A',
            'T1': 'T', 'T0': 'T', 'T7': 'T',
            'J1': 'J', 'J0': 'J',
            'Q1': 'Q', 'Q0': 'Q',
            'K1': 'K', 'K0': 'K',
            'A1': 'A', 'A0': 'A',
            'ra': 'A', 'rb': 'K', 'rc': 'Q', 'rd': 'J',
            're': 'T', 'rf': '9', 'rg': '8', 'rh': '7'
        }

    def detect_cards(self, image: np.ndarray, region_name: str = "hand_area") -> List[Card]:
        """
        Détecte les cartes avec une approche multi-méthodes robuste
        """
        try:
            self.logger.debug(f"Détection de cartes dans {region_name}")
            
            # APPROCHE 1: OCR avec prétraitement optimisé
            ocr_cards = self._detect_cards_ocr_optimized(image)
            if ocr_cards:
                self.logger.debug(f"Cartes détectées par OCR: {[f'{c.rank}{c.suit}' for c in ocr_cards]}")
                return ocr_cards
            
            # APPROCHE 2: Détection par contours et analyse de forme
            contour_cards = self._detect_cards_by_contours(image)
            if contour_cards:
                self.logger.debug(f"Cartes détectées par contours: {[f'{c.rank}{c.suit}' for c in contour_cards]}")
                return contour_cards
            
            # APPROCHE 3: Détection par couleur (rouge/noir)
            color_cards = self._detect_cards_by_color(image)
            if color_cards:
                self.logger.debug(f"Cartes détectées par couleur: {[f'{c.rank}{c.suit}' for c in color_cards]}")
                return color_cards
            
            return []
            
        except Exception as e:
            self.logger.error(f"Erreur détection cartes: {e}")
            return []

    def debug_card_detection(self, image: np.ndarray, region_name: str = "debug") -> Dict:
        """
        Méthode de debug pour analyser la détection de cartes
        """
        debug_info = {
            'region': region_name,
            'image_shape': image.shape if image is not None else None,
            'ocr_results': [],
            'detected_cards': [],
            'errors': []
        }
        
        try:
            if image is None or image.size == 0:
                debug_info['errors'].append("Image vide")
                return debug_info
            
            # Test OCR avec différentes configurations
            configs = [
                ('Standard', '--oem 3 --psm 6'),
                ('Dense', '--oem 3 --psm 8'),
                ('Single char', '--oem 3 --psm 10'),
                ('Cards only', '--oem 3 --psm 6 -c tessedit_char_whitelist=23456789TJQKA♠♥♦♣'),
                ('Cards dense', '--oem 3 --psm 8 -c tessedit_char_whitelist=23456789TJQKA♠♥♦♣'),
            ]
            
            for config_name, config in configs:
                try:
                    processed = self._preprocess_for_ocr(image)
                    text = pytesseract.image_to_string(processed, config=config)
                    debug_info['ocr_results'].append({
                        'config': config_name,
                        'text': text.strip(),
                        'length': len(text.strip())
                    })
                except Exception as e:
                    debug_info['errors'].append(f"OCR {config_name}: {e}")
            
            # Test de détection complète
            detected_cards = self.detect_cards(image, region_name)
            debug_info['detected_cards'] = [f"{c.rank}{c.suit}" for c in detected_cards]
            
            # Analyse des couleurs
            try:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                red_pixels = cv2.countNonZero(cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])))
                black_pixels = cv2.countNonZero(cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 30])))
                debug_info['color_analysis'] = {
                    'red_pixels': red_pixels,
                    'black_pixels': black_pixels,
                    'total_pixels': image.shape[0] * image.shape[1]
                }
            except Exception as e:
                debug_info['errors'].append(f"Analyse couleur: {e}")
            
            return debug_info
            
        except Exception as e:
            debug_info['errors'].append(f"Erreur générale: {e}")
            return debug_info

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Prétraitement optimisé pour la détection de cartes
        """
        try:
            # Conversion en niveaux de gris
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Redimensionnement pour améliorer l'OCR
            height, width = gray.shape
            if width < 100:  # Image trop petite
                scale = 2.0
                new_width = int(width * scale)
                new_height = int(height * scale)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Amélioration du contraste avec CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Réduction du bruit avec filtre bilatéral
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Seuillage adaptatif multiple
            # Méthode 1: Seuillage adaptatif gaussien
            binary1 = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Méthode 2: Seuillage adaptatif moyen
            binary2 = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                cv2.THRESH_BINARY, 15, 3
            )
            
            # Méthode 3: Seuillage Otsu
            _, binary3 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Combiner les résultats
            combined = cv2.bitwise_and(binary1, binary2)
            combined = cv2.bitwise_or(combined, binary3)
            
            # Morphologie pour nettoyer
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            self.logger.error(f"Erreur prétraitement OCR: {e}")
            return image

    def _detect_cards_ocr_optimized(self, image: np.ndarray) -> List[Card]:
        """
        OCR optimisé pour les cartes avec plusieurs configurations
        """
        cards = []
        
        # Configuration 1: OCR standard
        config1 = '--oem 3 --psm 6 -c tessedit_char_whitelist=23456789TJQKA♠♥♦♣'
        cards.extend(self._try_ocr_config(image, config1))
        
        # Configuration 2: OCR pour texte dense
        config2 = '--oem 3 --psm 8 -c tessedit_char_whitelist=23456789TJQKA♠♥♦♣'
        cards.extend(self._try_ocr_config(image, config2))
        
        # Configuration 3: OCR pour caractères individuels
        config3 = '--oem 3 --psm 10 -c tessedit_char_whitelist=23456789TJQKA♠♥♦♣'
        cards.extend(self._try_ocr_config(image, config3))
        
        # Dédupliquer et valider
        unique_cards = self._deduplicate_cards(cards)
        return [card for card in unique_cards if self._validate_card(card)]

    def _try_ocr_config(self, image: np.ndarray, config: str) -> List[Card]:
        """
        Essaie une configuration OCR spécifique
        """
        try:
            # Prétraitement spécifique
            processed = self._preprocess_for_ocr(image)
            
            # OCR avec la configuration
            text = pytesseract.image_to_string(
                processed, 
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            # Parser le résultat
            return self._parse_ocr_text_advanced(text['text'])
            
        except Exception as e:
            self.logger.debug(f"OCR config échouée: {e}")
            return []

    def _parse_ocr_text_advanced(self, text: str) -> List[Card]:
        """
        Parse avancé du texte OCR avec plus de patterns
        """
        cards = []
        try:
            # Nettoyer le texte
            clean_text = text.strip().upper()
            self.logger.debug(f"Texte OCR brut: '{clean_text}'")
            
            # Pattern 1: "AS KH QD JC" (rang + couleur)
            pattern1 = r'([2-9TJQKA])([♠♥♦♣SHDC])'
            import re
            matches1 = re.findall(pattern1, clean_text)
            for rank, suit in matches1:
                card = Card(
                    rank=rank,
                    suit=self._normalize_suit(suit),
                    confidence=0.9,
                    position=(0, 0)
                )
                cards.append(card)
            
            # Pattern 2: "A♠ K♥" (avec symboles)
            pattern2 = r'([2-9TJQKA])([♠♥♦♣])'
            matches2 = re.findall(pattern2, clean_text)
            for rank, suit in matches2:
                card = Card(
                    rank=rank,
                    suit=suit,
                    confidence=0.95,
                    position=(0, 0)
                )
                cards.append(card)
            
            # Pattern 3: Caractères individuels proches
            if len(clean_text) >= 2:
                for i in range(len(clean_text) - 1):
                    char1 = clean_text[i]
                    char2 = clean_text[i + 1]
                    
                    # Vérifier si c'est un rang + couleur
                    if char1 in '23456789TJQKA' and char2 in '♠♥♦♣SHDC':
                        card = Card(
                            rank=char1,
                            suit=self._normalize_suit(char2),
                            confidence=0.8,
                            position=(0, 0)
                        )
                        cards.append(card)
            
            # NOUVEAU: Pattern 4 - Détection de rangs seuls (pour cartes sans couleur visible)
            # Chercher tous les rangs dans le texte
            rank_pattern = r'([2-9TJQKA])'
            rank_matches = re.findall(rank_pattern, clean_text)
            
            # Si on trouve des rangs mais pas de couleurs, créer des cartes avec couleur par défaut
            if rank_matches and not cards:
                for rank in rank_matches:
                    # Créer une carte avec couleur par défaut
                    card = Card(
                        rank=rank,
                        suit='?',  # Couleur inconnue
                        confidence=0.6,  # Confiance réduite
                        position=(0, 0)
                    )
                    cards.append(card)
                    self.logger.debug(f"Carte détectée (rang seul): {rank}?")
            
            # NOUVEAU: Pattern 5 - Détection de doubles (paires)
            # Chercher les répétitions de rangs
            for rank in '23456789TJQKA':
                count = clean_text.count(rank)
                if count >= 2:
                    # Probablement une paire
                    card = Card(
                        rank=rank,
                        suit='?',
                        confidence=0.7,
                        position=(0, 0)
                    )
                    cards.append(card)
                    self.logger.debug(f"Paire détectée: {rank}{rank}")
            
            # Corriger les erreurs OCR courantes
            corrected_cards = []
            for card in cards:
                # Corriger le rang
                corrected_rank = self.ocr_mapping.get(card.rank, card.rank)
                if corrected_rank != card.rank:
                    card.rank = corrected_rank
                    card.confidence *= 0.9  # Réduire la confiance si correction
                
                corrected_cards.append(card)
            
            # Dédupliquer les cartes
            unique_cards = self._deduplicate_cards(corrected_cards)
            
            self.logger.debug(f"Cartes parsées: {[f'{c.rank}{c.suit}' for c in unique_cards]}")
            return unique_cards
            
        except Exception as e:
            self.logger.error(f"Erreur parsing OCR avancé: {e}")
            return []

    def _detect_cards_by_contours(self, image: np.ndarray) -> List[Card]:
        """
        Détecte les cartes par analyse de contours
        """
        try:
            cards = []
            
            # Conversion en niveaux de gris
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Seuillage adaptatif
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Recherche de contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filtre les petits contours
                    # Approximation du rectangle
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    # Extraction de la région
                    x, y, w, h = cv2.boundingRect(contour)
                    card_region = image[y:y+h, x:x+w]
                    
                    # Analyser la région
                    card = self._analyze_card_region(card_region)
                    if card:
                        cards.append(card)
            
            return cards
            
        except Exception as e:
            self.logger.error(f"Erreur détection par contours: {e}")
            return []

    def _analyze_card_region(self, card_region: np.ndarray) -> Optional[Card]:
        """
        Analyse une région de carte pour déterminer rank et suit
        """
        try:
            if card_region.size == 0:
                return None
            
            # OCR sur la région
            text = pytesseract.image_to_string(
                card_region, 
                config='--oem 3 --psm 6 -c tessedit_char_whitelist=23456789TJQKA♠♥♦♣',
                output_type=pytesseract.Output.DICT
            )
            
            # Analyser le texte
            clean_text = text['text'].strip().upper()
            if len(clean_text) >= 2:
                rank = clean_text[0]
                suit = clean_text[1]
                
                # Valider
                if rank in '23456789TJQKA' and suit in '♠♥♦♣SHDC':
                    return Card(
                        rank=rank,
                        suit=self._normalize_suit(suit),
                        confidence=0.7,
                        position=(0, 0)
                    )
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Erreur analyse région carte: {e}")
            return None

    def _detect_cards_by_color(self, image: np.ndarray) -> List[Card]:
        """
        Détecte les cartes par analyse de couleur (rouge/noir)
        """
        try:
            cards = []
            
            # Conversion en HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Détection du rouge (♥, ♦)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            # Détection du noir (♠, ♣)
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 30])
            mask_black = cv2.inRange(hsv, lower_black, upper_black)
            
            # Si on détecte des couleurs de cartes, essayer OCR
            if cv2.countNonZero(mask_red) > 100 or cv2.countNonZero(mask_black) > 100:
                # OCR sur l'image originale
                ocr_cards = self._detect_cards_ocr_optimized(image)
                return ocr_cards
            
            return []
            
        except Exception as e:
            self.logger.error(f"Erreur détection par couleur: {e}")
            return []

    def _deduplicate_cards(self, cards: List[Card]) -> List[Card]:
        """
        Déduplique les cartes détectées
        """
        unique_cards = []
        seen = set()
        
        for card in cards:
            card_key = f"{card.rank}{card.suit}"
            if card_key not in seen:
                seen.add(card_key)
                unique_cards.append(card)
        
        return unique_cards

    def _normalize_suit(self, suit: str) -> str:
        """
        Normalise les symboles de couleur
        """
        suit_mapping = {
            'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣',
            '♠': '♠', '♥': '♥', '♦': '♦', '♣': '♣',
            '?': '?'
        }
        return suit_mapping.get(suit, '?')

    def _validate_card(self, card: Card) -> bool:
        """Valide une carte détectée"""
        try:
            # Vérifier que le rang est valide
            valid_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
            if card.rank not in valid_ranks:
                self.logger.debug(f"Rang invalide: {card.rank}")
                return False
            
            # Vérifier que la couleur est valide (incluant '?' pour inconnue)
            valid_suits = ['♠', '♥', '♦', '♣', '?']
            if card.suit not in valid_suits:
                self.logger.debug(f"Couleur invalide: {card.suit}")
                return False
            
            # Vérifier la confiance (seuil plus bas pour les cartes avec couleur inconnue)
            min_confidence = 0.05 if card.suit == '?' else 0.1
            if card.confidence < min_confidence:
                self.logger.debug(f"Confiance trop faible: {card.confidence}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Erreur validation carte: {e}")
            return False

    def extract_text(self, image: np.ndarray, region_name: str = "unknown") -> str:
        """
        Extrait du texte d'une image avec OCR
        """
        try:
            # Configuration OCR pour le texte général
            config = '--oem 3 --psm 6'
            
            # Prétraitement
            processed = self._preprocess_for_ocr(image)
            
            # Extraction
            text = pytesseract.image_to_string(processed, config=config)
            
            # Nettoyage
            clean_text = text.strip()
            
            self.logger.debug(f"Texte extrait de {region_name}: '{clean_text}'")
            return clean_text
            
        except Exception as e:
            self.logger.error(f"Erreur extraction texte: {e}")
            return ""

    def detect_chips(self, image: np.ndarray) -> List[int]:
        """
        Détecte les jetons dans une image
        """
        try:
            text = self.extract_text(image, "chips")
            
            # Chercher les nombres
            import re
            numbers = re.findall(r'\d+', text)
            
            chips = []
            for num in numbers:
                try:
                    chips.append(int(num))
                except ValueError:
                    continue
            
            self.logger.debug(f"Jetons détectés: {chips}")
            return chips
            
        except Exception as e:
            self.logger.error(f"Erreur détection jetons: {e}")
            return [] 