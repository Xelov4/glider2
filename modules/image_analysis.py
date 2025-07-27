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
from typing import List, Optional, Tuple
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
        Détecte les cartes en utilisant uniquement l'OCR
        """
        try:
            self.logger.debug(f"Détection de cartes dans {region_name} - OCR uniquement")
            
            # Prétraitement de l'image pour améliorer l'OCR
            processed_image = self._preprocess_for_ocr(image)
            
            # Extraction du texte avec OCR
            text = pytesseract.image_to_string(
                processed_image, 
                config=self.ocr_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Analyse du texte extrait
            detected_cards = self._parse_ocr_text(text['text'])
            
            # Validation des cartes détectées
            valid_cards = []
            for card in detected_cards:
                if self._validate_card(card):
                    valid_cards.append(card)
            
            self.logger.info(f"Cartes détectées par OCR: {[f'{c.rank}{c.suit}' for c in valid_cards]}")
            return valid_cards
            
        except Exception as e:
            self.logger.error(f"Erreur détection cartes OCR: {e}")
            return []

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Prétraitement de l'image pour optimiser l'OCR
        """
        try:
            # Conversion en niveaux de gris
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Amélioration du contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Réduction du bruit
            denoised = cv2.medianBlur(enhanced, 3)
            
            # Binarisation adaptative
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            return binary
            
        except Exception as e:
            self.logger.error(f"Erreur prétraitement OCR: {e}")
            return image

    def _parse_ocr_text(self, text: str) -> List[Card]:
        """
        Parse le texte OCR pour extraire les cartes
        """
        cards = []
        try:
            # Nettoyer le texte
            clean_text = text.strip().upper()
            self.logger.debug(f"Texte OCR brut: '{clean_text}'")
            
            # Chercher les patterns de cartes
            # Format attendu: "AS KH QD JC" ou "A♠ K♥ Q♦ J♣"
            words = clean_text.split()
            
            for word in words:
                if len(word) >= 2:
                    # Extraire rang et couleur
                    rank = word[0]
                    suit = word[1] if len(word) > 1 else '?'
                    
                    # Corriger les erreurs OCR courantes
                    rank = self.ocr_mapping.get(rank, rank)
                    suit = self._normalize_suit(suit)
                    
                    # Créer l'objet Card
                    card = Card(
                        rank=rank,
                        suit=suit,
                        confidence=0.8,  # Confiance par défaut pour OCR
                        position=(0, 0)
                    )
                    
                    cards.append(card)
            
            self.logger.debug(f"Cartes parsées: {[f'{c.rank}{c.suit}' for c in cards]}")
            return cards
            
        except Exception as e:
            self.logger.error(f"Erreur parsing OCR: {e}")
            return []

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