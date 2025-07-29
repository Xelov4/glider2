"""
Module d'analyse d'images pour l'agent IA Poker

Fonctionnalités:
- Detection Template Matching des cartes
- Detection OCR des cartes
- OCR robuste pour les montants
- Validation automatique des résultats
- Gestion d'erreurs complète
"""

import cv2
import numpy as np
import pytesseract
import os
import re
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import logging

@dataclass
class Card:
    rank: str
    suit: str
    confidence: float
    position: Tuple[int, int]

class ImageAnalyzer:
    """
    Analyseur d'images pour la détection de cartes et OCR
    """
    
    def __init__(self, tesseract_path: str = None):
        self.logger = logging.getLogger(__name__)
        
        # Configuration Tesseract
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Mapping OCR pour corriger les erreurs courantes
        self.ocr_mapping = {
            '0': 'O',
            '1': 'I',
            'l': 'I',
            'I': 'I',
            'O': '0',
            'S': '5',
            'G': '6',
            'B': '8'
        }
        
        # NOUVEAU: Cache intelligent pour performance
        self.image_cache = {}
        self.decision_cache = {}
        self.last_capture_time = 0
        self.cache_ttl = 0.05  # 50ms TTL pour le cache
        
        # NOUVEAU: Métriques de performance optimisées
        self.performance_metrics = {
            'capture_times': [],
            'decision_times': [],
            'ocr_times': [],
            'total_cycles': 0,
            'avg_cycle_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # NOUVEAU: Configuration de performance ultra-optimisée
        self.performance_config = {
            'capture_interval': 0.005,  # 5ms entre captures
            'decision_timeout': 0.02,  # 20ms max pour décision
            'cache_enabled': True,
            'parallel_processing': False,  # Désactivé pour stabilité
            'ultra_fast_mode': True,
            'max_cache_size': 100,  # Limite la taille du cache
            'memory_cleanup_interval': 100  # Nettoyage tous les 100 cycles
        }

    def detect_cards(self, image: np.ndarray, region_name: str = "hand_area") -> List[Card]:
        """
        Détecte les cartes dans une image avec le nouveau workflow
        """
        try:
            self.logger.debug(f"Detection cartes {region_name} - Image: {image.shape}")
            
            # NOUVEAU WORKFLOW: Template Matching + Validation, puis OCR + Color Detection
            
            # 1. TEMPLATE MATCHING + VALIDATION (priorité absolue)
            template_cards = self._detect_cards_template_matching(image)
            if template_cards:
                # Validation des cartes détectées par template matching
                validated_template_cards = []
                for card in template_cards:
                    if self._validate_card(card):
                        validated_template_cards.append(card)
                    else:
                        self.logger.debug(f"Carte template invalide: {card.rank}{card.suit} (conf: {card.confidence:.3f})")
                
                if validated_template_cards:
                    self.logger.debug(f"Template + Validation: {len(validated_template_cards)} cartes: {[f'{c.rank}{c.suit}' for c in validated_template_cards]}")
                    return validated_template_cards
                else:
                    self.logger.debug("Template matching echoue - passage a OCR + Color Detection")
            
            # 2. OCR + COLOR DETECTION (fallback systématique)
            self.logger.debug("Lancement OCR + Color Detection...")
            
            # 2a. Détection OCR
            ocr_cards = self._detect_cards_ocr_optimized(image)
            if ocr_cards:
                self.logger.debug(f"OCR detecte {len(ocr_cards)} cartes: {[f'{c.rank}{c.suit}' for c in ocr_cards]}")
                
                # 2b. Validation OCR + Amélioration par couleur
                validated_ocr_cards = []
                for card in ocr_cards:
                    if self._validate_card(card):
                        # Améliorer la couleur si nécessaire
                        if card.suit == '?':
                            improved_suit = self._determine_suit_by_position_and_color(image, card.rank, self._analyze_colors_ultra_fast(image))
                            if improved_suit != '?':
                                card.suit = improved_suit
                                self.logger.debug(f"Couleur amelioree pour {card.rank}: {improved_suit}")
                        
                        validated_ocr_cards.append(card)
                    else:
                        self.logger.debug(f"Carte OCR invalide: {card.rank}{card.suit}")
                
                if validated_ocr_cards:
                    self.logger.debug(f"OCR + Color + Validation: {len(validated_ocr_cards)} cartes: {[f'{c.rank}{c.suit}' for c in validated_ocr_cards]}")
                    return validated_ocr_cards
            
            # 3. DÉTECTION PAR COULEURS SEULE (dernier recours)
            self.logger.debug("Lancement detection par couleurs seule...")
            color_cards = self._detect_cards_by_color(image)
            if color_cards:
                validated_color_cards = []
                for card in color_cards:
                    if self._validate_card(card):
                        validated_color_cards.append(card)
                
                if validated_color_cards:
                    self.logger.debug(f"Couleurs + Validation: {len(validated_color_cards)} cartes: {[f'{c.rank}{c.suit}' for c in validated_color_cards]}")
                    return validated_color_cards
            
            self.logger.debug(f"Aucune carte valide detectee dans {region_name}")
            return []
            
        except Exception as e:
            self.logger.error(f"Erreur detection cartes: {e}")
            return []

    def _detect_cards_template_matching(self, image: np.ndarray) -> List[Card]:
        """
        Détecte les cartes avec template matching complet (52 cartes)
        """
        try:
            cards = []
            
            # Conversion en niveaux de gris
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Charger les templates avec noms ASCII
            fixed_dir = "templates/cards/fixed"
            if not os.path.exists(fixed_dir):
                self.logger.debug("Dossier templates fixés non trouvé")
                return []
            
            # Mapping des couleurs
            suit_mapping = {
                'spades': '♠',
                'hearts': '♥', 
                'diamonds': '♦',
                'clubs': '♣'
            }
            
            # Toutes les cartes disponibles
            all_cards = [
                # Piques
                ('A', 'spades'), ('2', 'spades'), ('3', 'spades'), ('4', 'spades'), ('5', 'spades'),
                ('6', 'spades'), ('7', 'spades'), ('8', 'spades'), ('9', 'spades'), ('T', 'spades'),
                ('J', 'spades'), ('Q', 'spades'), ('K', 'spades'),
                # Cœurs
                ('A', 'hearts'), ('2', 'hearts'), ('3', 'hearts'), ('4', 'hearts'), ('5', 'hearts'),
                ('6', 'hearts'), ('7', 'hearts'), ('8', 'hearts'), ('9', 'hearts'), ('T', 'hearts'),
                ('J', 'hearts'), ('Q', 'hearts'), ('K', 'hearts'),
                # Carreaux
                ('A', 'diamonds'), ('2', 'diamonds'), ('3', 'diamonds'), ('4', 'diamonds'), ('5', 'diamonds'),
                ('6', 'diamonds'), ('7', 'diamonds'), ('8', 'diamonds'), ('9', 'diamonds'), ('T', 'diamonds'),
                ('J', 'diamonds'), ('Q', 'diamonds'), ('K', 'diamonds'),
                # Trèfles
                ('A', 'clubs'), ('2', 'clubs'), ('3', 'clubs'), ('4', 'clubs'), ('5', 'clubs'),
                ('6', 'clubs'), ('7', 'clubs'), ('8', 'clubs'), ('9', 'clubs'), ('T', 'clubs'),
                ('J', 'clubs'), ('Q', 'clubs'), ('K', 'clubs')
            ]
            
            # Template matching pour chaque carte
            for rank, suit_ascii in all_cards:
                suit_symbol = suit_mapping[suit_ascii]
                
                # Chemin vers le template
                template_path = os.path.join(fixed_dir, suit_ascii, f"card_{rank}_{suit_ascii}.png")
                
                if os.path.exists(template_path):
                    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                    if template is not None:
                        # Template matching avec différentes échelles
                        scale_factors = [0.5, 0.7, 1.0, 1.3, 1.6, 2.0]
                        
                        for scale in scale_factors:
                            h, w = template.shape
                            new_h, new_w = int(h * scale), int(w * scale)
                            if new_h < 10 or new_w < 10:
                                continue
                            
                            resized_template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                            
                            # Template matching avec seuil plus strict
                            result = cv2.matchTemplate(gray, resized_template, cv2.TM_CCOEFF_NORMED)
                            locations = np.where(result >= 0.7)  # Seuil augmenté pour plus de précision
                            
                            for pt in zip(*locations[::-1]):
                                confidence = result[pt[1], pt[0]]
                                
                                # Vérifier si c'est un nouveau match (éviter les doublons)
                                is_new_match = True
                                for existing_card in cards:
                                    if (abs(existing_card.position[0] - pt[0]) < 50 and 
                                        abs(existing_card.position[1] - pt[1]) < 50):
                                        is_new_match = False
                                        break
                                
                                if is_new_match:
                                    card = Card(
                                        rank=rank,
                                        suit=suit_symbol,
                                        confidence=confidence,
                                        position=pt
                                    )
                                    
                                    # Validation immédiate de la carte
                                    if self._validate_card(card):
                                        cards.append(card)
                                        self.logger.debug(f"Template match valide: {rank}{suit_symbol} (conf: {confidence:.3f})")
                                    else:
                                        self.logger.debug(f"Template match rejete: {rank}{suit_symbol} (conf: {confidence:.3f})")
            
            # Dédupliquer les cartes
            cards = self._deduplicate_cards(cards)
            
            self.logger.debug(f"Template matching détecté {len(cards)} cartes")
            return cards
            
        except Exception as e:
            self.logger.error(f"Erreur template matching: {e}")
            return []



    def _detect_cards_ultra_fast(self, image: np.ndarray) -> List[Card]:
        """
        Détection ultra-rapide des cartes (méthode la plus efficace)
        """
        try:
        cards = []
        
            # NOUVEAU: OCR ultra-rapide avec configuration optimisée
            config = '--oem 3 --psm 6 -c tessedit_char_whitelist=23456789TJQKA'
            
            # Prétraitement minimal
            if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Amélioration du contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # OCR
            text = pytesseract.image_to_string(enhanced, config=config)
            clean_text = text.strip().upper()
            
            self.logger.debug(f"Texte OCR ultra-rapide: '{clean_text}'")
            
            # NOUVEAU: Parser intelligent du texte
            detected_ranks = []
            for char in clean_text:
                if char in '23456789TJQKA':
                    detected_ranks.append(char)
            
            # NOUVEAU: Détection de couleur par analyse d'image
            if detected_ranks:
                # Analyser les couleurs dans l'image
                color_analysis = self._analyze_colors_ultra_fast(image)
                
                # Associer rangs et couleurs
                for rank in detected_ranks:
                    # Déterminer la couleur basée sur la position et l'analyse
                    suit = self._determine_suit_by_position_and_color(image, rank, color_analysis)
                    
                    card = Card(
                        rank=rank,
                        suit=suit,
                        confidence=0.8,  # Confiance élevée pour l'OCR
                        position=(0, 0)
                    )
                    cards.append(card)
                    self.logger.debug(f"Carte détectée ultra-rapide: {rank}{suit}")
            
        return cards
    
        except Exception as e:
            self.logger.error(f"Erreur détection ultra-rapide: {e}")
            return []

    def _analyze_colors_ultra_fast(self, image: np.ndarray) -> Dict:
        """
        Analyse ultra-rapide des couleurs dans l'image
        """
        try:
            # Conversion HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Masques pour rouge et noir
            # Rouge (♥, ♦)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            # Noir (♠, ♣)
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 50])
            mask_black = cv2.inRange(hsv, lower_black, upper_black)
            
            # Compter les pixels
            red_pixels = cv2.countNonZero(mask_red)
            black_pixels = cv2.countNonZero(mask_black)
            total_pixels = image.shape[0] * image.shape[1]
            
            return {
                'red_pixels': red_pixels,
                'black_pixels': black_pixels,
                'total_pixels': total_pixels,
                'red_ratio': red_pixels / total_pixels if total_pixels > 0 else 0,
                'black_ratio': black_pixels / total_pixels if total_pixels > 0 else 0
            }
            
        except Exception as e:
            self.logger.debug(f"Erreur analyse couleurs ultra-rapide: {e}")
            return {'red_pixels': 0, 'black_pixels': 0, 'total_pixels': 1, 'red_ratio': 0, 'black_ratio': 0}

    def _determine_suit_by_position_and_color(self, image: np.ndarray, rank: str, color_analysis: Dict) -> str:
        """
        Détermine la couleur d'une carte basée sur la position et l'analyse de couleur
        """
        try:
            # NOUVEAU: Logique intelligente basée sur les ratios de couleur
            red_ratio = color_analysis['red_ratio']
            black_ratio = color_analysis['black_ratio']
            
            # Seuils pour détecter les couleurs
            color_threshold = 0.05  # 5% de pixels colorés
            
            if red_ratio > color_threshold:
                # Rouge détecté - déterminer ♥ ou ♦
                if red_ratio > 0.1:  # Beaucoup de rouge
                    return '♥'
                else:
                    return '♦'
            elif black_ratio > color_threshold:
                # Noir détecté - déterminer ♠ ou ♣
                if black_ratio > 0.1:  # Beaucoup de noir
                    return '♠'
                else:
                    return '♣'
            else:
                # Pas de couleur claire détectée
                # NOUVEAU: Heuristique basée sur le rang
                if rank in ['A', 'K', 'Q', 'J']:
                    # Figures - plus souvent noires
                    return '♠'
                elif rank in ['T', '9', '8']:
                    # 10-8 - plus souvent rouges
                    return '♥'
                else:
                    # Chiffres - alterner
                    return '♣'
                
        except Exception as e:
            self.logger.debug(f"Erreur détermination couleur: {e}")
            return '?'  # Couleur inconnue

    def _detect_cards_ocr_optimized(self, image: np.ndarray) -> List[Card]:
        """
        OCR optimisé pour les cartes avec intégration couleur
        """
        cards = []
        
        # NOUVEAU: Analyse des couleurs en premier
        color_analysis = self._analyze_colors_ultra_fast(image)
        self.logger.debug(f"Analyse couleurs: Rouge={color_analysis['red_ratio']:.3f}, Noir={color_analysis['black_ratio']:.3f}")
        
        # Configuration 1: OCR standard
        config1 = '--oem 3 --psm 6 -c tessedit_char_whitelist=23456789TJQKA♠♥♦♣'
        cards.extend(self._try_ocr_config(image, config1))
        
        # Configuration 2: OCR pour texte dense
        config2 = '--oem 3 --psm 8 -c tessedit_char_whitelist=23456789TJQKA♠♥♦♣'
        cards.extend(self._try_ocr_config(image, config2))
        
        # Configuration 3: OCR pour caractères individuels
        config3 = '--oem 3 --psm 10 -c tessedit_char_whitelist=23456789TJQKA♠♥♦♣'
        cards.extend(self._try_ocr_config(image, config3))
        
        # NOUVEAU: Amélioration des couleurs pour les cartes détectées
        improved_cards = []
        for card in cards:
            # Si la couleur est inconnue ou douteuse, essayer de l'améliorer
            if card.suit == '?' or card.suit not in ['♠', '♥', '♦', '♣']:
                improved_suit = self._determine_suit_by_position_and_color(image, card.rank, color_analysis)
                if improved_suit != '?':
                    card.suit = improved_suit
                    self.logger.debug(f"Couleur amelioree pour {card.rank}: {improved_suit}")
            
            improved_cards.append(card)
        
        # Dédupliquer et valider
        unique_cards = self._deduplicate_cards(improved_cards)
        validated_cards = []
        
        for card in unique_cards:
            if self._validate_card(card):
                validated_cards.append(card)
            else:
                self.logger.debug(f"Carte OCR rejetee: {card.rank}{card.suit}")
        
        self.logger.debug(f"OCR + Color: {len(validated_cards)} cartes validees")
        return validated_cards

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
        Détecte les cartes par analyse de couleur (rouge/noir) avec validation
        """
        try:
            self.logger.debug("Lancement detection par couleurs...")
            cards = []
            
            # NOUVEAU: Analyse des couleurs ultra-rapide
            color_analysis = self._analyze_colors_ultra_fast(image)
            self.logger.debug(f"Analyse couleurs: Rouge={color_analysis['red_ratio']:.3f}, Noir={color_analysis['black_ratio']:.3f}")
            
            # Conversion en HSV pour meilleure détection des couleurs
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Détection du rouge (♥, ♦) - deux plages HSV
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
            
            # Analyser les régions de couleur
            red_regions = self._find_color_regions(mask_red, 'red')
            black_regions = self._find_color_regions(mask_black, 'black')
            
            self.logger.debug(f"Regions detectees: Rouge={len(red_regions)}, Noir={len(black_regions)}")
            
            # NOUVEAU: Détection OCR simple pour les rangs
            ocr_text = pytesseract.image_to_string(
                self._preprocess_for_ocr(image),
                config='--oem 3 --psm 6 -c tessedit_char_whitelist=23456789TJQKA'
            )
            
            # Extraire les rangs détectés
            detected_ranks = []
            for char in ocr_text.strip().upper():
                if char in '23456789TJQKA':
                    detected_ranks.append(char)
            
            self.logger.debug(f"Rangs detectes: {detected_ranks}")
            
            # NOUVEAU: Créer des cartes basées sur les couleurs et rangs
            for rank in detected_ranks:
                # Déterminer la couleur basée sur l'analyse
                if color_analysis['red_ratio'] > 0.05:  # Rouge détecté
                    if color_analysis['red_ratio'] > 0.1:
                        suit = '♥'
            else:
                        suit = '♦'
                elif color_analysis['black_ratio'] > 0.05:  # Noir détecté
                    if color_analysis['black_ratio'] > 0.1:
                        suit = '♠'
                    else:
                        suit = '♣'
                else:
                    # Couleur indéterminée - utiliser heuristique
                    suit = self._determine_suit_by_position_and_color(image, rank, color_analysis)
                
                card = Card(
                    rank=rank,
                    suit=suit,
                    confidence=0.6,  # Confiance plus faible pour détection par couleur
                    position=(0, 0)
                )
                
                # Validation de la carte
                if self._validate_card(card):
                    cards.append(card)
                    self.logger.debug(f"Carte couleur validee: {rank}{suit}")
                else:
                    self.logger.debug(f"Carte couleur rejetee: {rank}{suit}")
            
            self.logger.debug(f"Detection couleurs: {len(cards)} cartes validees")
            return cards
            
        except Exception as e:
            self.logger.error(f"Erreur détection par couleur: {e}")
            return []

    def _find_color_regions(self, mask: np.ndarray, color_type: str) -> List[Dict]:
        """
        Trouve les régions de couleur spécifique dans l'image
        """
        regions = []
        try:
            # Recherche de contours dans le masque
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Filtre les petits contours
                    x, y, w, h = cv2.boundingRect(contour)
                    regions.append({
                        'area': area,
                        'position': (x, y),
                        'size': (w, h),
                        'color_type': color_type
                    })
            
            return regions
            
        except Exception as e:
            self.logger.error(f"Erreur recherche régions couleur: {e}")
            return []

    def _associate_colors_with_ranks(self, ocr_cards: List[Card], red_regions: List[Dict], 
                                   black_regions: List[Dict], image: np.ndarray) -> List[Card]:
        """
        Associe les couleurs détectées avec les rangs OCR
        """
        try:
            cards = []
            
            for card in ocr_cards:
                # Trouver la meilleure correspondance de couleur
                best_suit = self._find_best_color_match(card, red_regions, black_regions, image)
                
                if best_suit:
                    card.suit = best_suit
                    cards.append(card)
                else:
                    # Garder la carte même sans couleur
                    cards.append(card)
            
            return cards
                
        except Exception as e:
            self.logger.error(f"Erreur association couleurs: {e}")
            return ocr_cards
            
    def _find_best_color_match(self, card: Card, red_regions: List[Dict], 
                              black_regions: List[Dict], image: np.ndarray) -> Optional[str]:
        """
        Trouve la meilleure correspondance de couleur pour une carte
        """
        try:
            # Analyser la région autour de la carte
            card_region = self._get_card_region(card, image)
            if card_region is None:
                return None
            
            # Compter les pixels rouges et noirs dans la région
            hsv_region = cv2.cvtColor(card_region, cv2.COLOR_BGR2HSV)
            
            # Masques pour rouge et noir
            mask_red = cv2.inRange(hsv_region, np.array([0, 100, 100]), np.array([10, 255, 255]))
            mask_red2 = cv2.inRange(hsv_region, np.array([170, 100, 100]), np.array([180, 255, 255]))
            mask_red_total = cv2.bitwise_or(mask_red, mask_red2)
            
            mask_black = cv2.inRange(hsv_region, np.array([0, 0, 0]), np.array([180, 255, 30]))
            
            red_pixels = cv2.countNonZero(mask_red_total)
            black_pixels = cv2.countNonZero(mask_black)
            total_pixels = card_region.shape[0] * card_region.shape[1]
            
            # Déterminer la couleur dominante
            if red_pixels > black_pixels and red_pixels > total_pixels * 0.1:
                # Déterminer si ♥ ou ♦ basé sur la forme
                return self._determine_red_suit(card_region)
            elif black_pixels > red_pixels and black_pixels > total_pixels * 0.1:
                # Déterminer si ♠ ou ♣ basé sur la forme
                return self._determine_black_suit(card_region)
            
            return None
                
        except Exception as e:
            self.logger.debug(f"Erreur correspondance couleur: {e}")
            return None
    
    def _get_card_region(self, card: Card, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrait la région d'une carte depuis l'image
        """
        try:
            # Pour l'instant, utiliser une région par défaut
            # TODO: Améliorer avec la position réelle de la carte
            height, width = image.shape[:2]
            region_size = min(width, height) // 4
            
            # Région centrale par défaut
            x = width // 4
            y = height // 4
            w = region_size
            h = region_size
            
            return image[y:y+h, x:x+w]
            
        except Exception as e:
            self.logger.debug(f"Erreur extraction région carte: {e}")
            return None
    
    def _determine_red_suit(self, card_region: np.ndarray) -> str:
        """
        Détermine si c'est ♥ ou ♦ basé sur la forme
        """
        try:
            # Conversion en niveaux de gris
            gray = cv2.cvtColor(card_region, cv2.COLOR_BGR2GRAY)
            
            # Seuillage pour isoler la forme
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Recherche de contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:
                    # Analyser la forme
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area
                    
                    # ♥ a généralement une solidité plus élevée (forme pleine)
                    if solidity > 0.9:
                        return '♥'
                    else:
                        return '♦'
            
            # Par défaut, retourner ♥
            return '♥'
            
        except Exception as e:
            self.logger.debug(f"Erreur détermination rouge: {e}")
            return '♥'
    
    def _determine_black_suit(self, card_region: np.ndarray) -> str:
        """
        Détermine si c'est ♠ ou ♣ basé sur la forme
        """
        try:
            # Conversion en niveaux de gris
            gray = cv2.cvtColor(card_region, cv2.COLOR_BGR2GRAY)
            
            # Seuillage pour isoler la forme
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Recherche de contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:
                    # Analyser la forme
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area
                    
                    # ♠ a généralement une solidité plus élevée (forme pleine)
                    if solidity > 0.9:
                        return '♠'
                    else:
                        return '♣'
            
            # Par défaut, retourner ♠
            return '♠'
                
        except Exception as e:
            self.logger.debug(f"Erreur détermination noir: {e}")
            return '♠'

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